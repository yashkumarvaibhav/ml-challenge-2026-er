"""Independent, exact per-S1 expected-F audit of #27-16 removals vs #27-11.

Only reads existing artifacts. Writes reports under handoff/.
"""
import json
from collections import defaultdict
from pathlib import Path

import polars as pl

R = Path(__file__).resolve().parents[1]
ARCH = R / "submissions/archive"


def pairs(i):
    path = ARCH / f"sub-2026-09-27-{i}/matching_results.tsv"
    frame = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0)
    a, b = frame.columns
    return (frame.select(pl.col(a).alias("s1"), pl.col(b).fill_null("").str.split(",").alias("m"))
            .explode("m").filter(pl.col("m") != "").select("s1", "m"))


def score(tp, truth, pred):
    if truth == 0 and pred == 0:
        return 1.0
    if pred == 0:
        return 0.0
    return 1.25 * tp / (0.25 * truth + pred)


def expected_delta(k_current, false_probabilities):
    """Marginal value of removing the listed pairs from a current k prediction.

    Assumes current predictions true and no unseen true records. Unlike a binomial
    approximation, this convolves distinct false probabilities for mixed drops.
    """
    dist = [1.0]
    for f in false_probabilities:
        new = [0.0] * (len(dist) + 1)
        for j, p in enumerate(dist):
            new[j] += p * f
            new[j + 1] += p * (1.0 - f)
        dist = new
    d = len(false_probabilities)
    return sum(p * (score(k_current, k_current + j, k_current)
                    - score(k_current + j, k_current + j, k_current + d))
               for j, p in enumerate(dist))


def main():
    B = {i: pairs(i) for i in range(11, 17)}
    final = B[16]
    drops = B[11].join(final, on=["s1", "m"], how="anti")
    adds = final.join(B[11], on=["s1", "m"], how="anti")
    for i in range(12, 17):
        drops = drops.join(B[i].with_columns(pl.lit(True).alias(f"p{i}")), on=["s1", "m"], how="left")
        drops = drops.with_columns(pl.col(f"p{i}").fill_null(False))
    drops = drops.with_columns(
        pl.when(pl.col("p15") & ~pl.col("p16")).then(16)
        .when(pl.col("p14") & ~pl.col("p15")).then(15)
        .when(pl.col("p13") & ~pl.col("p14")).then(14)
        .when(pl.col("p12") & ~pl.col("p13")).then(13)
        .otherwise(12).alias("last_removed_stage")
    )
    c = pl.read_parquet(R / "cache/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    p2 = pl.read_parquet(R / "cache/v6/s2/test_pred_e5base2.parquet").rename({"p": "p2"})
    p3 = pl.read_parquet(R / "cache/v6/s2/test_pred_e5base2_llm3v4ab_qwen7b_r3.parquet").rename({"p": "p3"})
    drops = drops.join(c, on="s1").join(p2, on=["s1", "m"], how="left").join(p3, on=["s1", "m"], how="left")
    drops = drops.with_columns(
        pl.when(pl.col("last_removed_stage") == 12).then(pl.lit("france_v4"))
        .when(pl.col("last_removed_stage") == 13).then(pl.lit("owner_or_cap"))
        .when(pl.col("last_removed_stage") == 14).then(pl.lit("refit_or_candidate"))
        .when(pl.col("last_removed_stage") == 15).then(pl.lit("france_house_tail"))
        .when(pl.col("last_removed_stage") == 16).then(pl.lit("blend"))
        .otherwise(pl.lit("unclassified")).alias("source")
    )
    # Published working assumptions: #12 low-p2 false 0.80, other #12 false
    # 0.60; France house tail 0.40; blend 0.35/0.45 by p3 band.
    drops = drops.with_columns(
        pl.when((pl.col("source") == "france_v4") & (pl.col("p2") <= 0.5)).then(0.80)
        .when(pl.col("source") == "france_v4").then(0.60)
        .when(pl.col("source") == "france_house_tail").then(0.40)
        .when((pl.col("source") == "blend") & (pl.col("p3") > 0.75)).then(0.35)
        .when(pl.col("source") == "blend").then(0.45)
        .otherwise(None).alias("f_assumed")
    )
    k = final.group_by("s1").len().rename({"len": "k16"})
    drops = drops.join(k, on="s1", how="left").with_columns(pl.col("k16").fill_null(0))
    drops.select("s1", "m", "country", "source", "p2", "p3", "f_assumed", "k16").write_csv(
        R / "handoff/codex_drop_detail16.tsv", separator="\t"
    )
    known = defaultdict(list)
    missing = defaultdict(list)
    for row in drops.select("s1", "m", "source", "f_assumed", "k16").iter_rows(named=True):
        (known if row["f_assumed"] is not None else missing)[row["s1"]].append(row)
    neg_rows = []
    by_source = defaultdict(lambda: {"s1": set(), "pairs": 0, "negative_s1": 0, "negative_pairs": 0, "sum_dF": 0.0})
    for row in drops.select("s1", "source").iter_rows(named=True):
        d = by_source[row["source"]]
        d["s1"].add(row["s1"])
        d["pairs"] += 1
    sum_delta = 0.0
    negative_count = 0
    for s1, rows in known.items():
        k16 = rows[0]["k16"]
        delta = expected_delta(k16, [r["f_assumed"] for r in rows])
        sum_delta += delta
        if delta < -1e-12:
            negative_count += 1
            neg_rows.append({"s1": s1, "k16": k16, "n_drops": len(rows),
                             "drop_sources": ",".join(sorted({r["source"] for r in rows})),
                             "dF": delta, "restorable_pairs": ",".join(r["m"] for r in rows)})
            for source in {r["source"] for r in rows}:
                by_source[source]["negative_s1"] += 1
                by_source[source]["negative_pairs"] += sum(r["source"] == source for r in rows)
                by_source[source]["sum_dF"] += delta
    neg = pl.DataFrame(neg_rows) if neg_rows else pl.DataFrame(schema={
        "s1": pl.String, "k16": pl.Int64, "n_drops": pl.Int64, "drop_sources": pl.String,
        "dF": pl.Float64, "restorable_pairs": pl.String})
    neg.sort("dF").write_csv(R / "handoff/codex_negative16.tsv", separator="\t")
    summary = {
        "drops_vs_11": drops.height, "adds_vs_11": adds.height,
        "stage_counts": drops.group_by("source", "country").agg(pl.len().alias("pairs"), pl.col("s1").n_unique().alias("s1s")).sort("source", "country").to_dicts(),
        "f_assumed_coverage": drops.filter(pl.col("f_assumed").is_not_null()).height,
        "unestimated_drop_pairs": drops.filter(pl.col("f_assumed").is_null()).height,
        "unestimated_sources": {k: len(v) for k, v in missing.items()},
        "known_source_summary": {source: {**{k: v for k, v in d.items() if k != "s1"}, "s1": len(d["s1"])} for source, d in by_source.items()},
        "known_negative_s1_count": negative_count,
        "known_negative_drop_pairs": int(neg["n_drops"].sum()) if neg.height else 0,
        "known_total_dF": sum_delta,
        "known_public_delta_at_N_1917670": sum_delta / 1917670,
        "known_negative_k16_distribution": neg.group_by("k16").len().sort("k16").to_dicts() if neg.height else [],
        "known_negative_examples": neg.head(10).to_dicts(),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
