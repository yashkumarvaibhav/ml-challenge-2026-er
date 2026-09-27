"""Independent structural and pair-diff audit of Claude's #27-17 TSV."""
import json
from pathlib import Path
import polars as pl

R = Path(__file__).resolve().parents[1]


def pairs(i, kind="matching_results"):
    p = R / f"submissions/archive/sub-2026-09-27-{i}/{kind}.tsv"
    m = pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0)
    a, b = m.columns
    return m, (m.select(pl.col(a).alias("s1"), pl.col(b).fill_null("").str.split(",").alias("m"))
               .explode("m").filter(pl.col("m") != ""))


def main():
    m16, b16 = pairs(16)
    m17, b17 = pairs(17)
    _, cand = pairs(17, "candidate_pairs")
    restored = b17.join(b16, on=["s1", "m"], how="anti")
    new_drops = b16.join(b17, on=["s1", "m"], how="anti")
    absent_cand = b17.join(cand, on=["s1", "m"], how="anti")
    owners = b17.group_by("m").len().filter(pl.col("len") > 1)
    caps = b17.with_columns(pl.col("m").str.starts_with("S2-").alias("s2")).group_by("s1").agg(
        pl.col("s2").sum().alias("s2"), (~pl.col("s2")).sum().alias("s3"))
    bad_caps = caps.filter((pl.col("s2") > 5) | (pl.col("s3") > 6))
    c = pl.read_parquet(R / "cache/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1", "country": "c1"})
    t = pl.concat([pl.read_parquet(R / f"cache/test_source{i}.parquet", columns=["entity_id", "country"])
                   for i in (2, 3)]).rename({"entity_id": "m", "country": "cm"})
    bad_country = b17.join(c, on="s1").join(t, on="m").filter(pl.col("c1") != pl.col("cm"))
    details = pl.read_csv(R / "handoff/codex_drop_detail16.tsv", separator="\t")
    recovered_by_source = restored.join(details.select("s1", "m", "country", "source", "k16", "f_assumed"), on=["s1", "m"], how="left")
    neg = pl.read_csv(R / "handoff/codex_negative16.tsv", separator="\t")
    rr = recovered_by_source.join(neg.select("s1"), on="s1", how="left", suffix="_neg")
    # Compare row order, not just pair membership: submission format remains aligned.
    same_s1_order = m16[m16.columns[0]].equals(m17[m17.columns[0]])
    out = {
        "matching_rows": m17.height, "matching_pairs": b17.height, "same_s1_order_as_16": same_s1_order,
        "restored_pairs_vs_16": restored.height, "new_drops_vs_16": new_drops.height,
        "restored_s1s": restored["s1"].n_unique(), "restored_from_11_drops": recovered_by_source.filter(pl.col("source").is_not_null()).height,
        "restored_by_source": recovered_by_source.group_by("source", "country").agg(pl.len().alias("pairs"), pl.col("s1").n_unique().alias("s1s")).sort("source", "country").to_dicts(),
        "restored_on_our_negative_s1s": restored.join(neg.select("s1"), on="s1").height,
        "restored_negative_s1s": restored.join(neg.select("s1"), on="s1")["s1"].n_unique(),
        "restored_nonnegative_pairs": restored.join(neg.select("s1"), on="s1", how="anti").height,
        "new_drop_samples": new_drops.head(5).to_dicts(),
        "multiply_owned_records": owners.height, "source_cap_violations": bad_caps.height,
        "source_cap_samples": bad_caps.head(10).to_dicts(),
        "max_s2": caps["s2"].max(), "max_s3": caps["s3"].max(),
        "country_mismatches": bad_country.height, "matches_absent_from_candidates": absent_cand.height,
    }
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
