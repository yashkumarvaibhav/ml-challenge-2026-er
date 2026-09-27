"""Official metric and blocking reports (PLAN.md §2, V4, B2).

Per S1: F0.5 = 1.25*TP / (0.25*|truth| + |pred|); an empty prediction scores 1 on a
true singleton and 0 otherwise. Macro average over the evaluated S1 set.
"""
import polars as pl

TEST_MIX = {"US": 0.45, "India": 0.55}  # known-country shares of test S1 (PLAN V4)


def per_s1_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """pred/truth: (s1, m) pairs; s1: evaluated S1s with columns (s1, country)."""
    t = truth.join(s1.select("s1"), on="s1").group_by("s1").len().rename({"len": "T"})
    p = pred.join(s1.select("s1"), on="s1").group_by("s1").len().rename({"len": "P"})
    tp = pred.join(truth, on=["s1", "m"]).join(s1.select("s1"), on="s1").group_by("s1").len().rename({"len": "TP"})
    d = s1.join(t, on="s1", how="left").join(p, on="s1", how="left").join(tp, on="s1", how="left").fill_null(0)
    return d.with_columns(
        pl.when(pl.col("T") == 0)
        .then((pl.col("P") == 0).cast(pl.Float64))
        .otherwise(1.25 * pl.col("TP") / (0.25 * pl.col("T") + pl.col("P")))
        .alias("f05")
    )


def summarize(per: pl.DataFrame) -> dict:
    by = {c: round(v, 5) for c, v in per.group_by("country").agg(pl.col("f05").mean()).iter_rows()}
    out = {"macro_f05": round(per["f05"].mean(), 5), "by_country": by, "n_s1": per.height}
    if all(c in by for c in TEST_MIX):
        out["reweighted_us45_in55"] = round(sum(TEST_MIX[c] * by[c] for c in TEST_MIX), 5)
    return out


def blocking_report(cand: pl.DataFrame, truth: pl.DataFrame, s1: pl.DataFrame, mflags: pl.DataFrame) -> dict:
    """cand: (s1, m); truth: (s1, m); s1: (s1, country); mflags: (m, n_native, a_null)."""
    tr = truth.join(s1.select("s1"), on="s1")
    hit = tr.join(cand.select("s1", "m").with_columns(pl.lit(True).alias("hit")), on=["s1", "m"], how="left") \
        .with_columns(pl.col("hit").fill_null(False))
    hit = hit.join(mflags, on="m", how="left")
    ent = s1.join(tr.group_by("s1").len().rename({"len": "T"}), on="s1", how="left").fill_null(0) \
        .join(hit.group_by("s1").agg(pl.col("hit").sum().alias("R")), on="s1", how="left").fill_null(0)
    ns = ent.filter(pl.col("T") > 0)
    oracle = per_s1_f05(hit.filter(pl.col("hit")).select("s1", "m"), tr.select("s1", "m"), s1)
    npairs = cand.join(s1.select("s1"), on="s1").height

    def seg(expr):
        h = hit.filter(expr)
        return {"pairs": h.height, "recall": round(h["hit"].mean(), 4) if h.height else None}

    rep = {
        "pair_recall": round(hit["hit"].mean(), 4),
        "nonsingleton_all_retrieved": round((ns["R"] == ns["T"]).mean(), 4),
        "nonsingleton_none_retrieved": round((ns["R"] == 0).mean(), 4),
        "oracle": summarize(oracle),
        "pairs_per_s1": round(npairs / max(1, s1.height), 2),
        "segment_native_name": seg(pl.col("n_native")),
        "segment_address_null": seg(pl.col("a_null")),
        "by_country_recall": {c: round(v, 4) for c, v in
                              hit.join(s1, on="s1").group_by("country").agg(pl.col("hit").mean()).iter_rows()},
    }
    return rep
