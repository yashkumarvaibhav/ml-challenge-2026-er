"""Label-free test diagnostic for the v3 `h` channel (reviewer re-audit, risk 2).

Per country, comparing v3 test candidates with v2's:
* candidates per S1 and share of S1s at the global cap (70)
* share of v3 candidates found ONLY by `h`
* v2 candidates dropped in v3, and how many of those were v2 *predicted matches*
  (p >= v2 thresholds): a label-free proxy for good candidates crowded out.

  python code/business_entity_resolution/src/ber/experiments/h_diagnostic.py
"""
import json

import polars as pl

from ber import config as C
from ber.blocking import CAP
from ber.model import decide

if __name__ == "__main__":
    s1 = pl.read_parquet(C.CACHE / "test_source1.parquet").select(pl.col("entity_id").alias("s1"), "country")
    v3 = pl.read_parquet(C.CACHE / "cand" / "test.parquet")
    v2 = pl.read_parquet(C.CACHE / "runs" / "v2" / "test_cand.parquet", columns=["s1", "m"])
    v2p = pl.read_parquet(C.CACHE / "runs" / "v2" / "test_pred.parquet")
    prm2 = json.loads((C.EXPERIMENTS / "v2" / "decision_params.json").read_text())
    v2pred = decide(v2p, prm2["t_any"], prm2["t_add"]).with_columns(pl.lit(True).alias("v2_pred"))
    others = ["r_a", "r_b", "r_c", "r_e", "r_f", "r_g"]
    v3 = v3.with_columns(
        (pl.col("r_h").is_not_null() & pl.all_horizontal([pl.col(c).is_null() for c in others]) & ~pl.col("in_d")).alias("h_only"))
    per_s1 = v3.group_by("s1").agg(pl.len().alias("n"), pl.col("h_only").sum().alias("n_h_only")).join(s1, on="s1")
    dropped = v2.join(v3.select("s1", "m"), on=["s1", "m"], how="anti").join(v2pred, on=["s1", "m"], how="left") \
        .with_columns(pl.col("v2_pred").fill_null(False)).join(s1, on="s1")
    added = v3.select("s1", "m").join(v2, on=["s1", "m"], how="anti").join(s1, on="s1")
    rep = (per_s1.group_by("country").agg(
        pl.col("n").mean().round(2).alias("cand_per_s1"),
        (pl.col("n") >= CAP).mean().round(4).alias("share_at_cap"),
        (pl.col("n_h_only").sum() / pl.col("n").sum()).round(4).alias("share_h_only"))
        .join(dropped.group_by("country").agg(pl.len().alias("v2_cands_dropped"),
                                              pl.col("v2_pred").sum().alias("v2_predicted_matches_dropped")), on="country", how="left")
        .join(added.group_by("country").len().rename({"len": "v3_cands_added"}), on="country", how="left")
        .join(v2pred.join(s1, on="s1").group_by("country").len().rename({"len": "v2_predicted_matches"}), on="country", how="left")
        .sort("country"))
    print(rep)
    (C.EXPERIMENTS / "v3" / "h_diagnostic.json").write_text(json.dumps(rep.to_dicts(), indent=1))
