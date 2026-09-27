"""Day-2 France A/B, variant B: stage 2 + CE decisions for every country EXCEPT the listed ones, which
keep the v5 decisions (v4 model on France-normalized test). US/India are identical to variant A, so
the public-score difference between A and B isolates France.

  python code/business_entity_resolution/src/ber/experiments/write_hybrid.py France
"""
import json
import sys

import polars as pl

from ber import config as C
from ber.io import load_source
from ber.model import decide
from ber.progress import step

if __name__ == "__main__":
    keep_v5 = sys.argv[1:] or ["France"]
    s1 = load_source("test", 1).select(pl.col("entity_id").alias("s1"), "country")
    pa = json.loads((C.EXPERIMENTS / "decision_params_s2ce.json").read_text())
    pb = json.loads((C.EXPERIMENTS / "v4" / "decision_params.json").read_text())
    a = decide(pl.read_parquet(C.CACHE / "s2" / "test_pred.parquet"), pa["t_any"], pa["t_add"]).join(s1, on="s1")
    b = decide(pl.read_parquet(C.CACHE / "pred" / "test.parquet"), pb["t_any"], pb["t_add"]).join(s1, on="s1")
    pred = pl.concat([a.filter(~pl.col("country").is_in(keep_v5)), b.filter(pl.col("country").is_in(keep_v5))]).select("s1", "m")
    cand = pl.read_parquet(C.CACHE / "cand" / "test.parquet").select("s1", "m")

    def write(pairs, col, path):
        g = pairs.group_by("s1").agg(pl.col("m").unique().sort().str.join(",").alias(col))
        s1.select("s1").join(g, on="s1", how="left").with_columns(pl.col(col).fill_null("")) \
            .rename({"s1": "source1_entity_id"}).write_csv(path, separator="\t", quote_style="never")

    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    write(pred, "matched_entity_ids", C.OUTPUT / "matching_results.tsv")
    write(cand, "candidate_entity_ids", C.OUTPUT / "candidate_pairs.tsv")
    step("hybrid", f"variant B written: {pred.height:,} matches; v5 decisions kept for {keep_v5}")
