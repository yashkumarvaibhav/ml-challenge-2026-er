"""France v5 test run, France only: re-block + re-score France with the unchanged v4 model and
reuse v4's US/India candidates and scores. That is equivalent to a full run, because blocking and
features are computed per country and the France maps change only France records.

  python code/business_entity_resolution/src/ber/experiments/v5_france_only.py
"""
import lightgbm as lgb
import polars as pl

from ber import config as C
from ber.blocking import block
from ber.features import build as build_features
from ber.io import load_source
from ber.model import predict
from ber.progress import step

if __name__ == "__main__":
    s1 = load_source("test", 1).select("entity_id", "country")
    fr = s1.filter(pl.col("country") == "France").select("entity_id")
    cand_fr = block("test", fr, s1.select("entity_id"))
    v4c = pl.read_parquet(C.CACHE / "runs" / "v4" / "test_cand.parquet")
    other = v4c.join(fr.rename({"entity_id": "s1"}), on="s1", how="anti")
    cand = pl.concat([other, cand_fr.select(other.columns)])
    cand.write_parquet(C.CACHE / "cand" / "test.parquet")
    step("v5", f"candidates: France {cand_fr.height:,} (re-blocked) + US/India {other.height:,} (v4) = {cand.height:,}")
    booster = lgb.Booster(model_file=str(C.ROOT / "models" / "lgb_v4.txt"))
    f = build_features("test", cand_fr, s1.select("entity_id"), name="features+score test France (v5)")
    p_fr = f.select("s1", "m").with_columns(pl.Series("p", predict(booster, f)))
    v4p = pl.read_parquet(C.CACHE / "runs" / "v4" / "test_pred.parquet").join(fr.rename({"entity_id": "s1"}), on="s1", how="anti")
    pred = pl.concat([v4p, p_fr])
    pred.write_parquet(C.CACHE / "pred" / "test.parquet")
    step("v5", f"scores: France {p_fr.height:,} (v5) + US/India {v4p.height:,} (v4) = {pred.height:,}")
