"""Where does holdout macro-F0.5 lose points? (20%-ghost world, chosen thresholds)."""
import json

import polars as pl

from ber import config as C
from ber.io import load_train_pairs
from ber.metrics import per_s1_f05
from ber.model import decide

if __name__ == "__main__":
    prm = json.loads((C.EXPERIMENTS / "decision_params.json").read_text())
    truth = load_train_pairs().select("s1", "m")
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country")
    sc = pl.read_parquet(C.CACHE / "pred" / "train_holdout_w20.parquet")
    pred = decide(sc, prm["t_any"], prm["t_add"])
    per = per_s1_f05(pred, truth, hold)
    cand = pl.read_parquet(C.CACHE / "cand" / "train_w20.parquet", columns=["s1", "m"]).join(hold.select("s1"), on="s1")
    tr = truth.join(hold.select("s1"), on="s1")
    got = tr.join(cand.with_columns(pl.lit(1).alias("c")), on=["s1", "m"], how="left").group_by("s1").agg(pl.col("c").sum().alias("Rb"))
    per = per.join(got, on="s1", how="left").fill_null(0)
    per = per.with_columns((1 - pl.col("f05")).alias("loss"))
    n = per.height
    cats = per.with_columns(
        pl.when(pl.col("T") == 0).then(pl.when(pl.col("P") > 0).then(pl.lit("singleton: predicted a match")).otherwise(pl.lit("singleton: ok")))
        .when(pl.col("P") == 0).then(pl.lit("non-singleton: predicted empty"))
        .when(pl.col("TP") < pl.col("P")).then(pl.lit("has false positive(s)"))
        .when(pl.col("TP") < pl.col("T")).then(pl.lit("only misses (FN)"))
        .otherwise(pl.lit("perfect")).alias("cat"))
    tab = cats.group_by("cat").agg(pl.len().alias("n_s1"), (pl.col("loss").sum() / n).alias("macro_loss")).sort("macro_loss", descending=True)
    print("total macro loss:", round(per["loss"].mean(), 5))
    print(tab)
    # FN decomposition: missed by blocking vs rejected by model
    fn_block = (per["T"] - per["Rb"]).sum()
    fn_model = (per["Rb"] - per["TP"]).sum()
    fp = (per["P"] - per["TP"]).sum()
    print(f"pairs: truth {per['T'].sum():,}; FN missed by blocking {fn_block:,}; FN rejected by model {fn_model:,}; FP {fp:,}; TP {per['TP'].sum():,}")
    print(cats.group_by("country", "cat").len().sort("country", "len", descending=[False, True]))
