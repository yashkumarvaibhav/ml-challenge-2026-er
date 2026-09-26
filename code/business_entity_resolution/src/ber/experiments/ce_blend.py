"""More CE weight in the uncertain band: p' = (1-w)*p_stage2 + w*ce for pairs the CE scored (stage-1 band).

Holdout (US/India, w20 and w0): macro F0.5 at the fixed decision thresholds, per half, to size the known-country
cost. Test France: how many France decisions each w changes versus the e5base2 package (US/India untouched).
Motivation: the e5base2 public probe (0.984185 vs v6 0.983526) implies the CE's France band changes were net
positive (~+0.0027 France F0.5 under the 85/15 scenario).
"""
import json

import polars as pl

from .. import config as C
from ..metrics import per_s1_f05, summarize
from ..model import decide
from ..io import load_train_pairs
from ..stage2 import _split

S2 = C.WORK / "s2"
WS = (0.0, 0.2, 0.35, 0.5, 0.7, 1.0)
TA, TD = 0.4, 0.75


def blend(pred: pl.DataFrame, ce: pl.DataFrame, w: float) -> pl.DataFrame:
    return (pred.join(ce, on=["s1", "m"], how="left")
            .with_columns(pl.when(pl.col("ce").is_not_null()).then((1 - w) * pl.col("p") + w * pl.col("ce")).otherwise(pl.col("p")).alias("p"))
            .select("s1", "m", "p"))


def main():
    rep = {}
    sp = _split()
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country", "holdout_half")
    truth = load_train_pairs().select("s1", "m")
    for wd in (20, 0):
        pred = pl.read_parquet(S2 / f"holdout_pred_w{wd}_e5base2.parquet")
        ce = pl.read_parquet(S2 / f"ce_w{wd}_e5base2.parquet").unique(["s1", "m"])
        for w in WS:
            d = decide(blend(pred, ce, w), TA, TD)
            r = {}
            for h in ("A", "B"):
                hs = hold.filter(pl.col("holdout_half") == h)
                r[h] = summarize(per_s1_f05(d, truth, hs))["macro_f05"]
            r["by_country"] = summarize(per_s1_f05(d, truth, hold))["by_country"]
            rep[f"holdout w{wd} w={w}"] = r
            print(f"w{wd} w={w}", r, flush=True)
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").select(pl.col("entity_id").alias("s1"), "country")
    fr = t1.filter(pl.col("country") == "France").select("s1")
    pred = pl.read_parquet(S2 / "test_pred_e5base2.parquet").join(fr, on="s1")
    ce = pl.read_parquet(S2 / "ce_test_e5base2.parquet").unique(["s1", "m"]).join(fr, on="s1")
    base = decide(pred, TA, TD)
    for w in WS[1:]:
        d = decide(blend(pred, ce, w), TA, TD)
        add, rem = d.join(base, on=["s1", "m"], how="anti"), base.join(d, on=["s1", "m"], how="anti")
        ch = pl.concat([add.select("s1"), rem.select("s1")])["s1"].n_unique()
        rep[f"test France w={w}"] = {"added": add.height, "removed": rem.height, "S1_changed": ch, "share": round(ch / fr.height, 4)}
        print(f"France w={w}", rep[f"test France w={w}"], flush=True)
    (C.EXP_RUN / "ce_blend.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
