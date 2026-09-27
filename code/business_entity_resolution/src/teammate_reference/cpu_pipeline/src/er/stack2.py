"""Stacker v2 (US/India): LightGBM on team / ours / Qwen-7B / Qwen-1.5B logits + per-S1 and per-record competition
features, 4-fold cross-fitted over holdout S1s (every holdout pair scored by a model that never saw its S1), then
two-way threshold tuning on the out-of-fold scores (tune half A -> report half B and vice versa).

  python -m er.stack2 eval
  python -m er.stack2 test     # fit on the whole holdout -> cache/stack2_test.parquet (US/India)
"""
import argparse
import itertools
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.metrics import per_s1_f05, summarize
from er.blend4 import sources
from er.common import CACHE, hold_s1, log, truth_pairs
from er.model import decide

import os
V3 = os.environ.get("ER_STACK_V3") == "1"
PARAMS = dict(objective="binary", learning_rate=0.04, num_leaves=63 if V3 else 31, min_data_in_leaf=300, feature_fraction=0.85,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=7)
ROUNDS = 500 if V3 else 350
PAIR_COLS = ["n_cov1", "n_cov2", "a_cov1", "a_cov2", "num_shared", "num_miss1", "num_miss2", "a_null2", "n_tset", "a_tset",
             "cmp_ratio", "core_cnt1", "rec_core_cnt2", "n_soft1", "src3"]
ADDON_COLS = ["tf_name", "tf_core", "tf_addr"]


def pair_feats(split, ids):
    """Our pair features for the pairs in our candidate universe (feats2 + TF-IDF add-on), US/India S1s in ids."""
    sp = "train" if split == "hold" else "test"
    out = []
    for k in range(16):
        f = pl.scan_parquet(CACHE / "feats2" / sp / f"part{k}.parquet").select(["s1", "m"] + PAIR_COLS).join(ids.lazy(), on="s1", how="semi")
        a = pl.scan_parquet(CACHE / "addon" / sp / f"part{k}.parquet").select(["s1", "m"] + ADDON_COLS)
        out.append(f.join(a, on=["s1", "m"], how="left").collect())
    return pl.concat(out).with_columns(pl.col(pl.Float64).cast(pl.Float32))


def _lg(c):
    x = pl.col(c).clip(1e-6, 1 - 1e-6)
    return (x / (1 - x)).log()


def feats(d: pl.DataFrame, raw: pl.DataFrame) -> pl.DataFrame:
    """d: sources() output (missing sources already coalesced to pt); raw: presence flags."""
    x = d.join(raw, on=["s1", "m"], how="left").with_columns(
        _lg("pt").alias("lt"), _lg("po").alias("lo"), _lg("q7").alias("l7"), _lg("q15").alias("l15"))
    x = x.with_columns((0.7 * pl.col("lt") + 0.3 * pl.col("lo")).alias("lb"))
    cols = [pl.col(c) for c in ("s1", "m", "lt", "lo", "l7", "l15", "lb", "has_o", "has7", "has15")]
    for c in ("lt", "lb"):
        cols += [pl.col(c).rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias(f"{c}_rk_s1"),
                 (pl.col(c) - pl.col(c).max().over("s1")).alias(f"{c}_gap_s1"),
                 pl.col(c).rank("ordinal", descending=True).over("m").cast(pl.Int16).alias(f"{c}_rk_m"),
                 (pl.col(c) - pl.col(c).max().over("m")).alias(f"{c}_gap_m"),
                 (pl.col(c) > 0).sum().over("s1").cast(pl.Int16).alias(f"{c}_npos_s1"),
                 (pl.col(c) > 0).sum().over("m").cast(pl.Int16).alias(f"{c}_npos_m")]
    return x.select(cols).with_columns(pl.col(pl.Float64).cast(pl.Float32))


def raw_flags(split, ids):
    """Which pairs each non-team source actually scored (before coalescing)."""
    from er.final import blend as our_blend
    from er.blend4 import L
    if split == "hold":
        o = our_blend(["p22_train_holdout.parquet", "p23b_train_holdout.parquet"], ids.lazy())
        q7 = [L / "llm_s3band_w20_qwen7b_r3.parquet", L / "llm_hiband_holdout_qwen7b_r3.parquet"]
        q15 = [L / f for f in ("llm_s3band_w20_qwen15_r3.parquet", "llm_hiband_holdout_qwen15_r3.parquet", "llm_loband_holdout_qwen15_r3.parquet")]
    else:
        o = our_blend(["p22_test.parquet", "p23b_test.parquet"], ids.lazy())
        q7 = [L / "llm_s3band_test_qwen7b_r3.parquet", L / "llm_hiband_test_qwen7b_r3.parquet"]
        q15 = [L / f for f in ("llm_s3band_test_qwen15_r3.parquet", "llm_hiband_test_qwen15_r3.parquet", "llm_loband_test_qwen15_r3.parquet")]
    f = o.select("s1", "m", pl.lit(1, pl.Int8).alias("has_o"))
    f7 = pl.concat([pl.read_parquet(p).select("s1", "m") for p in q7]).unique().with_columns(pl.lit(1, pl.Int8).alias("has7"))
    f15 = pl.concat([pl.read_parquet(p).select("s1", "m") for p in q15]).unique().with_columns(pl.lit(1, pl.Int8).alias("has15"))
    return f.join(f7, on=["s1", "m"], how="full", coalesce=True).join(f15, on=["s1", "m"], how="full", coalesce=True) \
        .with_columns([pl.col(c).fill_null(0) for c in ("has_o", "has7", "has15")])


FEAT_EXCLUDE = ("s1", "m", "y", "fold", "half")
GRID_T = list(itertools.product([0.4, 0.5, 0.6, 0.7], [0.65, 0.7, 0.75, 0.8]))


def _data_hold():
    hold = hold_s1()
    ids = hold.select("s1")
    d = feats(sources("hold", ids), raw_flags("hold", ids))
    if V3:
        d = d.join(pair_feats("hold", ids), on=["s1", "m"], how="left")
    truth = truth_pairs().join(ids, on="s1").with_columns(pl.lit(1, pl.Int8).alias("y"))
    d = d.join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    fold = hold.select("s1", pl.col("holdout_half").alias("half"), (pl.col("s1").hash(seed=17) % 2).alias("sub"))
    d = d.join(fold, on="s1").with_columns((pl.when(pl.col("half") == "A").then(0).otherwise(2) + pl.col("sub")).alias("fold")).drop("sub")
    return hold, truth, d


def _fit(d):
    fs = [c for c in d.columns if c not in FEAT_EXCLUDE]
    b = lgb.train(PARAMS, lgb.Dataset(d.select(fs).to_numpy(), d["y"].to_numpy(), feature_name=fs), ROUNDS)
    return b, fs


def evaluate():
    hold, truth, d = _data_hold()
    log(f"stack2 eval (v3={V3}): {d.height:,} pairs, {d.width} cols")
    oof = []
    for k in range(4):
        b, fs = _fit(d.filter(pl.col("fold") != k))
        te = d.filter(pl.col("fold") == k)
        oof.append(te.select("s1", "m").with_columns(pl.Series("p", b.predict(te.select(fs).to_numpy()).astype(np.float32))))
        log(f"stack2 fold {k} done")
    sc = pl.concat(oof)
    res = {}
    for th, rh in (("A", "B"), ("B", "A")):
        ht, hr = hold.filter(pl.col("holdout_half") == th), hold.filter(pl.col("holdout_half") == rh)
        f, g = max((summarize(per_s1_f05(decide(sc.join(ht.select("s1"), on="s1"), *g, "p"), truth, ht))["macro_f05"], g) for g in GRID_T)
        rep = summarize(per_s1_f05(decide(sc.join(hr.select("s1"), on="s1"), *g, "p"), truth, hr))
        res[f"tune{th}_rep{rh}"] = (rep["macro_f05"], rep["by_country"], g)
    res["mean"] = round((res["tuneA_repB"][0] + res["tuneB_repA"][0]) / 2, 5)
    res["full"] = max((summarize(per_s1_f05(decide(sc, *g, "p"), truth, hold))["macro_f05"], g) for g in GRID_T)
    log("stack2 eval " + json.dumps(res))
    print(json.dumps(res, indent=1))


def test():
    hold, truth, d = _data_hold()
    b, fs = _fit(d)
    del d
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    outs = []
    for c in ("US", "India"):
        ids = cty.filter(pl.col("country") == c).select("s1")
        x = feats(sources("test", ids), raw_flags("test", ids))
        if V3:
            x = x.join(pair_feats("test", ids), on=["s1", "m"], how="left")
        outs.append(x.select("s1", "m").with_columns(pl.Series("p", b.predict(x.select(fs).to_numpy()).astype(np.float32))))
        log(f"stack2 test {c}: {x.height:,} pairs")
        del x
    pl.concat(outs).write_parquet(CACHE / ("stack3_test.parquet" if V3 else "stack2_test.parquet"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["eval", "test"])
    a = ap.parse_args()
    evaluate() if a.step == "eval" else test()
