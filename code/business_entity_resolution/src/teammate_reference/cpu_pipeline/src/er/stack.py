"""Stacker over the team's e5base2 scores and ours (US/India): LightGBM on both logits + competition features of both
(per S1 and per record), trained on the labelled holdout. Two-way check: fit on half A -> score B and vice versa,
thresholds tuned on the fitted half's out-of-fold scores.

  python -m er.stack eval
  python -m er.stack test      # fit on the full holdout -> cache/stack_test.parquet (US/India team universe)
"""
import argparse
import itertools
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.metrics import per_s1_f05, summarize
from er.common import CACHE, hold_s1, log, truth_pairs
from er.final import blend as our_blend
from er.model import decide
from er.teamblend import TEAM, combine

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.9,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=8, seed=7)
ROUNDS = 400


def _lg(c):
    x = pl.col(c).clip(1e-6, 1 - 1e-6)
    return (x / (1 - x)).log()


def features(team: pl.DataFrame, ours: pl.DataFrame) -> pl.DataFrame:
    d = team.rename({"p": "pt"}).join(ours.rename({"p": "po"}), on=["s1", "m"], how="left")
    d = d.with_columns(_lg("pt").alias("lt"), pl.col("po").is_not_null().cast(pl.Int8).alias("has_o"),
                       pl.coalesce("po", "pt").alias("po_f"))
    d = d.with_columns(_lg("po_f").alias("lo"), (0.7 * _lg("pt") + 0.3 * _lg("po_f")).alias("lb"))
    out = [pl.col(c) for c in ("s1", "m", "lt", "lo", "lb", "has_o")]
    for c in ("lt", "lb"):
        out += [
            pl.col(c).rank("ordinal", descending=True).over("s1").cast(pl.Int16).alias(f"{c}_rk_s1"),
            (pl.col(c) - pl.col(c).max().over("s1")).alias(f"{c}_gap_s1top"),
            pl.col(c).rank("ordinal", descending=True).over("m").cast(pl.Int16).alias(f"{c}_rk_m"),
            (pl.col(c) - pl.col(c).max().over("m")).alias(f"{c}_gap_mtop"),
            (pl.col(c) > 0).sum().over("s1").cast(pl.Int16).alias(f"{c}_n_pos_s1"),
            (pl.col(c) > 0).sum().over("m").cast(pl.Int16).alias(f"{c}_n_pos_m"),
        ]
    # second-best competitor for the record (excluding this pair)
    d = d.select(out)
    return d.with_columns(pl.col(pl.Float64).cast(pl.Float32))


FEATS = None


def _fit(d: pl.DataFrame):
    feats = [c for c in d.columns if c not in ("s1", "m", "y", "half")]
    ds = lgb.Dataset(d.select(feats).to_numpy(), d["y"].to_numpy(), feature_name=feats)
    return lgb.train(PARAMS, ds, ROUNDS), feats


def _pred(b, feats, d):
    return d.select("s1", "m").with_columns(pl.Series("p", b.predict(d.select(feats).to_numpy()).astype(np.float32)))


GRID = list(itertools.product([0.4, 0.5, 0.6, 0.7], [0.65, 0.7, 0.75, 0.8]))


def evaluate():
    hold = hold_s1()
    truth = truth_pairs().join(hold.select("s1"), on="s1").with_columns(pl.lit(1, pl.Int8).alias("y"))
    team = pl.read_parquet(TEAM / "holdout_pred_w20_e5base2.parquet").select("s1", "m", "p").join(hold.select("s1"), on="s1")
    ours = our_blend(["p22_train_holdout.parquet", "p23b_train_holdout.parquet"], hold.select("s1").lazy())
    d = features(team, ours).join(truth.select("s1", "m", "y"), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0)) \
        .join(hold.select("s1", pl.col("holdout_half").alias("half")), on="s1")
    log(f"stack eval: {d.height:,} pairs, {d.width} cols")
    res = {}
    oof = {}
    for tr, te in (("A", "B"), ("B", "A")):
        b, feats = _fit(d.filter(pl.col("half") == tr).drop("half"))
        oof[te] = _pred(b, feats, d.filter(pl.col("half") == te))
        log(f"stack: fitted on {tr}, scored {te}")
    for th, rh in (("A", "B"), ("B", "A")):   # thresholds tuned on one half's out-of-fold scores, reported on the other
        ht, hr = hold.filter(pl.col("holdout_half") == th), hold.filter(pl.col("holdout_half") == rh)
        f, prm = max((summarize(per_s1_f05(decide(oof[th], *g, "p"), truth, ht))["macro_f05"], g) for g in GRID)
        rep = summarize(per_s1_f05(decide(oof[rh], *prm, "p"), truth, hr))
        res[f"tune{th}_rep{rh}"] = (rep["macro_f05"], rep["by_country"], prm)
        log(f"stack tune {th} -> {rh}: {rep['macro_f05']} {rep['by_country']} params {prm}")
    oof = list(oof.values())
    oofd = pl.concat(oof)
    full = max((summarize(per_s1_f05(decide(oofd, *g, "p"), truth, hold))["macro_f05"], g) for g in GRID)
    res["oof_full_best"] = full
    blend_ref = summarize(per_s1_f05(decide(combine(team, ours, 0.7), 0.6, 0.75, "p"), truth, hold))["macro_f05"]
    res["blend_ref_full"] = blend_ref
    log("stack eval: " + json.dumps(res))
    print(json.dumps(res, indent=1))


def test():
    hold = hold_s1()
    truth = truth_pairs().join(hold.select("s1"), on="s1").with_columns(pl.lit(1, pl.Int8).alias("y"))
    team = pl.read_parquet(TEAM / "holdout_pred_w20_e5base2.parquet").select("s1", "m", "p").join(hold.select("s1"), on="s1")
    ours = our_blend(["p22_train_holdout.parquet", "p23b_train_holdout.parquet"], hold.select("s1").lazy())
    d = features(team, ours).join(truth.select("s1", "m", "y"), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    b, feats = _fit(d)
    del d, team, ours
    cty = pl.read_parquet(CACHE / "test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": "s1"})
    outs = []
    for c in ("US", "India"):
        ids = cty.filter(pl.col("country") == c).select("s1")
        t = pl.scan_parquet(TEAM / "test_pred_e5base2.parquet").select("s1", "m", "p").join(ids.lazy(), on="s1", how="semi").collect()
        o = our_blend(["p22_test.parquet", "p23b_test.parquet"], ids.lazy())
        x = features(t, o)
        del t, o
        outs.append(_pred(b, feats, x))
        log(f"stack test {c}: {x.height:,} pairs scored")
        del x
    pl.concat(outs).write_parquet(CACHE / "stack_test.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["eval", "test"])
    a = ap.parse_args()
    evaluate() if a.step == "eval" else test()
