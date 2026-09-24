"""LightGBM pair classifier (PLAN.md B4) and decision rule (B5).

Training rows: candidates of training-portion S1s (world20). Early stopping uses a
10% slice of the training-portion S1s, never the holdout. Thresholds are tuned on
holdout half A and reported on half B, and vice versa (V6).
"""
import json

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config as C
from .features import FEATURES
from .metrics import per_s1_f05, summarize
from .progress import step

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=C.N_PROCS, verbose=-1, seed=C.SEED)


def label(feats: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    return feats.join(truth.select("s1", "m").with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "m"], how="left") \
        .with_columns(pl.col("y").fill_null(0))


def train(feats: pl.DataFrame, train_s1: pl.DataFrame, rounds: int = 2000, name: str = "lgb") -> lgb.Booster:
    d = feats.join(train_s1.select(pl.col("entity_id").alias("s1")), on="s1")
    s1s = d.select("s1").unique().sort("s1").sample(fraction=0.1, seed=C.SEED)
    es = d.join(s1s, on="s1")
    tr = d.join(s1s, on="s1", how="anti")
    step(name, f"train rows {tr.height:,} (pos {int(tr['y'].sum()):,}), early-stop rows {es.height:,}")
    dtr = lgb.Dataset(tr.select(FEATURES).to_numpy(), tr["y"].to_numpy(), feature_name=FEATURES, free_raw_data=True)
    des = lgb.Dataset(es.select(FEATURES).to_numpy(), es["y"].to_numpy(), reference=dtr)

    def log_cb(env):
        if env.iteration % 50 == 0:
            res = ", ".join(f"{n} {m}={v:.5f}" for n, m, v, _ in env.evaluation_result_list)
            step(name, f"round {env.iteration}/{rounds} {res}")

    booster = lgb.train(PARAMS | {"metric": "binary_logloss"}, dtr, rounds, valid_sets=[des], valid_names=["es"],
                        callbacks=[lgb.early_stopping(50, verbose=False), log_cb])
    step(name, f"best iteration {booster.best_iteration}")
    return booster


def predict(booster: lgb.Booster, feats: pl.DataFrame) -> np.ndarray:
    return booster.predict(feats.select(FEATURES).to_numpy(), num_iteration=booster.best_iteration)


def decide(scored: pl.DataFrame, t_any: float, t_add: float, resolve: bool = False) -> pl.DataFrame:
    """scored: (s1, m, p). Returns predicted (s1, m).

    t_any: predict the top candidate only if its p >= t_any (else abstain = singleton).
    t_add: further candidates need p >= t_add.
    resolve: each m is kept only for its highest-p S1 among the scored rows (ordering B:
             threshold first, then resolve conflicts).
    """
    d = scored.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("rk"))
    d = d.filter(((pl.col("rk") == 1) & (pl.col("p") >= t_any)) | ((pl.col("rk") > 1) & (pl.col("p") >= t_add)))
    if resolve:
        d = d.filter(pl.col("p") == pl.col("p").max().over("m"))
    return d.select("s1", "m")


def tune(scored: pl.DataFrame, truth: pl.DataFrame, s1: pl.DataFrame, grid_any=None, grid_add=None) -> dict:
    """Grid search (t_any, t_add) on s1 (holdout half); returns best params and score."""
    grid_any = grid_any or [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    grid_add = grid_add or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sc = scored.join(s1.select("s1"), on="s1")
    best = None
    for ta in grid_any:
        for tb in grid_add:
            f = summarize(per_s1_f05(decide(sc, ta, tb), truth, s1))["macro_f05"]
            if best is None or f > best[0]:
                best = (f, ta, tb)
    return {"f05": best[0], "t_any": best[1], "t_add": best[2]}


def two_way(scored: pl.DataFrame, truth: pl.DataFrame, hold: pl.DataFrame) -> dict:
    """V6: tune on A report on B, tune on B report on A; hold has (s1, country, holdout_half)."""
    out = {}
    for tune_h, rep_h in (("A", "B"), ("B", "A")):
        ht = hold.filter(pl.col("holdout_half") == tune_h)
        hr = hold.filter(pl.col("holdout_half") == rep_h)
        best = tune(scored, truth, ht)
        rep = summarize(per_s1_f05(decide(scored.join(hr.select("s1"), on="s1"), best["t_any"], best["t_add"]), truth, hr))
        out[f"tune{tune_h}_report{rep_h}"] = {"params": best, "report": rep}
    return out


def save_json(obj, path):
    path.write_text(json.dumps(obj, indent=2))
