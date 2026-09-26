"""Reviewer 01:57 / 09:34 UTC: training-size ablation on EXISTING v6 feature rows, unchanged S1 partitions.
Stage 1: fit on 100k / 200k / 400k training-portion S1s (subsets of the v4 sample present in cache/v6/feats/train_w20)
and report holdout two-way macro-F0.5 (world 20). Stage 2: same on cache/v6/s2/s2feats_w20_e5base with the stage-2
fit sample. Nested subsets (same seed), same PARAMS, early stopping as in production."""
import json

import lightgbm as lgb
import polars as pl

from ber import config as C
from ber import stage2 as S2
from ber.features import FEATURES
from ber.io import load_train_pairs
from ber.model import PARAMS, label, two_way
from ber.progress import step

sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country", "holdout_half")
truth = load_train_pairs()
tp = truth.select("s1", "m")


def fit_eval(d, feats, train_s1, extra):
    x = d.join(train_s1, on="s1")
    es_s1 = x.select("s1").unique().sort("s1").sample(fraction=0.1, seed=C.SEED)
    es, tr = x.join(es_s1, on="s1"), x.join(es_s1, on="s1", how="anti")
    b = lgb.train(PARAMS | {"metric": "binary_logloss"} | extra, lgb.Dataset(tr.select(feats).to_numpy(), tr["y"].to_numpy(), feature_name=feats),
                  3000, valid_sets=[lgb.Dataset(es.select(feats).to_numpy(), es["y"].to_numpy())], callbacks=[lgb.early_stopping(100, verbose=False)])
    h = d.join(hold.select("s1"), on="s1")
    s = h.select("s1", "m").with_columns(pl.Series("p", b.predict(h.select(feats).to_numpy(), num_iteration=b.best_iteration)))
    r = two_way(s, tp, hold)
    return {k: v["report"]["macro_f05"] for k, v in r.items()} | {"best_iter": b.best_iteration, "train_rows": tr.height}


rep = {}
d1 = label(pl.read_parquet(C.WORK / "feats" / "train_w20.parquet"), truth)
tr1 = d1.select("s1").unique().join(sp.filter(pl.col("is_train")).select(pl.col("entity_id").alias("s1")), on="s1").sort("s1")
for n in (100_000, 200_000, tr1.height):
    sub = tr1.sample(min(n, tr1.height), seed=C.SEED + 1)
    rep[f"stage1 {sub.height}"] = fit_eval(d1, FEATURES, sub, {})
    step("lc", f"stage1 {sub.height}: {rep[f'stage1 {sub.height}']}")
del d1
d2 = pl.read_parquet(S2.S2 / f"s2feats_w20{S2.SFX}.parquet").join(tp.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "m"], how="left") \
    .with_columns(pl.col("y").fill_null(0))
F2 = S2._feat_list(d2)
tr2 = sp.filter(pl.col("is_train")).select(pl.col("entity_id").alias("s1")).sample(S2.S2_TRAIN_S1, seed=C.SEED + 99).sort("s1")
for n in (100_000, 200_000, tr2.height):
    sub = tr2.sample(min(n, tr2.height), seed=C.SEED + 2)
    rep[f"stage2 {sub.height}"] = fit_eval(d2, F2, sub, {"num_leaves": 63})
    step("lc", f"stage2 {sub.height}: {rep[f'stage2 {sub.height}']}")
(C.EXP_RUN / "learning_curve_w20.json").write_text(json.dumps(rep, indent=1))
