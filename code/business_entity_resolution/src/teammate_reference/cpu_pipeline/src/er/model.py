"""Stage-1 / stage-2 LightGBM matchers, competition features and the decision rule."""
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from ber.metrics import per_s1_f05, summarize

SEED = 20260927
PARAMS = dict(objective="binary", learning_rate=float(os.environ.get("ER_LR", 0.08)), num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=int(os.environ.get("ER_THREADS", 8)),
              verbose=-1, seed=SEED, max_bin=127)
NON_FEAT = {"s1", "m", "y", "country", "fold", "is_hold", "half"}


def feat_cols(df: pl.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEAT and not c.startswith("_")]


def context(c: pl.DataFrame) -> pl.DataFrame:
    """Blocking-context features over the candidate universe (label-free)."""
    return c.with_columns(
        pl.len().over("s1").alias("s_ncand"),
        pl.len().over("m").alias("m_ncand"),
        (pl.col("bscore") / pl.col("bscore").max().over("s1")).alias("b_rel_s"),
        (pl.col("bscore") / pl.col("bscore").max().over("m")).alias("b_rel_m"),
    )


def competition(d: pl.DataFrame, p: str = "p1") -> pl.DataFrame:
    """Competition features from stage-1 scores over the whole universe (every rival S1 present)."""
    d = d.with_columns(
        pl.col(p).rank("ordinal", descending=True).over("m").alias("m_rank"),
        pl.col(p).rank("ordinal", descending=True).over("s1").alias("s_rank_p"),
        pl.col(p).max().over("s1").alias("s_top"),
        pl.col(p).sum().over("s1").alias("s_sum"),
        (pl.col(p) >= 0.5).sum().over("s1").alias("s_n_hi"),
        (pl.col(p) >= 0.5).sum().over("m").alias("m_n_hi"),
        pl.col(p).sum().over("m").alias("m_sum"),
    )
    # best score of this record at any other S1 / this S1's second best (via top-2 per group)
    top2m = d.group_by("m").agg(pl.col(p).top_k(2).alias("_t"))
    top2m = top2m.with_columns(pl.col("_t").list.get(0).alias("_m1"), pl.col("_t").list.get(1, null_on_oob=True).fill_null(0.0).alias("_m2")).drop("_t")
    top2s = d.group_by("s1").agg(pl.col(p).top_k(2).alias("_t"))
    top2s = top2s.with_columns(pl.col("_t").list.get(0).alias("_s1"), pl.col("_t").list.get(1, null_on_oob=True).fill_null(0.0).alias("_s2")).drop("_t")
    d = d.join(top2m, on="m", how="left").join(top2s, on="s1", how="left")
    d = d.with_columns(
        pl.when(pl.col("m_rank") == 1).then(pl.col("_m2")).otherwise(pl.col("_m1")).alias("m_best_other"),
        pl.when(pl.col("s_rank_p") == 1).then(pl.col("_s2")).otherwise(pl.col("_s1")).alias("s_best_other"),
    ).with_columns(
        (pl.col(p) - pl.col("m_best_other")).alias("m_gap"),
        (pl.col(p) - pl.col("s_best_other")).alias("s_gap"),
        (pl.col("s_top") - pl.col(p)).alias("s_gap_top"),
        (pl.col("m_sum") - pl.col(p)).alias("m_sum_other"),
    ).drop("_m1", "_m2", "_s1", "_s2")
    return d


def train(df: pl.DataFrame, feats: list, rounds: int = 2000, valid_frac: float = 0.1, log=print) -> lgb.Booster:
    s1s = df.select("s1").unique().sort("s1").sample(fraction=valid_frac, seed=SEED)
    es = df.join(s1s, on="s1")
    tr = df.join(s1s, on="s1", how="anti")
    log(f"  lgb train rows {tr.height:,} (pos {int(tr['y'].sum()):,}), es rows {es.height:,}, {len(feats)} feats")
    dtr = lgb.Dataset(tr.select(feats).to_numpy().astype(np.float32), tr["y"].to_numpy(), feature_name=feats, free_raw_data=True)
    des = lgb.Dataset(es.select(feats).to_numpy().astype(np.float32), es["y"].to_numpy(), reference=dtr)

    def cb(env):
        if env.iteration % 100 == 0:
            log(f"  lgb round {env.iteration} es_logloss={env.evaluation_result_list[0][2]:.5f}")

    b = lgb.train(PARAMS | {"metric": "binary_logloss"}, dtr, rounds, valid_sets=[des], valid_names=["es"],
                  callbacks=[lgb.early_stopping(50, verbose=False), cb])
    log(f"  lgb best iteration {b.best_iteration}")
    return b


def predict(b: lgb.Booster, df: pl.DataFrame, feats: list) -> np.ndarray:
    out = np.empty(df.height, dtype=np.float32)
    step = 2_000_000
    for lo in range(0, df.height, step):
        out[lo:lo + step] = b.predict(df.slice(lo, step).select(feats).to_numpy().astype(np.float32), num_iteration=b.best_iteration)
    return out


def decide(scored: pl.DataFrame, t_any: float, t_add: float, p: str = "p", unique: bool = True) -> pl.DataFrame:
    """Accept an S1's top candidate if p >= t_any, further ones if p >= t_add; a record goes only to its best S1."""
    d = scored.select("s1", "m", p)
    if unique:
        d = d.filter(pl.col(p) == pl.col(p).max().over("m"))
    d = d.with_columns(pl.col(p).rank("ordinal", descending=True).over("s1").alias("_rk"), pl.col(p).max().over("s1").alias("_top"))
    return d.filter((pl.col("_top") >= t_any) & ((pl.col("_rk") == 1) | (pl.col(p) >= t_add))).select("s1", "m")


GRID_ANY = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
GRID_ADD = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


def tune(scored, truth, hold, p="p"):
    best = None
    for ta in GRID_ANY:
        for tb in GRID_ADD:
            f = summarize(per_s1_f05(decide(scored, ta, tb, p), truth, hold))["macro_f05"]
            if best is None or f > best[0]:
                best = (f, ta, tb)
    return best


def two_way(scored, truth, hold, p="p"):
    """Tune on half A, report on B and vice versa; plus full-holdout best."""
    out = {}
    for th, rh in (("A", "B"), ("B", "A")):
        ht, hr = hold.filter(pl.col("holdout_half") == th), hold.filter(pl.col("holdout_half") == rh)
        f, ta, tb = tune(scored.join(ht.select("s1"), on="s1"), truth, ht, p)
        rep = summarize(per_s1_f05(decide(scored.join(hr.select("s1"), on="s1"), ta, tb, p), truth, hr))
        out[f"tune{th}_rep{rh}"] = {"t_any": ta, "t_add": tb, **rep}
    f, ta, tb = tune(scored.join(hold.select("s1"), on="s1"), truth, hold, p)
    out["full"] = {"t_any": ta, "t_add": tb, "f05": f}
    return out
