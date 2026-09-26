"""Stage-2 competition re-ranker (PLAN.md §6.3; plan-discussion 2026-09-25 05:28 UTC).

The stage-1 LightGBM scores each (S1, record) pair in isolation. Each S2/S3 record belongs to at
most one S1, and address-less / shared-name records are exactly where stage 1 is unsure (v4 error
analysis). Stage 2 re-scores every candidate with label-free *competition* features computed from
stage-1 scores over the whole candidate universe (every rival S1 present, as on test):

* record side: best score of this record at any OTHER S1, gap to it, this S1's rank for the record,
  number of S1s scoring the record >= 0.5
* S1 side: rank of the pair within the S1, the S1's top score, gap to the top, number of candidates
  >= 0.5, sum of scores
* context: address-null flag of the record, candidate count, blocking channel ranks

Cross-fitting (no leakage):
* training-portion S1s get OUT-OF-FOLD stage-1 scores (3 folds; each fold model trained on <= 400k
  training-portion S1s from the other folds, v4 features/params);
* every other world S1 (holdout, and in world 0 also the non-holdout ghost S1s) is scored by the v4
  model (trained on training-portion S1s only), exactly like test; coverage of the whole candidate
  universe is asserted before competition features are built.

Steps (world20 = ghosts removed; resumable):
  python -m ber.stage2 --steps feats,oof,s2feats,fit,eval --world 20
  python -m ber.stage2 --steps test          (uses cache/pred/test.parquet = v4 stage-1 test scores)
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config as C
from .features import FEATURES
from .features import build as build_features
from .io import load_train_pairs
from .metrics import per_s1_f05, summarize
from .model import GRID_ADD, GRID_ANY, PARAMS, decide, two_way
from .progress import Progress, step

S2 = C.WORK / "s2"
PARTS = 6
N_FOLDS = 3
FOLD_TRAIN_S1 = 400_000
S2_TRAIN_S1 = 400_000
BLK = (["d_cos", "r_df", "r_dr", "r_h", "r_j", "r_cg", "pri"] if C.V6 else
       ["s_a", "s_b", "s_c", "r_a", "r_b", "r_c", "r_e", "r_f", "r_g", "r_h", "r_j", "prio"])
S2_FEATURES = ["p1", "lp1", "m_best_other", "m_gap", "m_rank", "m_n_hi", "m_n_cand",
               "s_rank", "s_top", "s_gap_top", "s_n_hi", "s_sum", "s_n_cand", "m_a_null"] + BLK
# multilingual cross-encoder score, only for the stage-1 uncertain band (missing elsewhere, as on test)
CE_BAND = (0.01, 0.995)
CE_FEATURES = ["ce", "ce_minus_p1"]
MODEL_V4 = C.ROOT / "models" / ("lgb_v6.txt" if C.V6 else "lgb_v4.txt")  # the stage-1 model of this run
# CE variant (BER_CE, see ber.cross_encoder): non-default CEs get their own score files, stage-2 model and reports
SFX = "" if os.environ.get("BER_CE", "e5small") == "e5small" else "_" + os.environ["BER_CE"]
# stage-2 challenger (reviewer 2026-09-26 12:59 UTC; BER_S2X=1): also feed the direct stage-1 pair features, so
# stage 2 sees the name/address evidence behind p1 (e.g. address-null noisy-name extras). Competition features, CE
# scores and S1 partitions are unchanged; its model, reports and predictions get the extra suffix "_x".
XF = os.environ.get("BER_S2X") == "1"
# per-source count features (BER_S2SRC=1; 26 Sep): the generator gives almost every entity records in BOTH sources
# (train: n2 = 0 for 13%, n3 = 0 for 12% of entities), so an S1 that already holds strong records of one source but
# none of the other is the likelier owner of an ambiguous record from the missing source (address-null namesakes).
SRCF = os.environ.get("BER_S2SRC") == "1"
FSFX = "_src" if SRCF else ""
SRC_FEATURES = ["s_n_hi_src", "s_n_hi_oth", "s_sum_src", "s_top_src", "m_rank_src_rival"]
MSFX = SFX + ("_x" if XF else "") + FSFX
MODEL_S2 = C.ROOT / "models" / (("lgb_s2_v6" if C.V6 else "lgb_s2") + MSFX + ".txt")


PAIR_X = [f for f in FEATURES if f not in S2_FEATURES and f not in BLK]


def _with_pair(d: pl.DataFrame, feat_dir) -> pl.DataFrame:
    """Join the direct pair features (PAIR_X) for the S1s in d, part by part (bounded memory)."""
    s1 = d.select("s1").unique()
    x = pl.concat([pl.read_parquet(p, columns=["s1", "m"] + PAIR_X).join(s1, on="s1") for p in sorted(feat_dir.glob("part*.parquet"))])
    out = d.join(x, on=["s1", "m"], how="left")
    assert out.height == d.height and out[PAIR_X[0]].null_count() == 0, "pair features missing for some stage-2 rows"
    return out


def _split():
    return pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")


def _part_expr(k):
    return (pl.col("s1").hash(seed=7) % PARTS) == k


def feats(w: int):
    """Stage-1 features for the full universe, written in S1-partitioned parquet parts."""
    cand = pl.read_parquet(C.WORK / "cand" / f"train_full_w{w}.parquet")
    sp = _split()
    world = (sp.filter(~pl.col("is_ghost")) if w == 20 else sp).select("entity_id")
    out = S2 / f"feats_w{w}"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(PARTS):
        path = out / f"part{k}.parquet"
        if path.exists():
            continue
        ck = cand.filter(_part_expr(k))
        step("stage2", f"features part {k + 1}/{PARTS}: {ck.height:,} pairs ...")
        build_features("train", ck, world, name=f"stage2 features w{w} part {k + 1}/{PARTS}").write_parquet(path)


def _read_parts(w, s1_filter: pl.DataFrame = None, cols=None):
    parts = []
    for k in range(PARTS):
        d = pl.read_parquet(S2 / f"feats_w{w}" / f"part{k}.parquet", columns=cols)
        if s1_filter is not None:
            d = d.join(s1_filter, on="s1")
        parts.append(d)
    return pl.concat(parts)


def _not_bienc() -> pl.DataFrame:
    """Training-portion S1s outside the bi-encoder training set = v4 (stage-1) sample + stage-2 fit sample."""
    tr = _split().filter(pl.col("is_train"))
    return pl.concat([tr.sample(400_000, seed=C.SEED).select(pl.col("entity_id").alias("s1")),
                      tr.select(pl.col("entity_id").alias("s1")).sample(S2_TRAIN_S1, seed=C.SEED + 99)]).unique()


def _bienc() -> pl.DataFrame:
    """The bi-encoder's training S1s (as in ber.dense.data): is_train minus the v4 and stage-2 samples."""
    tr = _split().filter(pl.col("is_train")).select(pl.col("entity_id").alias("s1"))
    return tr.join(_not_bienc(), on="s1", how="anti")


def oof(w: int):
    """Stage-1 scores: out-of-fold for training-portion S1s, v4 model for every other world S1."""
    sp = _split().filter(~pl.col("is_ghost")) if w == 20 else _split()
    truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1, pl.Int8).alias("y"))
    tr = sp.filter(pl.col("is_train")).select(pl.col("entity_id").alias("s1")) \
        .with_columns((pl.col("s1").hash(seed=11) % N_FOLDS).alias("fold"))
    scored = []
    for f in range(N_FOLDS):
        step("stage2", f"OOF fold {f + 1}/{N_FOLDS}: training ...")
        pool_f = tr.filter(pl.col("fold") != f).select("s1")
        if C.V6:  # d_cos is in-sample for bi-encoder training S1s: fold models train only on S1s it never saw
            pool_f = pool_f.join(_not_bienc(), on="s1")
        others = pool_f.sample(min(FOLD_TRAIN_S1, pool_f.height), seed=C.SEED + f)
        if C.V6:  # reviewer 17:13 UTC: fold-training S1s never trained the bi-encoder
            assert others.join(_bienc(), on="s1").height == 0, "OOF fold training overlaps bi-encoder S1s"
        d = _read_parts(w, others, ["s1", "m"] + FEATURES).join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
        es_s1 = d.select("s1").unique().sort("s1").sample(fraction=0.1, seed=C.SEED)
        es, trn = d.join(es_s1, on="s1"), d.join(es_s1, on="s1", how="anti")
        dtr = lgb.Dataset(trn.select(FEATURES).to_numpy(), trn["y"].to_numpy(), feature_name=FEATURES, free_raw_data=True)
        des = lgb.Dataset(es.select(FEATURES).to_numpy(), es["y"].to_numpy(), reference=dtr)
        booster = lgb.train(PARAMS | {"metric": "binary_logloss"}, dtr, 3000, valid_sets=[des], valid_names=["es"],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        step("stage2", f"OOF fold {f + 1}/{N_FOLDS}: best iteration {booster.best_iteration}; predicting fold ...")
        del d, trn, es, dtr, des
        fold_s1 = tr.filter(pl.col("fold") == f).select("s1")
        for k in range(PARTS):
            x = pl.read_parquet(S2 / f"feats_w{w}" / f"part{k}.parquet", columns=["s1", "m"] + FEATURES).join(fold_s1, on="s1")
            scored.append(x.select("s1", "m").with_columns(
                pl.Series("p1", booster.predict(x.select(FEATURES).to_numpy(), num_iteration=booster.best_iteration))))
    out = pl.concat(scored + [_v4_scores(w, sp.filter(~pl.col("is_train")).select(pl.col("entity_id").alias("s1")))])
    _assert_coverage(out, w)
    out.write_parquet(S2 / f"stage1_w{w}.parquet")
    step("stage2", f"stage-1 scores written: {out.height:,} pairs")


def _v4_scores(w: int, s1: pl.DataFrame) -> pl.DataFrame:
    """v4 stage-1 scores for S1s outside the training portion (holdout AND, in world 0, the
    non-holdout ghost S1s): v4 never trained on them, so they are scored exactly like test."""
    step("stage2", f"non-training S1s ({s1.height:,}): scoring with the v4 model ...")
    v4 = lgb.Booster(model_file=str(MODEL_V4))
    out = []
    for k in range(PARTS):
        x = pl.read_parquet(S2 / f"feats_w{w}" / f"part{k}.parquet", columns=["s1", "m"] + FEATURES).join(s1, on="s1")
        out.append(x.select("s1", "m").with_columns(pl.Series("p1", v4.predict(x.select(FEATURES).to_numpy()))))
    return pl.concat(out)


def _assert_coverage(st: pl.DataFrame, w: int):
    """Every candidate pair of the world universe has exactly one stage-1 score (else rival features are wrong)."""
    cand = pl.read_parquet(C.WORK / "cand" / f"train_full_w{w}.parquet", columns=["s1", "m"])
    dup = st.height - st.select("s1", "m").unique().height
    missing = cand.join(st, on=["s1", "m"], how="anti").height
    extra = st.join(cand, on=["s1", "m"], how="anti").height
    assert dup == 0 and missing == 0 and extra == 0, f"stage-1 coverage w{w}: dup {dup}, missing {missing}, extra {extra}"
    step("stage2", f"stage-1 coverage w{w} OK: {st.height:,} pairs = candidate universe, unique")


def fill(w: int):
    """Repair an existing stage1_w{w}.parquet written before the coverage fix: add v4 scores for
    world S1s that have none (world-0 non-holdout ghosts), then assert coverage."""
    st = pl.read_parquet(S2 / f"stage1_w{w}.parquet")
    sp = _split().filter(~pl.col("is_ghost")) if w == 20 else _split()
    todo = sp.filter(~pl.col("is_train")).select(pl.col("entity_id").alias("s1")).join(st.select("s1").unique(), on="s1", how="anti")
    if todo.height:
        st = pl.concat([st, _v4_scores(w, todo)])
    _assert_coverage(st, w)
    st.write_parquet(S2 / f"stage1_w{w}.parquet")


def ce_train_universe(w: int):
    """CE scores on the uncertain band for the stage-2 fit sample + holdout (never CE-training S1s)."""
    from .cross_encoder import score
    st = pl.read_parquet(S2 / f"stage1_w{w}.parquet").filter(pl.col("p1").is_between(*CE_BAND))
    sp = _split()
    fit_s1 = sp.filter(pl.col("is_train")).select(pl.col("entity_id").alias("s1")).sample(S2_TRAIN_S1, seed=C.SEED + 99)
    hold = sp.filter(pl.col("is_holdout")).select(pl.col("entity_id").alias("s1"))
    rows = st.join(pl.concat([fit_s1, hold]).unique(), on="s1").select("s1", "m")
    step("stage2", f"CE scoring {rows.height:,} band pairs (fit sample + holdout) ...")
    rows.with_columns(pl.Series("ce", score(rows, "train", name=f"ce score train band w{w}"))).write_parquet(S2 / f"ce_w{w}{SFX}.parquet")


def ce_test():
    from .cross_encoder import score
    st = pl.read_parquet(C.WORK / "pred" / "test.parquet").filter(pl.col("p").is_between(*CE_BAND)).select("s1", "m")
    step("stage2", f"CE scoring {st.height:,} test band pairs ...")
    st.with_columns(pl.Series("ce", score(st, "test", name="ce score test band"))).write_parquet(S2 / f"ce_test{SFX}.parquet")


def _with_ce(d: pl.DataFrame, ce_path) -> pl.DataFrame:
    if ce_path is None or not ce_path.exists():
        return d.with_columns(pl.lit(None, pl.Float32).alias("ce"), pl.lit(None, pl.Float32).alias("ce_minus_p1"))
    ce = pl.read_parquet(ce_path)
    return d.join(ce, on=["s1", "m"], how="left").with_columns(
        pl.col("ce").cast(pl.Float32), (pl.col("ce") - pl.col("p1")).cast(pl.Float32).alias("ce_minus_p1"))


def s2_features(stage1: pl.DataFrame, cand: pl.DataFrame, a_null: pl.DataFrame) -> pl.DataFrame:
    """Competition features from stage-1 scores (label-free). stage1: s1, m, p1."""
    d = stage1.join(cand.select(["s1", "m"] + BLK), on=["s1", "m"], how="left").join(a_null, on="m", how="left")
    p = pl.col("p1")
    d = d.with_columns(
        (p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6))).log().alias("lp1"),
        # record side
        p.rank("ordinal", descending=True).over("m").cast(pl.Float32).alias("m_rank"),
        (p >= 0.5).sum().over("m").cast(pl.Float32).alias("m_n_hi"),
        pl.len().over("m").cast(pl.Float32).alias("m_n_cand"),
        p.max().over("m").alias("_m_max"),
        # S1 side
        p.rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("s_rank"),
        p.max().over("s1").alias("s_top"),
        (p >= 0.5).sum().over("s1").cast(pl.Float32).alias("s_n_hi"),
        p.sum().over("s1").alias("s_sum"),
        pl.len().over("s1").cast(pl.Float32).alias("s_n_cand"),
    )
    # best score of this record at any OTHER S1: if this pair is the max, take the second max
    second = d.select("m", "p1").sort(["m", "p1"], descending=[False, True]).group_by("m", maintain_order=True) \
        .agg(pl.col("p1").slice(1, 1).first().alias("_m_second"))
    d = d.join(second, on="m", how="left").with_columns(
        pl.when(pl.col("m_rank") == 1).then(pl.col("_m_second").fill_null(0.0)).otherwise(pl.col("_m_max")).alias("m_best_other"))
    d = d.with_columns((p - pl.col("m_best_other")).alias("m_gap"), (pl.col("s_top") - p).alias("s_gap_top"),
                       pl.col("m_a_null").cast(pl.Float32)).drop("_m_max", "_m_second")
    if SRCF:
        hi = (p >= 0.5).cast(pl.Float32)
        src = pl.col("m").str.slice(0, 2)
        d = d.with_columns(src.alias("_src"))
        d = d.with_columns(
            ((p >= 0.5).cast(pl.Float32).sum().over(["s1", "_src"]) - hi).alias("s_n_hi_src"),     # same-source strong records, excl. this
            ((p >= 0.5).cast(pl.Float32).sum().over("s1") - (p >= 0.5).cast(pl.Float32).sum().over(["s1", "_src"])).alias("s_n_hi_oth"),
            (p.sum().over(["s1", "_src"]) - p).alias("s_sum_src"),
            p.max().over(["s1", "_src"]).alias("s_top_src"),
        )
        # among S1s competing for this record: rank of this S1 by how few same-source strong records it already holds
        d = d.with_columns(pl.col("s_n_hi_src").rank("min").over("m").cast(pl.Float32).alias("m_rank_src_rival")).drop("_src")
        return d.with_columns([pl.col(c).cast(pl.Float32) for c in S2_FEATURES + SRC_FEATURES])
    return d.with_columns([pl.col(c).cast(pl.Float32) for c in S2_FEATURES])


def _a_null(split):
    return pl.concat([pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=["entity_id", "a_null"])
                      for s in (2, 3)]).rename({"entity_id": "m", "a_null": "m_a_null"})


def s2feats(w: int):
    st = pl.read_parquet(S2 / f"stage1_w{w}.parquet")
    _assert_coverage(st, w)
    cand = pl.read_parquet(C.WORK / "cand" / f"train_full_w{w}.parquet")
    step("stage2", f"competition features over {st.height:,} pairs ...")
    _with_ce(s2_features(st, cand, _a_null("train")), S2 / f"ce_w{w}{SFX}.parquet").write_parquet(S2 / f"s2feats_w{w}{SFX}{FSFX}.parquet")


def _feat_list(d: pl.DataFrame):
    return S2_FEATURES + (SRC_FEATURES if SRCF else []) + [c for c in CE_FEATURES if c in d.columns and d[c].null_count() < d.height]


def fit(w: int = 20):
    d = pl.read_parquet(S2 / f"s2feats_w{w}{SFX}{FSFX}.parquet")
    FEATS = _feat_list(d)
    step("stage2", f"fit features: {len(FEATS)} ({'with' if 'ce' in FEATS else 'without'} cross-encoder)")
    truth = load_train_pairs().select("s1", "m").with_columns(pl.lit(1, pl.Int8).alias("y"))
    tr_s1 = _split().filter(pl.col("is_train")).select(pl.col("entity_id").alias("s1")).sample(S2_TRAIN_S1, seed=C.SEED + 99)
    if C.V6:  # reviewer 17:13 UTC: stage-2 fit S1s never trained the bi-encoder
        assert tr_s1.join(_bienc(), on="s1").height == 0, "stage-2 fit sample overlaps bi-encoder S1s"
    d = d.join(tr_s1, on="s1").join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    if XF:
        d = _with_pair(d, S2 / f"feats_w{w}")
        FEATS = FEATS + PAIR_X
        step("stage2", f"fit: + {len(PAIR_X)} direct pair features -> {len(FEATS)}")
    es_s1 = d.select("s1").unique().sort("s1").sample(fraction=0.1, seed=C.SEED)
    es, trn = d.join(es_s1, on="s1"), d.join(es_s1, on="s1", how="anti")
    step("stage2", f"fit: {trn.height:,} rows (pos {int(trn['y'].sum()):,}), early-stop {es.height:,}")
    dtr = lgb.Dataset(trn.select(FEATS).to_numpy(), trn["y"].to_numpy(), feature_name=FEATS, free_raw_data=True)
    des = lgb.Dataset(es.select(FEATS).to_numpy(), es["y"].to_numpy(), reference=dtr)
    booster = lgb.train(PARAMS | {"metric": "binary_logloss", "num_leaves": 63}, dtr, 3000, valid_sets=[des], valid_names=["es"],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    step("stage2", f"fit: best iteration {booster.best_iteration}")
    booster.save_model(str(MODEL_S2), num_iteration=booster.best_iteration)
    imp = sorted(zip(booster.feature_name(), booster.feature_importance("gain")), key=lambda x: -x[1])
    (C.EXP_RUN / f"s2_importance{MSFX}.json").write_text(json.dumps([(n, round(g, 1)) for n, g in imp], indent=1))


def evaluate(w: int = 20):
    d = pl.read_parquet(S2 / f"s2feats_w{w}{SFX}{FSFX}.parquet")
    sp = _split()
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country", "holdout_half")
    d = d.join(hold.select("s1"), on="s1")
    if XF:
        d = _with_pair(d, S2 / f"feats_w{w}")
    truth = load_train_pairs().select("s1", "m")
    booster = lgb.Booster(model_file=str(MODEL_S2))
    s2 = d.select("s1", "m").with_columns(pl.Series("p", booster.predict(d.select(booster.feature_name()).to_numpy())))
    s1only = d.select("s1", "m", pl.col("p1").alias("p"))
    rep = {"stage2": two_way(s2, truth, hold), "stage1_same_universe": two_way(s1only, truth, hold)}
    best = None
    for ta in GRID_ANY:
        for tb in GRID_ADD:
            f = summarize(per_s1_f05(decide(s2, ta, tb), truth, hold))["macro_f05"]
            if best is None or f > best[0]:
                best = (f, ta, tb)
    rep["final_params"] = {"t_any": best[1], "t_add": best[2], "holdout_f05": best[0]}
    (C.EXP_RUN / f"s2_eval_w{w}{MSFX}.json").write_text(json.dumps(rep, indent=2))
    if w == 20:  # decision file for `ber.pipeline test --steps write` (BER_DECISION)
        (C.EXP_RUN / f"decision_params_s2{MSFX}.json").write_text(json.dumps(
            {"t_any": best[1], "t_add": best[2], "holdout_f05_w20": best[0],
             "source": f"{(C.EXP_RUN / f's2_eval_w{w}{MSFX}.json').relative_to(C.ROOT)}"}, indent=2))
    s2.write_parquet(S2 / f"holdout_pred_w{w}{MSFX}.parquet")
    step("stage2", "eval w%d: stage2 %s | stage1 same universe %s | final %s" % (
        w, {k: v["report"]["macro_f05"] for k, v in rep["stage2"].items()},
        {k: v["report"]["macro_f05"] for k, v in rep["stage1_same_universe"].items()}, rep["final_params"]))


def testfeats(parts: int = 4):
    """Stage-1 pair features for the test universe, kept for the BER_S2X challenger (ber.pipeline test_score discards them)."""
    cand = pl.read_parquet(C.WORK / "cand" / "test.parquet")
    world = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id"])
    out = S2 / "feats_test"
    out.mkdir(parents=True, exist_ok=True)
    cand = cand.with_columns((pl.col("s1").hash(seed=0) % parts).alias("_part"))
    for k in range(parts):
        path = out / f"part{k}.parquet"
        if path.exists():
            continue
        ck = cand.filter(pl.col("_part") == k).drop("_part")
        build_features("test", ck, world, name=f"stage2 test features part {k + 1}/{parts}").select(["s1", "m"] + PAIR_X).write_parquet(path)


def test():
    st = pl.read_parquet(C.WORK / "pred" / "test.parquet").rename({"p": "p1"})
    cand = pl.read_parquet(C.WORK / "cand" / "test.parquet")
    d = _with_ce(s2_features(st, cand, _a_null("test")), S2 / f"ce_test{SFX}.parquet")
    booster = lgb.Booster(model_file=str(MODEL_S2))
    if XF:  # predict per test-feature part (bounded memory)
        outs = []
        for p in sorted((S2 / "feats_test").glob("part*.parquet")):
            x = pl.read_parquet(p, columns=["s1", "m"] + PAIR_X)
            dk = d.join(x.select("s1").unique(), on="s1").join(x, on=["s1", "m"], how="left")
            outs.append(dk.select("s1", "m").with_columns(pl.Series("p", booster.predict(dk.select(booster.feature_name()).to_numpy()))))
        out = pl.concat(outs)
        assert out.height == d.height, "test pair features do not cover the stage-2 universe"
    else:
        out = d.select("s1", "m").with_columns(pl.Series("p", booster.predict(d.select(booster.feature_name()).to_numpy())))
    out.write_parquet(S2 / f"test_pred{MSFX}.parquet")
    step("stage2", f"test: stage-2 scores for {out.height:,} pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--world", type=int, default=20)
    a = ap.parse_args()
    S2.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fns = {"feats": feats, "oof": oof, "fill": fill, "ce": ce_train_universe, "s2feats": s2feats, "fit": fit, "eval": evaluate,
           "cetest": lambda w: ce_test(), "testfeats": lambda w: testfeats(), "test": lambda w: test()}
    for s in a.steps.split(","):
        step("stage2", f"step {s} (world{a.world}) ...")
        fns[s](a.world)
    step("stage2", f"steps {a.steps} DONE in {time.time() - t0:.0f}s")
