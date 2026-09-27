"""Stage 3 (27 Sep): re-rank the stage-2 uncertain band with the LLM cross-encoder (ber.llm_ce).

Band: stage-2 p in [LO, HI]; outside it p is kept. Stage-3 model = LightGBM on (logit p2, logit llm, address-null flag,
rank of the pair in its S1 by p2, the S1's top p2, the record's best p2 at another S1). Trained on the labelled US/India
holdout band (test-like world: w20 minus ghost-owned address-null records) -- the holdout was never used to train
stage 1/2 or either cross-encoder -- and evaluated by cross-fitting the holdout halves (fit A -> report B, fit B -> A).

  holdout : score the holdout band with the LLM (cached), cross-fit report (macro F0.5 two-way vs stage 2)
  test    : score the test band, fit stage 3 on the whole holdout band, write cache/v6/s2/test_pred_e5base2_llm3.parquet
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config as C
from .io import load_train_pairs
from .metrics import per_s1_f05
from .model import decide
from .progress import step

LO, HI = float(os.environ.get("BER_S3_LO", "0.05")), float(os.environ.get("BER_S3_HI", "0.95"))
TAG = os.environ.get("BER_LLM_TAG", "qwen7b")
S2 = C.WORK / "s2"
FEATS = ["lp2", "lllm", "m_a_null", "rk2", "s_top2", "m_other2"]
PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200, "verbose": -1,
          "feature_fraction": 1.0, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 7, "num_threads": 16}
ROUNDS = 400


def _ctx(sc: pl.DataFrame) -> pl.DataFrame:
    """sc: s1, m, p (stage 2 over all scored pairs) -> context columns computed on stage-2 scores."""
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    sc = sc.with_columns(lg("p").alias("lp2"), pl.col("p").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("rk2"),
                         pl.col("p").max().over("s1").alias("s_top2"))
    second = sc.select("m", "p").sort(["m", "p"], descending=[False, True]).group_by("m", maintain_order=True) \
        .agg(pl.col("p").first().alias("_mx"), pl.col("p").slice(1, 1).first().fill_null(0.0).alias("_2nd"))
    return sc.join(second, on="m").with_columns(pl.when(pl.col("p") >= pl.col("_mx")).then(pl.col("_2nd")).otherwise(pl.col("_mx")).alias("m_other2")).drop("_mx", "_2nd")


def _anull(split: str) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(C.CACHE / f"{split}_source{s}.parquet") for s in (2, 3)]).select(
        pl.col("entity_id").alias("m"), pl.col("business_address").is_null().cast(pl.Float32).alias("m_a_null"))


def _llm(band: pl.DataFrame, split: str, cache) -> pl.DataFrame:
    from .llm_ce import score
    done = pl.read_parquet(cache) if cache.exists() else pl.DataFrame(schema={"s1": pl.Utf8, "m": pl.Utf8, "llm": pl.Float32})
    todo = band.select("s1", "m").join(done, on=["s1", "m"], how="anti")
    if todo.height:
        from .llm_ce import _load
        tok, model = _load(False)
        step("s3", f"LLM scoring {todo.height:,} {split} pairs -> {cache.name}")
        for i in range(0, todo.height, 100_000):  # resumable: the cache is rewritten after every chunk
            part = todo.slice(i, 100_000)
            new = part.with_columns(pl.Series("llm", score(part, split, name=f"llm {split} band {i // 100_000}", model=model, tok=tok)))
            done = pl.concat([done, new])
            done.write_parquet(cache)
    return band.join(done, on=["s1", "m"], how="left")


def _with_feats(sc: pl.DataFrame, split: str, cache) -> tuple[pl.DataFrame, pl.DataFrame]:
    d = _ctx(sc).join(_anull(split), on="m", how="left")
    band = d.filter(pl.col("p").is_between(LO, HI))
    band = _llm(band, split, cache).with_columns(
        (pl.col("llm").clip(1e-6, 1 - 1e-6) / (1 - pl.col("llm").clip(1e-6, 1 - 1e-6))).log().alias("lllm"))
    return d, band


def _apply(d: pl.DataFrame, band: pl.DataFrame, booster) -> pl.DataFrame:
    p3 = band.select("s1", "m").with_columns(pl.Series("p3", booster.predict(band.select(FEATS).to_numpy())))
    return d.join(p3, on=["s1", "m"], how="left").with_columns(pl.coalesce("p3", "p").alias("p")).select("s1", "m", "p")


def holdout():
    """Fit on the world-0 band of one holdout half (as the shipped model), report the other half in world 0 and in the
    test-like world (w20 minus ghost-owned null records; ghost labels only define that evaluation world)."""
    from sklearn.metrics import roc_auc_score
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"), "country", "holdout_half")
    truth = load_train_pairs().select("s1", "m")
    ghost = sp.filter(pl.col("is_ghost")).select(pl.col("entity_id").alias("s1"))
    gnull = truth.join(ghost, on="s1").join(_anull("train").filter(pl.col("m_a_null") == 1), on="m").select("m")
    rep = {"band": [LO, HI], "fit": "world-0 band of one holdout half"}
    W = {}
    for w in (0, 20):
        sc = pl.read_parquet(S2 / f"holdout_pred_w{w}_e5base2.parquet").join(hold.select("s1"), on="s1")
        if w == 20:
            sc = sc.join(gnull, on="m", how="anti")
        d, band = _with_feats(sc, "train", S2 / f"llm_s3band_w{w}_{TAG}.parquet")
        band = band.join(hold.select("s1", "holdout_half"), on="s1").join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
        W[w] = (d, band)
        rep[f"w{w} band pairs"] = band.height
        rep[f"w{w} band auc p2 / llm"] = [round(float(roc_auc_score(band["y"], band["p"])), 4), round(float(roc_auc_score(band["y"], band["llm"])), 4)]
    for tr_h, te_h in (("A", "B"), ("B", "A")):
        tr = W[0][1].filter(pl.col("holdout_half") == tr_h)
        bst = lgb.train(PARAMS, lgb.Dataset(tr.select(FEATS).to_numpy(), tr["y"].to_numpy()), ROUNDS)
        h = hold.filter(pl.col("holdout_half") == te_h)
        for w in (0, 20):
            d, band = W[w]
            new = _apply(d.join(h.select("s1"), on="s1"), band.filter(pl.col("holdout_half") == te_h), bst)
            base = d.join(h.select("s1"), on="s1").select("s1", "m", "p")
            for ta, tb in ((0.4, 0.75), (0.4, 0.7), (0.5, 0.75)):
                f0 = per_s1_f05(decide(base, ta, tb), truth, h)["f05"].mean()
                f1 = per_s1_f05(decide(new, ta, tb), truth, h)["f05"].mean()
                rep[f"fit{tr_h}->rep{te_h} w{w}{'nn' if w == 20 else ''} {ta}/{tb}"] = [round(f0, 5), round(f1, 5), round(f1 - f0, 5)]
        step("s3", f"fit {tr_h} -> {te_h}: " + json.dumps({k: v for k, v in rep.items() if k.startswith(f"fit{tr_h}")}))
    (C.EXP_RUN / f"llm_stage3_{TAG}_holdout.json").write_text(json.dumps(rep, indent=1))
    step("s3", "holdout report: " + json.dumps(rep))


def test():
    """Final stage 3 = the model fit on holdout half A in world 0 (no ghost labels involved); half B stays an
    independent validation (reviewer 22:41 UTC). Applied to the test band."""
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost") & (pl.col("holdout_half") == "A")).select(pl.col("entity_id").alias("s1"))
    truth = load_train_pairs().select("s1", "m")
    sc = pl.read_parquet(S2 / "holdout_pred_w0_e5base2.parquet").join(hold, on="s1")
    _, band = _with_feats(sc, "train", S2 / f"llm_s3band_w0_{TAG}.parquet")
    band = band.join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    b = lgb.train(PARAMS, lgb.Dataset(band.select(FEATS).to_numpy(), band["y"].to_numpy()), ROUNDS)
    b.save_model(str(C.ROOT / "models" / f"lgb_s3_llm_{TAG}.txt"))
    ts = pl.read_parquet(S2 / "test_pred_e5base2.parquet")
    countries = os.environ.get("BER_S3_COUNTRIES", "US,India").split(",")
    cty = pl.read_parquet(C.CACHE / "test_source1.parquet").select(pl.col("entity_id").alias("s1"), "country")
    d = _ctx(ts).join(_anull("test"), on="m", how="left")
    band = d.filter(pl.col("p").is_between(LO, HI)).join(cty.filter(pl.col("country").is_in(countries)).select("s1"), on="s1")
    tband = _llm(band, "test", S2 / f"llm_s3band_test_{TAG}.parquet").with_columns(
        (pl.col("llm").clip(1e-6, 1 - 1e-6) / (1 - pl.col("llm").clip(1e-6, 1 - 1e-6))).log().alias("lllm"))
    out = _apply(d, tband, b)
    name = f"test_pred_e5base2_llm3_{TAG}_{'_'.join(sorted(countries))}.parquet"
    out.write_parquet(S2 / name)
    step("s3", f"test: stage-3 band {tband.height:,} pairs ({countries}) -> {name}")


# ---- v2 (27 Sep): wider band + stage-2 context features; the LLM score is used where cached (band 0.05-0.95) ----
EXTRA = ["lp1", "d_cos", "ce", "ce_minus_p1", "m_n_hi", "s_n_hi", "m_gap", "s_gap_top", "r_df", "r_dr", "m_n_cand", "s_n_cand"]
V2_LO, V2_HI = 0.01, 0.995


def _v2_band(sc: pl.DataFrame, split: str, llm_cache, extra: pl.DataFrame):
    d = _ctx(sc).join(_anull(split), on="m", how="left")
    band = d.filter(pl.col("p").is_between(V2_LO, V2_HI))
    lc = pl.read_parquet(llm_cache) if llm_cache.exists() else pl.DataFrame(schema={"s1": pl.Utf8, "m": pl.Utf8, "llm": pl.Float32})
    band = band.join(lc, on=["s1", "m"], how="left").with_columns(
        (pl.col("llm").clip(1e-6, 1 - 1e-6) / (1 - pl.col("llm").clip(1e-6, 1 - 1e-6))).log().alias("lllm"))
    return d, band.join(extra, on=["s1", "m"], how="left")


def _test_extra() -> pl.DataFrame:
    from .stage2 import _a_null, _with_ce, s2_features
    path = S2 / "s2x_test_e5base2.parquet"
    if not path.exists():
        st = pl.read_parquet(C.WORK / "pred" / "test.parquet").rename({"p": "p1"})
        cand = pl.read_parquet(C.WORK / "cand" / "test.parquet")
        d = _with_ce(s2_features(st, cand, _a_null("test")), S2 / "ce_test_e5base2.parquet")
        d.select(["s1", "m"] + EXTRA).write_parquet(path)
    return pl.read_parquet(path)


def test_v2():
    """Stage 3 v2: fit on the world-0 band of holdout half A, apply to the test band (all countries)."""
    F = FEATS + EXTRA
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost") & (pl.col("holdout_half") == "A")).select(pl.col("entity_id").alias("s1"))
    truth = load_train_pairs().select("s1", "m")
    sc = pl.read_parquet(S2 / "holdout_pred_w0_e5base2.parquet").join(hold, on="s1")
    ex = pl.read_parquet(S2 / "s2feats_w0_e5base2.parquet", columns=["s1", "m"] + EXTRA)
    _, band = _v2_band(sc, "train", S2 / f"llm_s3band_w0_{TAG}.parquet", ex)
    band = band.join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    b = lgb.train(PARAMS, lgb.Dataset(band.select(F).to_numpy(), band["y"].to_numpy()), ROUNDS)
    b.save_model(str(C.ROOT / "models" / f"lgb_s3v2_llm_{TAG}.txt"))
    ts = pl.read_parquet(S2 / "test_pred_e5base2.parquet")
    d, tband = _v2_band(ts, "test", S2 / f"llm_s3band_test_{TAG}.parquet", _test_extra())
    p3 = tband.select("s1", "m").with_columns(pl.Series("p3", b.predict(tband.select(F).to_numpy())))
    out = d.join(p3, on=["s1", "m"], how="left").with_columns(pl.coalesce("p3", "p").alias("p")).select("s1", "m", "p")
    out.write_parquet(S2 / f"test_pred_e5base2_llm3v2_{TAG}.parquet")
    step("s3", f"test_v2: band {tband.height:,} pairs ({tband['llm'].is_not_null().sum():,} with LLM) -> test_pred_e5base2_llm3v2_{TAG}.parquet")


# ---- v3 (27 Sep): v2 + the BER_S2X stage-2 challenger score + the Qwen2.5-1.5B pair classifier (qwen15_r3) ----
TAG15 = "qwen15_r3"


def _v3_extra(band: pl.DataFrame, split: str, w: int = 0) -> pl.DataFrame:
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    px = pl.read_parquet(S2 / (f"holdout_pred_w{w}_e5base2_x.parquet" if split == "train" else "test_pred_e5base2_x.parquet")).rename({"p": "px"})
    l15 = pl.read_parquet(S2 / (f"llm_s3band_w{w}_{TAG15}.parquet" if split == "train" else f"llm_s3band_test_{TAG15}.parquet")).rename({"llm": "llm15"})
    return band.join(px, on=["s1", "m"], how="left").join(l15, on=["s1", "m"], how="left").with_columns(lg("px").alias("lpx"), lg("llm15").alias("lllm15"))


def test_v3():
    """Stage 3 v3: fit on the world-0 band (0.01-0.995) of holdout half A, apply to the test band (all countries)."""
    F = FEATS + EXTRA + ["lpx", "lllm15"]
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost") & (pl.col("holdout_half") == "A")).select(pl.col("entity_id").alias("s1"))
    truth = load_train_pairs().select("s1", "m")
    sc = pl.read_parquet(S2 / "holdout_pred_w0_e5base2.parquet").join(hold, on="s1")
    ex = pl.read_parquet(S2 / "s2feats_w0_e5base2.parquet", columns=["s1", "m"] + EXTRA)
    _, band = _v2_band(sc, "train", S2 / f"llm_s3band_w0_{TAG}.parquet", ex)
    band = _v3_extra(band, "train", 0).join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    b = lgb.train(PARAMS, lgb.Dataset(band.select(F).to_numpy(), band["y"].to_numpy()), ROUNDS)
    b.save_model(str(C.ROOT / "models" / f"lgb_s3v3_llm_{TAG}.txt"))
    ts = pl.read_parquet(S2 / "test_pred_e5base2.parquet")
    d, tband = _v2_band(ts, "test", S2 / f"llm_s3band_test_{TAG}.parquet", _test_extra())
    tband = _v3_extra(tband, "test")
    p3 = tband.select("s1", "m").with_columns(pl.Series("p3", b.predict(tband.select(F).to_numpy())))
    out = d.join(p3, on=["s1", "m"], how="left").with_columns(pl.coalesce("p3", "p").alias("p")).select("s1", "m", "p")
    out.write_parquet(S2 / f"test_pred_e5base2_llm3v3_{TAG}.parquet")
    step("s3", f"test_v3: band {tband.height:,} pairs ({tband['llm'].is_not_null().sum():,} 7B, {tband['llm15'].is_not_null().sum():,} 1.5B) -> test_pred_e5base2_llm3v3_{TAG}.parquet")


# ---- v4 (27 Sep): v3 with the two LLM scores also on the upper band (stage-2 p in (0.95, 0.995]) ----
def _v4_fill(band: pl.DataFrame, which: str) -> pl.DataFrame:
    lg = lambda c: (pl.col(c).clip(1e-6, 1 - 1e-6) / (1 - pl.col(c).clip(1e-6, 1 - 1e-6))).log()
    h7 = pl.read_parquet(S2 / f"llm_hiband_{which}_{TAG}.parquet").rename({"llm": "_h7"})
    h15 = pl.read_parquet(S2 / f"llm_hiband_{which}_{TAG15}.parquet").rename({"llm": "_h15"})
    return (band.join(h7, on=["s1", "m"], how="left").join(h15, on=["s1", "m"], how="left")
            .with_columns(pl.coalesce("llm", "_h7").alias("llm"), pl.coalesce("llm15", "_h15").alias("llm15")).drop("_h7", "_h15")
            .with_columns(lg("llm").alias("lllm"), lg("llm15").alias("lllm15")))


def test_v4():
    F = FEATS + EXTRA + ["lpx", "lllm15"]
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost") & (pl.col("holdout_half") == "A")).select(pl.col("entity_id").alias("s1"))
    truth = load_train_pairs().select("s1", "m")
    sc = pl.read_parquet(S2 / "holdout_pred_w0_e5base2.parquet").join(hold, on="s1")
    ex = pl.read_parquet(S2 / "s2feats_w0_e5base2.parquet", columns=["s1", "m"] + EXTRA)
    _, band = _v2_band(sc, "train", S2 / f"llm_s3band_w0_{TAG}.parquet", ex)
    band = _v4_fill(_v3_extra(band, "train", 0), "holdout")
    band = band.join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    b = lgb.train(PARAMS, lgb.Dataset(band.select(F).to_numpy(), band["y"].to_numpy()), ROUNDS)
    b.save_model(str(C.ROOT / "models" / f"lgb_s3v4_llm_{TAG}.txt"))
    ts = pl.read_parquet(S2 / "test_pred_e5base2.parquet")
    d, tband = _v2_band(ts, "test", S2 / f"llm_s3band_test_{TAG}.parquet", _test_extra())
    tband = _v4_fill(_v3_extra(tband, "test"), "test")
    p3 = tband.select("s1", "m").with_columns(pl.Series("p3", b.predict(tband.select(F).to_numpy())))
    out = d.join(p3, on=["s1", "m"], how="left").with_columns(pl.coalesce("p3", "p").alias("p")).select("s1", "m", "p")
    out.write_parquet(S2 / f"test_pred_e5base2_llm3v4_{TAG}.parquet")
    step("s3", f"test_v4: band {tband.height:,} pairs ({tband['llm'].is_not_null().sum():,} 7B, {tband['llm15'].is_not_null().sum():,} 1.5B) -> test_pred_e5base2_llm3v4_{TAG}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    a = ap.parse_args()
    for s in a.steps.split(","):
        {"holdout": holdout, "test": test, "test_v2": test_v2, "test_v3": test_v3, "test_v4": test_v4}[s]()
