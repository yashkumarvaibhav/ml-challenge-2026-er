"""Train/validate/infer driver.

  python -m er.pipe --split train --steps feats,stage1,stage2
  python -m er.pipe --split test  --steps feats,stage1,stage2,write

Train (world 20 = 20% of S1s removed, their records stay as orphans, like test):
  stage1: 2-fold cross-fitting over training-portion S1s -> out-of-fold p1 for every training pair;
          holdout pairs get the average of the two fold models (exactly how test is scored).
  stage2: competition features from p1 over the whole universe + pair features -> LightGBM; two-way
          threshold tuning on holdout halves A/B.
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from er import features as F
from er.common import CACHE, EXP, hold_s1, log, norm, split_table, truth_pairs, world_s1
from er.model import competition, context, decide, feat_cols, predict, train, two_way
from ber.metrics import per_s1_f05, summarize

import os
PARTS = 16
RUN = os.environ.get("ER_RUN", "")                                  # "" = run 1; e.g. "2" -> separate artifacts
CANDTAG = "b2_" if os.environ.get("ER_BLOCK") == "2" else ""         # candidate files from blocking v2
MODELS = CACHE.parent / f"models{RUN}"
MODELS.mkdir(exist_ok=True)
FOLD_S1 = 250_000     # training S1s per fold model
S2_S1 = int(os.environ.get("ER_S2_S1", 300_000))   # training S1s for the stage-2 model
S2TAG = os.environ.get("ER_S2TAG", "")               # suffix for a stage-2 variant's artifacts (keeps the base run intact)


ADDON = os.environ.get("ER_ADDON") == "1"                         # join TF-IDF char-cosine add-on features
ADDON_COLS = ["tf_name", "tf_core", "tf_addr", "tf_addr_w"]


def _read_part(split, k, cols=None):
    d = pl.read_parquet(_paths(split) / f"part{k}.parquet", columns=cols)
    if ADDON:
        a = pl.read_parquet(CACHE / "addon" / split / f"part{k}.parquet")
        if cols is not None:
            a = a.select(["s1", "m"] + [c for c in ADDON_COLS if c in cols or True])
        d = d.join(a, on=["s1", "m"], how="left", maintain_order="left")
    return d


def _paths(split):
    d = CACHE / os.environ.get("ER_FEATS", f"feats{RUN}") / split
    d.mkdir(parents=True, exist_ok=True)
    return d


def _crowd_tables(E1: pl.DataFrame) -> dict:
    out = {}
    for key, name in (("n_core_str", "core_cnt"), ("n_cmp", "cmp_cnt"), ("akey", "akey_cnt")):
        out[name] = (key, E1.filter(pl.col(key).is_not_null() & (pl.col(key) != "")).group_by("country", key).len().rename({"len": name}))
    return out


def _with_crowd(E: pl.DataFrame, tabs: dict, names) -> pl.DataFrame:
    for name in names:
        key, cnt = tabs[name]
        E = E.join(cnt, on=["country", key], how="left").with_columns(pl.col(name).fill_null(0).cast(pl.Int32))
    return E


def _lazy_entities(split, sources, ids: pl.DataFrame) -> pl.DataFrame:
    """Only the normalized rows whose entity_id is in ids (lazy scan keeps memory small)."""
    lf = pl.concat([pl.scan_parquet(CACHE / "norm_final" / f"{split}_s{s}.parquet").select(F.NORM_COLS) for s in sources])
    return lf.join(ids.lazy(), on="entity_id", how="semi").collect()


def step_feats(split, world=20):
    t0 = time.time()
    out = _paths(split)
    s1 = norm(split, 1, F.NORM_COLS)
    if split == "train":
        s1 = s1.join(world_s1(world), on="entity_id")
    E1 = F.entity_table(s1)
    idf = F.idf_tables(s1)
    tabs = _crowd_tables(E1)
    del s1, E1
    # records sharing a core name (supply) vs S1s sharing it (demand): label-free ghost / ambiguity signal
    rc = pl.concat([pl.scan_parquet(CACHE / "norm_final" / f"{split}_s{s}.parquet").select("country", "n_tok") for s in (2, 3)]) \
        .select("country", pl.col("n_tok").list.eval(pl.element().filter(~pl.element().is_in(list(F.LEGAL)))).list.join(" ").alias("n_core_str")) \
        .filter(pl.col("n_core_str") != "").group_by("country", "n_core_str").len().rename({"len": "rec_core_cnt"}).collect()
    tabs["rec_core_cnt"] = ("n_core_str", rc)
    cand = pl.read_parquet(CACHE / "cand" / f"{CANDTAG}{split}_w{world}.parquet")
    cand = context(cand).with_columns((pl.col("s1").hash(seed=3) % PARTS).alias("_part"))
    log(f"feats {split}: {cand.height:,} pairs, tables ready {time.time() - t0:.0f}s")
    for k in range(PARTS):
        if (out / f"part{k}.parquet").exists():
            continue
        ck = cand.filter(pl.col("_part") == k).drop("_part")
        e1 = _with_crowd(F.entity_table(_lazy_entities(split, (1,), ck.select(pl.col("s1").alias("entity_id")).unique())),
                         tabs, ("core_cnt", "akey_cnt", "rec_core_cnt"))
        e2 = _with_crowd(F.entity_table(_lazy_entities(split, (2, 3), ck.select(pl.col("m").alias("entity_id")).unique())),
                         tabs, ("core_cnt", "cmp_cnt", "akey_cnt", "rec_core_cnt"))
        f = F.build(ck, e1, e2, idf, chunk=600_000)
        f = f.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        f.write_parquet(out / f"part{k}.parquet")
        log(f"feats {split} part {k + 1}/{PARTS}: {f.height:,} rows x {f.width} cols, {time.time() - t0:.0f}s")
        del ck, e1, e2, f


def load_feats(split, cols=None, s1_filter=None):
    parts = []
    for k in range(PARTS):
        d = _read_part(split, k, cols)
        if s1_filter is not None:
            d = d.join(s1_filter, on="s1")
        parts.append(d)
    return pl.concat(parts)


def _labels(split):
    sp = split_table().select(pl.col("entity_id").alias("s1"), "is_train", "is_holdout", "holdout_half")
    return sp.with_columns((pl.col("s1").hash(seed=11) % 2).alias("fold"))


def step_stage1(split):
    t0 = time.time()
    if split == "train":
        truth = truth_pairs().with_columns(pl.lit(1, pl.Int8).alias("y"))
        lab = _labels(split)
        boosters = []
        for fold in (0, 1):
            pick = lab.filter(pl.col("is_train") & (pl.col("fold") == fold)).select("s1").sample(n=FOLD_S1, seed=fold, shuffle=True) \
                if lab.filter(pl.col("is_train") & (pl.col("fold") == fold)).height > FOLD_S1 else \
                lab.filter(pl.col("is_train") & (pl.col("fold") == fold)).select("s1")
            d = load_feats(split, s1_filter=pick).join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
            feats = feat_cols(d)
            log(f"stage1 fold {fold}: training on {pick.height:,} S1")
            b = train(d, feats, log=log)
            b.save_model(str(MODELS / f"s1_fold{fold}.txt"), num_iteration=b.best_iteration)
            boosters.append(b)
            del d
        json.dump(feats, open(MODELS / "s1_feats.json", "w"))
        imp = sorted(zip(feats, boosters[0].feature_importance("gain")), key=lambda x: -x[1])
        (EXP / f"s1{RUN}_importance.json").write_text(json.dumps([(n, round(float(g), 1)) for n, g in imp], indent=1))
    feats = json.load(open(MODELS / "s1_feats.json"))
    boosters = [lgb.Booster(model_file=str(MODELS / f"s1_fold{f}.txt")) for f in (0, 1)]
    out = []
    lab = _labels("train") if split == "train" else None
    for k in range(PARTS):
        d = _read_part(split, k, ["s1", "m"] + [f for f in feats if f not in ADDON_COLS])
        p0, p1 = predict(boosters[0], d, feats), predict(boosters[1], d, feats)
        if split == "train":
            fl = d.select("s1").join(lab.select("s1", "fold", "is_train"), on="s1", how="left", maintain_order="left")
            oof = np.where(fl["fold"].to_numpy() == 0, p1, p0)  # fold-0 S1s scored by the fold-1 model
            p = np.where(fl["is_train"].to_numpy(), oof, (p0 + p1) / 2)
        else:
            p = (p0 + p1) / 2
        out.append(d.select("s1", "m").with_columns(pl.Series("p1", p.astype(np.float32))))
        log(f"stage1 score {split} part {k + 1}/{PARTS} {time.time() - t0:.0f}s")
    sc = pl.concat(out)
    sc.write_parquet(CACHE / f"p1{RUN}_{split}.parquet")
    if split == "train":
        evaluate(sc, "p1", "stage1")


def evaluate(sc, p, tag):
    truth = truth_pairs()
    hold = hold_s1()
    h = sc.join(hold.select("s1"), on="s1")
    rep = two_way(h, truth, hold, p)
    log(f"{tag} holdout: " + json.dumps({k: (v.get("macro_f05") or v.get("f05"), v.get("by_country"), v["t_any"], v["t_add"]) for k, v in rep.items()}))
    (EXP / f"{tag}{RUN}_holdout.json").write_text(json.dumps(rep, indent=1))
    return rep


S2_EXTRA = ["n_tset", "n_ratio", "cmp_ratio", "a_tset", "n_cov1", "n_cov2", "a_cov1", "a_cov2", "n_miss1_max", "n_miss2_max",
            "n_soft1", "n_soft2", "num_shared", "num_miss1", "num_miss2", "a_null2", "src3", "ini_eq", "n_domain2", "n_native2",
            "bscore", "r_rank", "s_rank", "s_ncand", "m_ncand"]


def _competition(split):
    """(s1, m, p1) for the whole universe -> competition features (s1/m categorical-free, float32)."""
    sc = pl.read_parquet(CACHE / f"p1{RUN}_{split}.parquet")
    d = competition(sc, "p1").with_columns(
        (pl.col("p1").clip(1e-6, 1 - 1e-6) / (1 - pl.col("p1").clip(1e-6, 1 - 1e-6))).log().alias("lp1"))
    return d.with_columns(pl.col(pl.Float64).cast(pl.Float32))


SUPPORT = RUN in ("2", "3")   # record-cluster support features (need _ch1/_ch2 hashes, written by run-2 features)


def _support(x: pl.DataFrame, comp: pl.DataFrame) -> pl.DataFrame:
    """Within an S1's candidate list: how many OTHER records share this record's core name, and how many of those
    the stage-1 model likes; plus whether this record's core differs from the S1's (sibling-cluster evidence)."""
    x = x.join(comp.select("s1", "m", "p1"), on=["s1", "m"], how="left")
    x = x.with_columns(
        (pl.len().over(["s1", "_ch2"]) - 1).cast(pl.Int16).alias("sup_same_core"),
        ((pl.col("p1") >= 0.5).sum().over(["s1", "_ch2"]) - (pl.col("p1") >= 0.5).cast(pl.UInt32)).cast(pl.Int16).alias("sup_same_core_hi"),
        (pl.col("_ch1") != pl.col("_ch2")).cast(pl.Int8).alias("core_diff"),
        (pl.col("_ch1") == pl.col("_ch2")).sum().over("s1").cast(pl.Int16).alias("s_n_core_eq"),
    )
    return x.drop("_ch1", "_ch2", "p1")


def _parts_with_extra(split, comp: pl.DataFrame, s1_filter=None):
    """Yield per-part stage-2 tables: competition features joined with the part's pair features."""
    cols = ["s1", "m"] + S2_EXTRA + (["_ch1", "_ch2"] if SUPPORT else [])
    for k in range(PARTS):
        x = _read_part(split, k, cols)
        if SUPPORT:
            x = _support(x, comp)   # parts are S1-partitioned, so per-S1 windows are complete
        if s1_filter is not None:
            x = x.join(s1_filter, on="s1")
        yield comp.join(x, on=["s1", "m"], how="inner")


def step_stage2(split):
    t0 = time.time()
    comp = _competition(split)
    log(f"stage2 {split}: competition features for {comp.height:,} pairs ({time.time() - t0:.0f}s)")
    if split == "train":
        truth = truth_pairs().with_columns(pl.lit(1, pl.Int8).alias("y"))
        lab = _labels(split)
        tr_s1 = lab.filter(pl.col("is_train")).select("s1").sample(n=S2_S1, seed=5)
        dt = pl.concat(list(_parts_with_extra(split, comp, tr_s1))).join(truth, on=["s1", "m"], how="left") \
            .with_columns(pl.col("y").fill_null(0))
        feats = [c for c in dt.columns if c not in ("s1", "m", "y")]
        log(f"stage2: training on {tr_s1.height:,} S1, {dt.height:,} rows ({time.time() - t0:.0f}s)")
        b = train(dt, feats, log=log)
        del dt
        b.save_model(str(MODELS / f"s2{S2TAG}.txt"), num_iteration=b.best_iteration)
        json.dump(feats, open(MODELS / f"s2{S2TAG}_feats.json", "w"))
        imp = sorted(zip(feats, b.feature_importance("gain")), key=lambda x: -x[1])
        (EXP / f"s2{RUN}_importance.json").write_text(json.dumps([(n, round(float(g), 1)) for n, g in imp], indent=1))
        h = pl.concat(list(_parts_with_extra(split, comp, hold_s1().select("s1"))))
        sc = h.select("s1", "m").with_columns(pl.Series("p", predict(b, h, feats)))
        sc.write_parquet(CACHE / f"p2{RUN}{S2TAG}_train_holdout.parquet")
        rep = evaluate(sc, "p", "stage2")
        (MODELS / f"decision{S2TAG}.json").write_text(json.dumps({"t_any": rep["full"]["t_any"], "t_add": rep["full"]["t_add"]}))
    else:
        feats = json.load(open(MODELS / f"s2{S2TAG}_feats.json"))
        b = lgb.Booster(model_file=str(MODELS / f"s2{S2TAG}.txt"))
        outs = []
        for k, d in enumerate(_parts_with_extra(split, comp)):
            outs.append(d.select("s1", "m").with_columns(pl.Series("p", predict(b, d, feats))))
            log(f"stage2 test part {k + 1}/{PARTS} ({time.time() - t0:.0f}s)")
        sc = pl.concat(outs)
        sc.write_parquet(CACHE / f"p2{RUN}{S2TAG}_{split}.parquet")
        log(f"stage2 test scored {sc.height:,} pairs")


def step_write(split="test", tag=None):
    tag = tag or f"run{RUN or 1}"
    from er.post import category_veto
    from er.write import write
    params = json.loads((MODELS / "decision.json").read_text())
    sc = pl.read_parquet(CACHE / f"p2{RUN}_test.parquet")
    pred = decide(sc, params["t_any"], params["t_add"], "p")
    kept, dropped, rep = category_veto(pred, "test")
    cand = pl.read_parquet(CACHE / "cand" / f"{CANDTAG}test_w20.parquet", columns=["s1", "m"])
    out = CACHE.parent / "output" / tag
    write(kept, cand, out)
    (out / "meta.json").write_text(json.dumps({"params": params, "veto": {k: {kk: vv for kk, vv in v.items() if kk != "words"}
                                                                         for k, v in rep.items()}}, indent=1))
    pred.write_parquet(out / "pred_noveto.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--steps", required=True)
    a = ap.parse_args()
    for s in a.steps.split(","):
        log(f"=== {a.split}: {s}")
        {"feats": step_feats, "stage1": step_stage1, "stage2": step_stage2, "write": step_write}[s](a.split)
