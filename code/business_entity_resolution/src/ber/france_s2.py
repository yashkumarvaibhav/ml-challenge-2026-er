"""France-aware stage 2 trained on the labelled synthetic France world (ber.synth_france / ber.synfr_run).

The v6 stage-2 LightGBM learned from US/India, where a one-word substitution at the same address is always the
generator's filler noise. France has sibling businesses (same stem and address, another category word), which
US/India labels never show. Here explicit, label-free sibling features are added to the standard stage-2 features
and a France model is fitted on synthetic France S1s (half A), evaluated on the other half (B), then applied to
the real France test universe. US/India are untouched.

Extra features (computed only for pairs with p1 >= P_MIN; others 0):
  f_swap      record swaps an S1 category word for another category word (strict or typo/filler-tolerant)
  f_cat_new   record contains a category word the S1 does not have (tokens or inside compact/domain strings)
  f_cat_drop  an S1 category word is missing from the record (dropped or replaced)
  f_fill_sub  a French filler word replaced an S1 word
  f_sib_s1    number of the S1's candidates (p1 >= 0.5) with f_swap = 1 (sibling evidence at this S1)
  f_sib_m     number of OTHER S1s at which this record has p1 >= 0.5

  BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2 python -m ber.france_s2 --steps feats_syn,fit,eval_syn,feats_test,apply
"""
import argparse
import json
import os
import re
import unicodedata

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from . import config as C
from .metrics import per_s1_f05
from .model import PARAMS, decide
from .progress import step
from .stage2 import CE_FEATURES, MODEL_S2, S2, S2_FEATURES, SFX, _a_null, _with_ce, s2_features

SYN = os.environ.get("BER_SYN", "synfr")
OUT = C.WORK / SYN
P_MIN = 0.005
LEG = set("inc llc ltd pvt private limited corp co corporation company pc pllc llp lp sarl sas sa sasu eurl ei snc sci s a r l".split())
FILL = {"services", "service", "france", "fils", "associes", "developpement", "groupe", "international", "participations",
        "distribution", "holding", "cie", "et", "enterprises", "trading"}
FEATS_FR = ["f_swap", "f_cat_new", "f_cat_drop", "f_fill_sub", "f_sib_s1", "f_sib_m"]
MODEL_FR = C.ROOT / "models" / f"lgb_s2_v6{SFX}_france{os.environ.get('BER_FR_MODEL', '')}.txt"  # BER_FR_MODEL: variant suffix


def _cats():
    return set(json.loads((C.EXP_RUN / "france_category_swap.json").read_text())["test France"]["category_words"])


def _toks(s):
    if s is None:
        return []
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return [t for t in re.split(r"[^a-z0-9]+", s) if t and t not in LEG]


def _pair_feats(n1s, n2s, cats):
    catl = sorted(cats, key=len, reverse=True)

    def iscat(w):
        if w in cats:
            return True
        if len(w) < 4:
            return False
        r = process.extractOne(w, catl, scorer=fuzz.ratio)
        return r is not None and r[1] >= 85

    out = np.zeros((len(n1s), 4), np.float32)
    for i, (n1, n2) in enumerate(zip(n1s, n2s)):
        t1, t2 = _toks(n1), _toks(n2)
        if not t1 or not t2:
            continue
        t1c = [t for t in t1 if t not in FILL]
        t2c = [t for t in t2 if t not in FILL]
        new = [w for w in t2c if (not t1c or max(fuzz.ratio(w, a) for a in t1c) < 75) and iscat(w)]
        if len(t2) == 1 and len(t2[0]) >= 8:  # compact / domain string: look for a foreign category word inside
            rest = t2[0]
            for a in sorted(t1, key=len, reverse=True):
                if len(a) >= 3:
                    rest = rest.replace(a, "|")
            if any(len(c) >= 4 and c in rest for c in catl):
                new = new or ["compact"]
        miss = [w for w in t1c if iscat(w) and (not t2 or max(fuzz.ratio(w, b) for b in t2) < 75)]
        fsub = [w for w in t2 if w in FILL and w not in t1]
        out[i, 0] = float(bool(new and miss))
        out[i, 1] = float(bool(new))
        out[i, 2] = float(bool(miss))
        out[i, 3] = float(bool(fsub and miss))
    return out


def add_france_feats(d: pl.DataFrame, split: str) -> pl.DataFrame:
    """d: stage-2 feature frame (s1, m, p1, ...). Adds FEATS_FR."""
    cats = _cats()
    s1 = pl.read_parquet(C.CACHE / f"{split}_source1.parquet", columns=["entity_id", "business_name"]).rename({"entity_id": "s1", "business_name": "n1"})
    rec = pl.concat([pl.read_parquet(C.CACHE / f"{split}_source{s}.parquet", columns=["entity_id", "business_name"]) for s in (2, 3)]).rename({"entity_id": "m", "business_name": "n2"})
    sub = d.filter(pl.col("p1") >= P_MIN).select("s1", "m").join(s1, on="s1").join(rec, on="m")
    step("france_s2", f"{split}: text features for {sub.height:,} pairs (p1 >= {P_MIN})")
    f = _pair_feats(sub["n1"].to_list(), sub["n2"].to_list(), cats)
    sub = sub.select("s1", "m").with_columns([pl.Series(n, f[:, i]) for i, n in enumerate(FEATS_FR[:4])])
    d = d.join(sub, on=["s1", "m"], how="left").with_columns([pl.col(c).fill_null(0.0) for c in FEATS_FR[:4]])
    d = d.with_columns(
        (((pl.col("f_swap") > 0) & (pl.col("p1") >= 0.5)).cast(pl.Float32).sum().over("s1")).alias("f_sib_s1"),
        ((pl.col("p1") >= 0.5).cast(pl.Float32).sum().over("m") - (pl.col("p1") >= 0.5).cast(pl.Float32)).alias("f_sib_m"))
    return d.with_columns([pl.col(c).cast(pl.Float32) for c in FEATS_FR])


def feats_syn():
    st = pl.read_parquet(OUT / "stage1.parquet").rename({"p": "p1"})
    cand = pl.read_parquet(OUT / "cand.parquet")
    d = _with_ce(s2_features(st, cand, _a_null(SYN)), OUT / "ce_band_e5base2.parquet")
    add_france_feats(d, SYN).write_parquet(OUT / "s2feats_fr.parquet")


def _halves():
    s1 = pl.read_parquet(C.CACHE / f"{SYN}_source1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"})
    return s1.with_columns(pl.when((pl.col("s1").hash(seed=5) % 2) == 0).then(pl.lit("A")).otherwise(pl.lit("B")).alias("half"))


def _feat_names(d):
    return S2_FEATURES + [c for c in CE_FEATURES if c in d.columns] + FEATS_FR


def fit():
    d = pl.read_parquet(OUT / "s2feats_fr.parquet")
    truth = pl.read_parquet(C.CACHE / f"{SYN}_truth.parquet").with_columns(pl.lit(1, pl.Int8).alias("y"))
    h = _halves()
    d = d.join(h.filter(pl.col("half") == "A").select("s1"), on="s1").join(truth, on=["s1", "m"], how="left").with_columns(pl.col("y").fill_null(0))
    F = _feat_names(d)
    es_s1 = d.select("s1").unique().sort("s1").sample(fraction=0.1, seed=C.SEED)
    es, trn = d.join(es_s1, on="s1"), d.join(es_s1, on="s1", how="anti")
    dtr = lgb.Dataset(trn.select(F).to_numpy(), trn["y"].to_numpy(), feature_name=F)
    des = lgb.Dataset(es.select(F).to_numpy(), es["y"].to_numpy(), reference=dtr)
    b = lgb.train(PARAMS | {"metric": "binary_logloss", "num_leaves": 63}, dtr, 3000, valid_sets=[des], valid_names=["es"],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    b.save_model(str(MODEL_FR), num_iteration=b.best_iteration)
    imp = sorted(zip(b.feature_name(), b.feature_importance("gain")), key=lambda x: -x[1])
    step("france_s2", f"fit: {trn.height:,} rows, best iteration {b.best_iteration}; top {[(n, round(g)) for n, g in imp[:10]]}")


def eval_cross():
    """Stress test (reviewer 16:45): a France model fitted on simulator `synfr` scored on ANOTHER simulator (BER_SYN),
    all of its S1s, against the unchanged pipeline + category rule on the same simulator."""
    from .experiments.france_category_swap import classify
    d = pl.read_parquet(OUT / "s2feats_fr.parquet")
    truth = pl.read_parquet(C.CACHE / f"{SYN}_truth.parquet")
    s1 = pl.read_parquet(C.CACHE / f"{SYN}_source1.parquet", columns=["entity_id"]).rename({"entity_id": "s1"}).with_columns(pl.lit("France").alias("country"))
    fr = lgb.Booster(model_file=str(MODEL_FR))
    new = d.select("s1", "m").with_columns(pl.Series("p", fr.predict(d.select(fr.feature_name()).to_numpy())))
    cats = _cats()
    n1 = pl.read_parquet(C.CACHE / f"{SYN}_source1.parquet", columns=["entity_id", "business_name"]).rename({"entity_id": "s1", "business_name": "n1"})
    n2 = pl.concat([pl.read_parquet(C.CACHE / f"{SYN}_source{k}.parquet", columns=["entity_id", "business_name"]) for k in (2, 3)]).rename({"entity_id": "m", "business_name": "n2"})
    bd = decide(pl.read_parquet(OUT / "pred_s2.parquet"), 0.4, 0.75).join(n1, on="s1").join(n2, on="m")
    sw = pl.Series([bool((c := classify(a, b)) and c[0] and c[1] in cats) for a, b in zip(bd["n1"], bd["n2"])])
    b8 = bd.filter(~sw).select("s1", "m")
    f = lambda x: round(per_s1_f05(x, truth, s1)["f05"].mean(), 5)
    rep = {"baseline": f(bd.select("s1", "m")), "baseline + category rule (#8 analog)": f(b8),
           "France model 0.5/0.7": f(decide(new, 0.5, 0.7)), "France model 0.4/0.75": f(decide(new, 0.4, 0.75))}
    x = b8.join(new, on=["s1", "m"])
    for t in (0.1, 0.3, 0.5):
        rep[f"#8 analog, remove if France-model p<{t}"] = f(x.filter(pl.col("p") >= t).select("s1", "m"))
    (C.EXP_RUN / f"france_s2_cross_{SYN}.json").write_text(json.dumps(rep, indent=1))
    step("france_s2", f"cross-simulator eval on {SYN}: {rep}")


def eval_syn():
    from .experiments.france_category_swap import classify
    d = pl.read_parquet(OUT / "s2feats_fr.parquet")
    truth = pl.read_parquet(C.CACHE / "synfr_truth.parquet")
    h = _halves()
    s1b = h.filter(pl.col("half") == "B").select("s1").with_columns(pl.lit("France").alias("country"))
    s1a = h.filter(pl.col("half") == "A").select("s1").with_columns(pl.lit("France").alias("country"))
    fr = lgb.Booster(model_file=str(MODEL_FR))
    base = pl.read_parquet(OUT / "pred_s2.parquet")
    new = d.select("s1", "m").with_columns(pl.Series("p", fr.predict(d.select(fr.feature_name()).to_numpy())))
    rep = {}
    # baseline + post-hoc strict category rule, both at packaged thresholds, on half B
    cats = _cats()
    names1 = pl.read_parquet(C.CACHE / "synfr_source1.parquet", columns=["entity_id", "business_name"]).rename({"entity_id": "s1", "business_name": "n1"})
    names2 = pl.concat([pl.read_parquet(C.CACHE / f"synfr_source{s}.parquet", columns=["entity_id", "business_name"]) for s in (2, 3)]).rename({"entity_id": "m", "business_name": "n2"})
    bd = decide(base.join(s1b.select("s1"), on="s1"), 0.4, 0.75).join(names1, on="s1").join(names2, on="m")
    sw = pl.Series([bool((c := classify(a, b)) and c[0] and c[1] in cats) for a, b in zip(bd["n1"], bd["n2"])])
    rep["B baseline"] = round(per_s1_f05(bd.select("s1", "m"), truth, s1b)["f05"].mean(), 5)
    rep["B baseline + category rule"] = round(per_s1_f05(bd.filter(~sw).select("s1", "m"), truth, s1b)["f05"].mean(), 5)
    best = None
    na = new.join(s1a.select("s1"), on="s1")
    for ta in (0.3, 0.4, 0.5, 0.6):
        for tb in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
            f = per_s1_f05(decide(na, ta, tb), truth, s1a)["f05"].mean()
            if best is None or f > best[0]:
                best = (f, ta, tb)
    nb = new.join(s1b.select("s1"), on="s1")
    rep["France model, thresholds tuned on A"] = {"t_any": best[1], "t_add": best[2], "A": round(best[0], 5),
                                                  "B": round(per_s1_f05(decide(nb, best[1], best[2]), truth, s1b)["f05"].mean(), 5)}
    rep["France model, packaged 0.4/0.75, B"] = round(per_s1_f05(decide(nb, 0.4, 0.75), truth, s1b)["f05"].mean(), 5)
    (C.EXP_RUN / "france_s2_synfr_eval.json").write_text(json.dumps(rep, indent=1))
    step("france_s2", f"eval synfr: {rep}")


def feats_test():
    fr = pl.read_parquet(C.CACHE / "test_source1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"))
    st = pl.read_parquet(C.WORK / "pred" / "test.parquet").rename({"p": "p1"})
    cand = pl.read_parquet(C.WORK / "cand" / "test.parquet")
    d = _with_ce(s2_features(st, cand, _a_null("test")), S2 / f"ce_test{SFX}.parquet").join(fr, on="s1")
    add_france_feats(d, "test").write_parquet(S2 / "s2feats_test_france_fr.parquet")


def apply():
    d = pl.read_parquet(S2 / "s2feats_test_france_fr.parquet")
    fr = lgb.Booster(model_file=str(MODEL_FR))
    d.select("s1", "m").with_columns(pl.Series("p", fr.predict(d.select(fr.feature_name()).to_numpy()))).write_parquet(S2 / f"test_pred{SFX}_france_model{os.environ.get('BER_FR_MODEL', '')}.parquet")
    step("france_s2", f"applied France model to {d.height:,} real France pairs")


def write(base: str, t_any: float = 0.5, t_add: float = 0.7):
    """output/ = archived `base` with France rows from the France model (t_any/t_add, thresholds tuned on synthetic
    half A), then the France street rule and category-swap rule applied (both confirmed on public). US/India untouched."""
    import shutil
    from .experiments.france_category_swap import classify
    from .experiments.france_street_diff import street_diff
    arch = C.ROOT / "submissions" / "archive" / base
    m = pl.read_csv(arch / "matching_results.tsv", separator="\t", quote_char=None, infer_schema_length=0)
    c1, c2 = m.columns
    t1 = pl.read_parquet(C.CACHE / "test_source1.parquet").filter(pl.col("country") == "France").select(
        pl.col("entity_id").alias("s1"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"))
    rec = pl.concat([pl.read_parquet(C.CACHE / f"test_source{k}.parquet") for k in (2, 3)]).select(
        pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    cats = _cats()
    d = decide(pl.read_parquet(S2 / f"test_pred{SFX}_france_model{os.environ.get('BER_FR_MODEL', '')}.parquet"), t_any, t_add).join(t1, on="s1").join(rec, on="m")
    keep = pl.Series([not street_diff(a, b) and not ((c := classify(x, y)) and c[0] and c[1] in cats)
                      for a, b, x, y in zip(d["a1"], d["a2"], d["n1"], d["n2"])])
    d = d.filter(keep).select("s1", "m")
    g = d.group_by("s1").agg(pl.col("m").sort().str.join(",").alias("_new")).rename({"s1": c1})
    frc = t1.select(pl.col("s1").alias(c1)).with_columns(pl.lit(True).alias("_fr"))
    out = m.join(frc, on=c1, how="left").join(g, on=c1, how="left").with_columns(
        pl.when(pl.col("_fr")).then(pl.col("_new").fill_null("")).otherwise(pl.col(c2).fill_null("")).alias(c2)).select(c1, c2)
    assert out[c1].to_list() == m[c1].to_list()
    C.OUTPUT.mkdir(parents=True, exist_ok=True)
    out.write_csv(C.OUTPUT / "matching_results.tsv", separator="\t", quote_style="never")
    shutil.copyfile(arch / "candidate_pairs.tsv", C.OUTPUT / "candidate_pairs.tsv")
    step("france_s2", f"write: France rows from the France model ({t_any}/{t_add}) + street + swap rules; {d.height:,} France pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    ap.add_argument("--base", default="sub-2026-09-27-2")
    a = ap.parse_args()
    fns = {"feats_syn": feats_syn, "fit": fit, "eval_syn": eval_syn, "eval_cross": eval_cross, "feats_test": feats_test, "apply": apply, "write": lambda: write(a.base)}
    for s in a.steps.split(","):
        step("france_s2", f"step {s} ...")
        fns[s]()
