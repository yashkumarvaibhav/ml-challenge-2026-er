"""Run the unchanged v6 pipeline on the synthetic France split `synfr` (ber.synth_france) and evaluate with its labels.

Steps (resumable; artifacts under cache/v6/synfr/):
  norm    : normalize raw + final (France self-maps copied from the test split)
  dense   : tokenize + embed S1 and pool with the fine-tuned bi-encoder
  block   : v6 candidate generation (world = all synfr S1s, as on test)
  score   : stage-1 features + lgb_v6 scores
  s2      : competition features + standard CE (e5base2) on the stage-1 band + stage-2 (lgb_s2_v6_e5base2)
  eval    : macro F0.5 vs truth at the packaged thresholds, with/without the France category-swap rule, error slices
  BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2 python -m ber.synfr_run --steps norm,dense,block,score,s2,eval
"""
import argparse
import json
import os
import shutil

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config as C
from .progress import step

SPLIT = os.environ.get("BER_SYN", "synfr")  # synthetic split name (stress test: a second simulator)
OUT = C.WORK / SPLIT


def norm():
    from .normalize import run
    src = C.CACHE / "maps" / "self_maps_test.json"
    dst = C.CACHE / "maps" / f"self_maps_{SPLIT}.json"
    if not dst.exists():
        shutil.copyfile(src, dst)
    run("raw", SPLIT)
    run("final", SPLIT)


def dense():
    import torch
    from transformers import AutoTokenizer
    from . import dense as D
    tok = AutoTokenizer.from_pretrained(D.BASE)
    D.DATA.mkdir(parents=True, exist_ok=True)
    D.EMB.mkdir(parents=True, exist_ok=True)
    dev = D.device()
    model = D.Encoder().to(dev)
    model.load_state_dict(torch.load(D.OUT / "model.pt", map_location=dev))
    model.eval()
    for which in ("s1", "pool"):
        t = D._texts(SPLIT, which)
        txt = t["txt"].to_list()
        n = len(txt)
        ids = np.full((n, D.MAX_LEN), tok.pad_token_id, np.int32)
        lens = np.zeros(n, np.int16)
        for b in range(0, n, 200_000):
            enc = tok(txt[b:b + 200_000], truncation=True, max_length=D.MAX_LEN, padding="max_length", return_tensors="np")
            ids[b:b + 200_000] = enc["input_ids"]
            lens[b:b + 200_000] = enc["attention_mask"].sum(1)
        t.select("entity_id", "country").write_parquet(D.DATA / f"ids_{SPLIT}_{which}.parquet")
        np.savez(D.DATA / f"tok_{SPLIT}_{which}.npz", ids=ids, lens=lens)
        order = np.argsort(lens, kind="stable")
        out = np.empty((n, model.enc.config.hidden_size), np.float16)
        with torch.no_grad():
            for b in range(0, n, 8192):
                idx = order[b:b + 8192]
                x, m = D._gather(ids, lens, idx, dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out[idx] = model(x, m).half().cpu().numpy()
        np.save(D.EMB / f"{SPLIT}_{which}.npy", out)
        step("synfr", f"dense {which}: {n:,} embedded")


def block():
    from .blocking_v6 import block as block_v6
    from .io import load_source
    s1 = load_source(SPLIT, 1).select("entity_id")
    cand = block_v6(SPLIT, s1, s1)
    OUT.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(OUT / "cand.parquet")


def score(parts: int = 2):
    from .features import build as build_features
    from .model import predict
    from .io import load_source
    booster = lgb.Booster(model_file=str(C.ROOT / "models" / "lgb_v6.txt"))
    cand = pl.read_parquet(OUT / "cand.parquet").with_columns((pl.col("s1").hash(seed=0) % parts).alias("_part"))
    world = load_source(SPLIT, 1).select("entity_id")
    out = []
    for k in range(parts):
        f = build_features(SPLIT, cand.filter(pl.col("_part") == k).drop("_part"), world, name=f"synfr features part {k + 1}/{parts}")
        out.append(f.select("s1", "m").with_columns(pl.Series("p", predict(booster, f))))
        del f
    pl.concat(out).write_parquet(OUT / "stage1.parquet")


def s2():
    from .cross_encoder import score as ce_score
    from .stage2 import CE_BAND, MODEL_S2, _a_null, _with_ce, s2_features
    st = pl.read_parquet(OUT / "stage1.parquet").rename({"p": "p1"})
    cand = pl.read_parquet(OUT / "cand.parquet")
    band = st.filter(pl.col("p1").is_between(*CE_BAND)).select("s1", "m")
    ce_path = OUT / "ce_band_e5base2.parquet"
    if not ce_path.exists():
        band.with_columns(pl.Series("ce", ce_score(band, SPLIT, name="ce score synfr band"))).write_parquet(ce_path)
    d = _with_ce(s2_features(st, cand, _a_null(SPLIT)), ce_path)
    booster = lgb.Booster(model_file=str(MODEL_S2))
    d.select("s1", "m").with_columns(pl.Series("p", booster.predict(d.select(booster.feature_name()).to_numpy()))).write_parquet(OUT / "pred_s2.parquet")


def evaluate():
    from .metrics import per_s1_f05
    from .model import decide
    from .experiments.france_category_swap import classify, toks
    truth = pl.read_parquet(C.CACHE / f"{SPLIT}_truth.parquet")
    s1 = pl.read_parquet(C.CACHE / f"{SPLIT}_source1.parquet").select(pl.col("entity_id").alias("s1"), "country", pl.col("business_name").alias("n1"))
    rec = pl.concat([pl.read_parquet(C.CACHE / f"{SPLIT}_source{s}.parquet") for s in (2, 3)]).select(pl.col("entity_id").alias("m"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
    cats = set(json.loads((C.EXP_RUN / "france_category_swap.json").read_text())["test France"]["category_words"])
    rep = {}
    cand = pl.read_parquet(OUT / "cand.parquet").select("s1", "m")
    rep["oracle_pair_recall"] = round(truth.join(cand, on=["s1", "m"]).height / truth.height, 4)
    for name, f in (("stage1", OUT / "stage1.parquet"), ("stage2", OUT / "pred_s2.parquet")):
        sc = pl.read_parquet(f)
        d = decide(sc, 0.4, 0.75)
        per = per_s1_f05(d, truth, s1.select("s1", "country"))
        acc = d.join(s1, on="s1").join(rec, on="m")
        sw = pl.Series([bool((c := classify(a, b)) and c[0] and c[1] in cats) for a, b in zip(acc["n1"], acc["n2"])])
        d2 = acc.filter(~sw).select("s1", "m")
        per2 = per_s1_f05(d2, truth, s1.select("s1", "country"))
        swp = acc.filter(sw).join(truth.with_columns(pl.lit(1).alias("y")), on=["s1", "m"], how="left")
        t = truth.group_by("s1").len().rename({"len": "T"})
        k = d2.group_by("s1").len().rename({"len": "k"})
        h = s1.select("s1").join(k, on="s1", how="left").join(t, on="s1", how="left").fill_null(0)
        tp = d2.join(truth, on=["s1", "m"]).height
        rep[name] = {"macro_f05": round(per["f05"].mean(), 5), "with_category_rule": round(per2["f05"].mean(), 5),
                     "swap_pairs_accepted": acc.filter(sw).height, "swap_true_rate": round(swp["y"].fill_null(0).mean(), 4) if swp.height else None,
                     "pred_k0": round(100 * (h["k"] == 0).mean(), 2), "true_k0": round(100 * (h["T"] == 0).mean(), 2),
                     "pred_k1": round(100 * (h["k"] == 1).mean(), 2), "pred_mean": round(h["k"].mean(), 3), "true_mean": round(h["T"].mean(), 3),
                     "precision": round(tp / d2.height, 4), "recall": round(tp / truth.height, 4)}
        step("synfr", f"eval {name}: {rep[name]}")
    rep["oracle"] = rep.pop("oracle_pair_recall")
    (C.EXP_RUN / f"{SPLIT}_eval.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fns = {"norm": norm, "dense": dense, "block": block, "score": score, "s2": s2, "eval": evaluate}
    for s in a.steps.split(","):
        step("synfr", f"step {s} ...")
        fns[s]()
