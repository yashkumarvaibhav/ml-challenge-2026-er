"""End-to-end runner with resumable steps.

Validation (train split):
  python -m ber.pipeline train --steps block,features,fit,eval
Test inference + submission files:
  python -m ber.pipeline test --steps block,score,write

Artifacts: cache/cand, cache/feats, cache/pred (parquet), models/, experiments/*.json,
output/{matching_results,candidate_pairs}.tsv.
"""
import argparse
import json
import time

import lightgbm as lgb
import polars as pl

from . import config as C
from .features import build as build_features
from .io import load_source, load_train_pairs
from .metrics import per_s1_f05, summarize
from .model import GRID_ADD, GRID_ANY, decide, label, predict, train, tune, two_way
from .progress import step
from .run_blocking import run_test as block_test
from .run_blocking import run_train as block_train

MODEL = C.ROOT / "models" / "lgb_v1.txt"
DECISION = C.EXPERIMENTS / "decision_params.json"


def _split():
    return pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")


def _world_ids(w: int) -> pl.DataFrame:
    sp = _split()
    return (sp.filter(~pl.col("is_ghost")) if w == 20 else sp).select("entity_id")


def train_features():
    (C.CACHE / "feats").mkdir(parents=True, exist_ok=True)
    for w in (20, 0):
        cand = pl.read_parquet(C.CACHE / "cand" / f"train_w{w}.parquet")
        f = build_features("train", cand, _world_ids(w), name=f"features train w{w}")
        f.write_parquet(C.CACHE / "feats" / f"train_w{w}.parquet")


def train_fit():
    truth = load_train_pairs()
    f = label(pl.read_parquet(C.CACHE / "feats" / "train_w20.parquet"), truth)
    sp = _split()
    booster = train(f, sp.filter(pl.col("is_train")).select("entity_id"), name="fit lgb_v1")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL), num_iteration=booster.best_iteration)
    imp = sorted(zip(booster.feature_name(), booster.feature_importance("gain")), key=lambda x: -x[1])
    (C.EXPERIMENTS / "b4_importance.json").write_text(json.dumps([(n, round(g, 1)) for n, g in imp], indent=1))


def train_eval():
    truth = load_train_pairs().select("s1", "m")
    sp = _split()
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(
        pl.col("entity_id").alias("s1"), "country", "holdout_half")
    booster = lgb.Booster(model_file=str(MODEL))
    report, scored_by_w = {}, {}
    (C.CACHE / "pred").mkdir(parents=True, exist_ok=True)
    for w in (20, 0):
        f = pl.read_parquet(C.CACHE / "feats" / f"train_w{w}.parquet").join(hold.select("s1"), on="s1")
        scored = f.select("s1", "m").with_columns(pl.Series("p", predict(booster, f)))
        scored.write_parquet(C.CACHE / "pred" / f"train_holdout_w{w}.parquet")
        scored_by_w[w] = scored
        report[f"world{w}"] = two_way(scored, truth, hold)
        step("eval", f"world{w}: " + json.dumps({k: v["report"]["macro_f05"] for k, v in report[f"world{w}"].items()}))
    # final thresholds: best average over both worlds on the full holdout
    best = None
    for ta in GRID_ANY:
        for tb in GRID_ADD:
            fs = [summarize(per_s1_f05(decide(scored_by_w[w], ta, tb), truth, hold))["macro_f05"] for w in (20, 0)]
            m = sum(fs) / 2
            if best is None or m > best[0]:
                best = (m, ta, tb, fs)
    params = {"t_any": best[1], "t_add": best[2], "holdout_f05_w20": best[3][0], "holdout_f05_w0": best[3][1]}
    DECISION.write_text(json.dumps(params, indent=2))
    report["final_params_full_holdout"] = params
    for w in (20, 0):
        report[f"world{w}_full_holdout_at_final_params"] = summarize(
            per_s1_f05(decide(scored_by_w[w], params["t_any"], params["t_add"]), truth, hold))
    (C.EXPERIMENTS / "b5_holdout_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def test_score(parts: int = 4):
    """Features + prediction in S1-partitioned parts (bounded memory; per-S1 context
    features stay exact because every S1's candidates are in the same part)."""
    booster = lgb.Booster(model_file=str(MODEL))
    cand = pl.read_parquet(C.CACHE / "cand" / "test.parquet")
    world = load_source("test", 1).select("entity_id")
    cand = cand.with_columns((pl.col("s1").hash(seed=0) % parts).alias("_part"))
    out = []
    for k in range(parts):
        ck = cand.filter(pl.col("_part") == k).drop("_part")
        f = build_features("test", ck, world, name=f"features+score test part {k + 1}/{parts}")
        out.append(f.select("s1", "m").with_columns(pl.Series("p", predict(booster, f))))
        del f
    scored = pl.concat(out)
    scored.write_parquet(C.CACHE / "pred" / "test.parquet")
    step("score", f"test: scored {scored.height:,} pairs")


def test_write():
    params = json.loads(DECISION.read_text())
    scored = pl.read_parquet(C.CACHE / "pred" / "test.parquet")
    cand = pl.read_parquet(C.CACHE / "cand" / "test.parquet").select("s1", "m")
    s1 = load_source("test", 1).select(pl.col("entity_id").alias("s1"))
    pred = decide(scored, params["t_any"], params["t_add"])
    C.OUTPUT.mkdir(parents=True, exist_ok=True)

    def write(pairs, col, path):
        g = pairs.group_by("s1").agg(pl.col("m").unique().sort().str.join(",").alias(col))
        out = s1.join(g, on="s1", how="left").with_columns(pl.col(col).fill_null("")).rename({"s1": "source1_entity_id"})
        out.write_csv(path, separator="\t", quote_style="never")
        return out

    write(pred, "matched_entity_ids", C.OUTPUT / "matching_results.tsv")
    write(cand, "candidate_entity_ids", C.OUTPUT / "candidate_pairs.tsv")
    n_pred = pred.group_by("s1").len()
    step("write", f"matching_results: {pred.height:,} matches over {n_pred.height:,} S1 "
                  f"({s1.height - n_pred.height:,} empty of {s1.height:,}); params {params}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=["train", "test"])
    ap.add_argument("--steps", required=True)
    a = ap.parse_args()
    steps = a.steps.split(",")
    t0 = time.time()
    if a.split == "train":
        fns = {"block": lambda: block_train([20, 0]), "features": train_features, "fit": train_fit, "eval": train_eval}
    else:
        fns = {"block": block_test, "score": test_score, "write": test_write}
    for s in steps:
        step("pipeline", f"{a.split}: step {s} ...")
        fns[s]()
    step("pipeline", f"{a.split}: steps {steps} DONE in {time.time() - t0:.0f}s")
