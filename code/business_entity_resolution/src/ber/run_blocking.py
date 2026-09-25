"""Run candidate generation on train (validation worlds) or test, with blocking reports.

train:
  * world20 (ghost S1s removed from the reference population): queries = non-ghost
    holdout S1s + a training-portion sample (model training data).
  * world0 (all S1s present): queries = the same non-ghost holdout S1s.
  Reports (holdout only) -> experiments/b2_blocking_<world>.json
test:
  * all test S1s against the full test pool -> cache/cand/test.parquet
"""
import argparse
import json
import time

import polars as pl

from . import config as C
from .blocking import block
from .io import load_train_pairs
from .metrics import blocking_report
from .progress import step

TRAIN_SAMPLE = 400_000


def mflags(split: str) -> pl.DataFrame:
    return pl.concat([
        pl.read_parquet(C.CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=["entity_id", "n_native", "a_null"])
        for s in (2, 3)
    ]).rename({"entity_id": "m"})


def run_train(worlds):
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    hold = sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost"))
    trn = sp.filter(pl.col("is_train")).sample(TRAIN_SAMPLE, seed=C.SEED)
    truth = load_train_pairs().select("s1", "m")
    flags = mflags("train")
    (C.CACHE / "cand").mkdir(parents=True, exist_ok=True)
    for w in worlds:
        t0 = time.time()
        if w == 20:
            world = sp.filter(~pl.col("is_ghost")).select("entity_id")
            queries = pl.concat([hold.select("entity_id"), trn.select("entity_id")])
        else:
            world = sp.select("entity_id")
            queries = hold.select("entity_id")
        step("run_blocking", f"train world{w}: {queries.height:,} queries, world {world.height:,} S1")
        cand = block("train", queries, world)
        cand.write_parquet(C.CACHE / "cand" / f"train_w{w}.parquet")
        s1 = hold.select(pl.col("entity_id").alias("s1"), "country")
        rep = blocking_report(cand, truth, s1, flags)
        rep["runtime_sec"] = round(time.time() - t0)
        rep["world"] = f"{w}% ghosts"
        (C.EXPERIMENTS / f"b2_blocking_w{w}.json").write_text(json.dumps(rep, indent=2))
        step("run_blocking", f"world{w} holdout report: " + json.dumps({k: rep[k] for k in ("pair_recall", "oracle", "pairs_per_s1")}))
        print(json.dumps(rep, indent=2), flush=True)


def run_train_full(worlds):
    """Stage-2 universe: candidates for ALL S1s of each world (every rival S1 present, as on test)."""
    sp = pl.read_parquet(C.CACHE / "splits" / "train_s1_split.parquet")
    (C.CACHE / "cand").mkdir(parents=True, exist_ok=True)
    for w in worlds:
        world = (sp.filter(~pl.col("is_ghost")) if w == 20 else sp).select("entity_id")
        step("run_blocking", f"train FULL world{w}: {world.height:,} queries = whole world")
        cand = block("train", world, world)
        cand.write_parquet(C.CACHE / "cand" / f"train_full_w{w}.parquet")
        step("run_blocking", f"train FULL world{w}: {cand.height:,} pairs written")


def run_test():
    from .io import load_source
    s1 = load_source("test", 1).select("entity_id")
    cand = block("test", s1, s1)
    (C.CACHE / "cand").mkdir(parents=True, exist_ok=True)
    cand.write_parquet(C.CACHE / "cand" / "test.parquet")
    step("run_blocking", f"test: {cand.height:,} pairs, {cand['s1'].n_unique():,} S1 with >=1 candidate of {s1.height:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--worlds", default="20,0")
    ap.add_argument("--full", action="store_true", help="queries = all S1s of the world (stage-2 universe)")
    a = ap.parse_args()
    if a.split == "train" and a.full:
        run_train_full([int(x) for x in a.worlds.split(",")])
    elif a.split == "train":
        run_train([int(x) for x in a.worlds.split(",")])
    else:
        run_test()
