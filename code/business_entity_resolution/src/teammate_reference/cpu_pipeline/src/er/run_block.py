"""Run candidate generation for a split / ghost world and report blocking quality on the holdout.

  python -m er.run_block --split train --world 20
  python -m er.run_block --split test
"""
import argparse
import json
import time

import os

import polars as pl

if os.environ.get("ER_BLOCK") == "2":
    from er.block2 import block_country
    TAG = "b2_"
else:
    from er.block import block_country
    TAG = ""
from er.common import CACHE, EXP, hold_s1, log, norm, truth_pairs, world_s1
from ber.metrics import per_s1_f05, summarize


def run(split: str, world: int):
    t0 = time.time()
    cols = ["entity_id", "country", "n_tok", "a_tok"]
    s1 = norm(split, 1, cols)
    if split == "train":
        s1 = s1.join(world_s1(world), on="entity_id")
    rec = pl.concat([norm(split, 2, cols), norm(split, 3, cols)])
    out = []
    for cty in sorted(s1["country"].unique().to_list()):
        a = s1.filter(pl.col("country") == cty).select("entity_id", "n_tok", "a_tok")
        b = rec.filter(pl.col("country") == cty).select("entity_id", "n_tok", "a_tok")
        log(f"block {split} w{world} {cty}: {a.height:,} S1 x {b.height:,} records")
        cp = CACHE / "cand" / f"{TAG}{split}_w{world}_{cty}.parquet"
        cp.parent.mkdir(parents=True, exist_ok=True)
        if cp.exists():
            c = pl.read_parquet(cp)
        else:
            c = block_country(a, b, log=log)
            c.write_parquet(cp)
        log(f"block {split} w{world} {cty}: {c.height:,} pairs ({c.height / a.height:.1f}/S1) {time.time() - t0:.0f}s")
        out.append(c)
    cand = pl.concat(out)
    path = CACHE / "cand" / f"{TAG}{split}_w{world}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    cand.write_parquet(path)
    if split == "train":
        report(cand, world)


def report(cand: pl.DataFrame, world: int) -> dict:
    tr = truth_pairs()
    h = hold_s1(world)
    t = tr.join(h.select("s1"), on="s1")
    hit = t.join(cand.select("s1", "m"), on=["s1", "m"], how="semi")
    rep = {"pairs_per_s1": round(cand.height / cand["s1"].n_unique(), 2),
           "pair_recall": round(hit.height / t.height, 5),
           "oracle": summarize(per_s1_f05(hit, t, h))}
    log(f"blocking w{world}: {json.dumps(rep)}")
    (EXP / f"{TAG}block_w{world}.json").write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--world", type=int, default=20)
    a = ap.parse_args()
    run(a.split, a.world)
