"""Shared paths and loaders for the er pipeline (reuses ber's normalized records and splits)."""
import os
from datetime import datetime
from pathlib import Path

import polars as pl

# work/ directory: $BER_ROOT if set, else the folder two levels above this file (work/src/er/common.py)
ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[2]))
CACHE = ROOT / "cache"
EXP = ROOT / "experiments"
EXP.mkdir(parents=True, exist_ok=True)
LOG = ROOT / "logs" / "progress.log"


def log(msg: str):
    line = f"{datetime.now():%H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def norm(split: str, s: int, cols=None) -> pl.DataFrame:
    return pl.read_parquet(CACHE / "norm_final" / f"{split}_s{s}.parquet", columns=cols)


def split_table() -> pl.DataFrame:
    return pl.read_parquet(CACHE / "splits" / "train_s1_split.parquet")


def world_s1(world: int) -> pl.DataFrame:
    sp = split_table()
    return (sp.filter(~pl.col("is_ghost")) if world == 20 else sp).select("entity_id")


def hold_s1(world: int = 20) -> pl.DataFrame:
    """Evaluated holdout S1s: non-ghost holdout (valid in both worlds), with country and half."""
    sp = split_table()
    return sp.filter(pl.col("is_holdout") & ~pl.col("is_ghost")).select(
        pl.col("entity_id").alias("s1"), "country", "holdout_half")


def truth_pairs() -> pl.DataFrame:
    return pl.read_parquet(CACHE / "train_pairs.parquet").select("s1", "m")
