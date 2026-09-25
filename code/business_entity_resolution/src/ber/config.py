"""Paths, seeds and shared constants.

All paths are resolved relative to the repository root so the pipeline runs from a
fresh clone with `student_resource/` placed next to `code/`.
"""
from pathlib import Path
import os

ROOT = Path(os.environ.get("BER_ROOT", Path(__file__).resolve().parents[4]))
DATA = ROOT / "student_resource" / "dataset"
CACHE = ROOT / "cache"
EXPERIMENTS = ROOT / "experiments"
OUTPUT = ROOT / "output"

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)

# Entity-level holdout and ghost-world validation design.
SEED = 20260925
HOLDOUT_SIZE = 200_000
GHOST_FRAC = 0.20
MATCH_BUCKETS = (0, 1, 2, 3, 4, 5)  # 6+ is the last bucket

# Run name: model file models/lgb_<RUN>.txt (v1 = first submission, v2 = blocking v2).
RUN = os.environ.get("BER_RUN", "v2")

# Default CPU budget; override BER_PROCS for the local machine.
N_PROCS = int(os.environ.get("BER_PROCS", "96"))


def src_path(split: str, source: int) -> Path:
    return DATA / split / f"{split}_source{source}.tsv"


def parquet_path(split: str, source: int) -> Path:
    return CACHE / f"{split}_source{source}.parquet"
