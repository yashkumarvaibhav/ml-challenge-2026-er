#!/bin/bash
# Run 2: blocking v2 (z keys), separate artifacts (feats2/, models2/, p1_2/p2_2), LightGBM lr 0.15.
cd "$(dirname "$0")"
export BER_ROOT="$PWD"
PY=${PY:-../.venv/bin/python}   # override with PY=/path/to/python
export PYTHONPATH=src POLARS_MAX_THREADS=8 ER_BLOCK=2 ER_RUN=2 ER_LR=0.15
L=logs/all2.log
run() { echo "=== $(date +%T) $*" >> $L; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; exit 1; }; }
[ -f cache/cand/b2_train_w20.parquet ] || run $PY -m er.run_block --split train --world 20
[ -f cache/cand/b2_test_w20.parquet ] || run $PY -m er.run_block --split test
[ -f cache/feats2/train/part15.parquet ] || run $PY -m er.pipe --split train --steps feats
[ -f cache/p12_train.parquet ] || run $PY -m er.pipe --split train --steps stage1
[ -f models2/s2.txt ] || run $PY -m er.pipe --split train --steps stage2
[ -f cache/feats2/test/part15.parquet ] || run $PY -m er.pipe --split test --steps feats
[ -f cache/p12_test.parquet ] || run $PY -m er.pipe --split test --steps stage1
[ -f cache/p22_test.parquet ] || run $PY -m er.pipe --split test --steps stage2
run $PY -m er.pipe --split test --steps write
echo "ALL_DONE $(date +%T)" >> $L
