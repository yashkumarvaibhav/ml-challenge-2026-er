#!/bin/bash
# Stage-2 variant "b" on run-3 stage-1 scores: 600k training S1s, learning rate 0.05 -> models3/s2b.txt, p23b_*.parquet
cd "$(dirname "$0")"
export PYTHONPATH=src BER_ROOT=$PWD POLARS_MAX_THREADS=8 ER_BLOCK=2 ER_RUN=3 ER_FEATS=feats2 ER_ADDON=1 ER_LR=0.05 ER_S2_S1=600000 ER_S2TAG=b
PY=${PY:-../.venv/bin/python}
L=logs/s2b.log
run() { echo "=== $(date +%T) $*" >> $L; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; exit 1; }; }
[ -f models3/s2b.txt ] || run $PY -m er.pipe --split train --steps stage2
[ -f cache/p23b_test.parquet ] || run $PY -m er.pipe --split test --steps stage2
echo "S2B_DONE $(date +%T)" >> $L
