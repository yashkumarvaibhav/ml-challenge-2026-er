#!/bin/bash
# Run 3: run-2 features + TF-IDF char-cosine add-on features; retrain stage 1/2 (models3/, p13/p23), write output/run3.
cd "$(dirname "$0")"
export BER_ROOT="$PWD"
PY=${PY:-../.venv/bin/python}   # override with PY=/path/to/python
[ -f cache/p12_test.parquet ] || { echo "run run_all2.sh first"; exit 1; }
export PYTHONPATH=src POLARS_MAX_THREADS=8 ER_BLOCK=2 ER_RUN=3 ER_FEATS=feats2 ER_ADDON=1 ER_LR=0.15
L=logs/all3.log
run() { echo "=== $(date +%T) $*" >> $L; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; exit 1; }; }
[ $(ls cache/addon/train/part*.parquet 2>/dev/null | wc -l) -ge 16 ] || POLARS_MAX_THREADS=2 run $PY -m er.addon --split train
[ $(ls cache/addon/test/part*.parquet 2>/dev/null | wc -l) -ge 16 ] || POLARS_MAX_THREADS=2 run $PY -m er.addon --split test
[ -f cache/p13_train.parquet ] || run $PY -m er.pipe --split train --steps stage1
[ -f models3/s2.txt ] || run $PY -m er.pipe --split train --steps stage2
[ -f cache/p13_test.parquet ] || run $PY -m er.pipe --split test --steps stage1
[ -f cache/p23_test.parquet ] || run $PY -m er.pipe --split test --steps stage2
run $PY -m er.pipe --split test --steps write
echo "ALL_DONE $(date +%T)" >> $L
