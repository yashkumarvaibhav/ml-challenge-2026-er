#!/bin/bash
# Regenerate run-2 full-test scores (stage 1 -> stage 2); stage 2 waits for the s2b test job (memory)
cd "$(dirname "$0")"
export PYTHONPATH=src BER_ROOT=$PWD POLARS_MAX_THREADS=6 ER_BLOCK=2 ER_FEATS=feats2
PY=${PY:-../.venv/bin/python}
L=logs/regen.log
run() { echo "=== $(date +%T) $*" >> $L; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; exit 1; }; }
[ -f cache/p12_test.parquet ] || ER_RUN=2 run $PY -m er.pipe --split test --steps stage1
[ -f cache/p22_test.parquet ] || ER_RUN=2 run $PY -m er.pipe --split test --steps stage2
[ -f cache/p23b_testfr.parquet ] || ER_RUN=3 ER_ADDON=1 ER_S2TAG=b run $PY -m er.pipe --split testfr --steps stage2
echo "REGEN_DONE $(date +%T)" >> $L
