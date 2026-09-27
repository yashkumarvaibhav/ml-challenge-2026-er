#!/bin/bash
# France filler normalization run: testfr split -> features -> add-on -> stage 1/2 with run-2, run-3 (and s2b) models
cd "$(dirname "$0")"
export PYTHONPATH=src BER_ROOT=$PWD POLARS_MAX_THREADS=8 ER_BLOCK=2 ER_FEATS=feats2
PY=${PY:-../.venv/bin/python}
L=logs/frfill.log
run() { echo "=== $(date +%T) $*" >> $L; "$@" >> $L 2>&1 || { echo "FAILED: $*" >> $L; exit 1; }; }
[ -f cache/cand/b2_testfr_w20.parquet ] || run $PY -m er.frfill
[ $(ls cache/feats2/testfr/part*.parquet 2>/dev/null | wc -l) -ge 16 ] || ER_RUN=2 run $PY -m er.pipe --split testfr --steps feats
[ $(ls cache/addon/testfr/part*.parquet 2>/dev/null | wc -l) -ge 16 ] || POLARS_MAX_THREADS=2 run $PY -m er.addon --split testfr
[ -f cache/p22_testfr.parquet ] || ER_RUN=2 run $PY -m er.pipe --split testfr --steps stage1,stage2
[ -f cache/p23_testfr.parquet ] || ER_RUN=3 ER_ADDON=1 run $PY -m er.pipe --split testfr --steps stage1,stage2
[ -f models3/s2b.txt ] && [ ! -f cache/p23b_testfr.parquet ] && ER_RUN=3 ER_ADDON=1 ER_S2TAG=b run $PY -m er.pipe --split testfr --steps stage2
echo "FRFILL_DONE $(date +%T)" >> $L
