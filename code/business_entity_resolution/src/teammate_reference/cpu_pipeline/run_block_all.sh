#!/bin/bash
# Blocking v1 (early run 1 only; the final pipeline uses blocking v2 from run_all2.sh)
cd "$(dirname "$0")"
export BER_ROOT="$PWD"
PY=${PY:-../.venv/bin/python}   # override with PY=/path/to/python
export PYTHONPATH=src POLARS_MAX_THREADS=8
$PY -m er.run_block --split train --world 20 > logs/block_train.log 2>&1
$PY -m er.run_block --split test > logs/block_test.log 2>&1
echo BLOCK_ALL_DONE >> logs/block_test.log
