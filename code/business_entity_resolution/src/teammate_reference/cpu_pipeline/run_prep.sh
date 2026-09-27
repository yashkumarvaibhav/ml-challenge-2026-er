#!/bin/bash
set -e
# Splits, raw normalization, learned maps, final normalization (train + test) -> cache/
cd "$(dirname "$0")"
export BER_ROOT="$PWD"
PY=${PY:-../.venv/bin/python}   # override with PY=/path/to/python
export PYTHONPATH=src BER_PROCS=8
$PY -m ber.splits
$PY -m ber.normalize --stage raw --split train
$PY -m ber.learn_maps
$PY -m ber.normalize --stage final --split train
$PY -m ber.normalize --stage final --split test
echo PREP_DONE
