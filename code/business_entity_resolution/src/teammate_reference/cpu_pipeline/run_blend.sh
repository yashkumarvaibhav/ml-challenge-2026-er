#!/bin/bash
# Blend run 2 + run 3 -> output/blend23, then France post-processing -> output/blend23_fr
set -e
cd "$(dirname "$0")"
export PYTHONPATH=src BER_ROOT=$PWD POLARS_MAX_THREADS=4
PY=${PY:-../.venv/bin/python}
$PY -m er.blend --tag blend23
$PY -m er.blend --france-scores
$PY -m er.france_post --base output/blend23/matching_results.tsv --out output/blend23_fr --scores cache/p2b_test.parquet --rescue
echo BLEND_DONE
