#!/bin/bash
# Apply the France post-processing (house-number veto + filler rescue) to the team's best submission.
#   ./apply_team_sub.sh <dir with the team's matching_results.tsv and candidate_pairs.tsv> [out dir]
# US/India rows stay byte-identical; only France rows change. Output is validated with the official validator.
set -e
cd "$(dirname "$0")"
SUB=${1:?usage: $0 <team submission dir> [out dir]}
OUT=${2:-output/team_sub_fr}
export PYTHONPATH=src BER_ROOT=$PWD POLARS_MAX_THREADS=4
PY=${PY:-../.venv/bin/python}
[ -f cache/p2b_test.parquet ] || $PY -m er.blend --france-scores
CAND="$SUB/candidate_pairs.tsv"
if [ ! -f "$CAND" ]; then   # no candidate file shipped: our candidates + the submission's own matches
    CAND=output/team_cand/candidate_pairs.tsv
    $PY -m er.candunion --matching "$SUB/matching_results.tsv" --out "$CAND"
fi
$PY -m er.france_post --base "$SUB/matching_results.tsv" --cand-file "$CAND" \
    --scores cache/p2b_test.parquet --rescue --out "$OUT"
$PY - "$SUB/matching_results.tsv" "$OUT/matching_results.tsv" <<'PYEOF'
import sys, polars as pl
a, b = (pl.read_csv(p, separator="\t", quote_char=None, infer_schema_length=0) for p in sys.argv[1:3])
c = pl.read_parquet("cache/test_source1.parquet", columns=["entity_id", "country"]).rename({"entity_id": a.columns[0]})
j = a.join(b, on=a.columns[0], suffix="_new").join(c, on=a.columns[0])
norm = lambda col: pl.col(col).fill_null("").str.split(",").list.sort().list.join(",")
j = j.with_columns((norm(a.columns[1]) != norm(a.columns[1] + "_new")).alias("changed"))
print(j.group_by("country").agg(pl.len().alias("s1"), pl.col("changed").sum().alias("s1_changed")).sort("country"))
PYEOF
echo "done: $OUT"
