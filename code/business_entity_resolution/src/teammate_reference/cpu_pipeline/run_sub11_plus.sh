#!/bin/bash
# Rebuilds the two files the team used in sub-2026-09-27-11 .. -19:
#   output/sub7_fr/     France house-number veto list (france_numveto_dropped.parquet, 9,547 pairs) + filler rescue on sub-7
#   output/sub11_plus/  sub-11 + CPU-matcher blend changes (US/India) + the extra house-number conflicts (France)
# Inputs (set the env vars if the folders live elsewhere):
#   ER_TEAM_PREDS  folder with the team's test_pred_e5base2.parquet (and holdout_pred_w20_e5base2.parquet)
#   ER_SUB7        folder with sub-2026-09-27-7 matching_results.tsv + candidate_pairs.tsv
#   ER_SUB11       folder with sub-2026-09-27-11 matching_results.tsv + candidate_pairs.tsv
#   cache/p22_test.parquet (run_all2.sh), cache/p23b_test.parquet (run_all3.sh + run_s2b.sh)
set -e
cd "$(dirname "$0")"
export BER_ROOT="$PWD" PYTHONPATH=src POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-6}
PY=${PY:-../.venv/bin/python}
TEAM=${ER_TEAM_PREDS:-../e5base2_preds}
SUB7=${ER_SUB7:-../best_sub7_0.988599_tsvs}
OUT7=${OUT7:-output/sub7_fr}
OUT11=${OUT11:-output/sub11_plus}
# 1. France rows of the team's e5base2 test scores (used to protect typo-like, >=0.999 pairs and to pick rescue targets)
$PY - "$TEAM/test_pred_e5base2.parquet" <<'PYEOF'
import sys, polars as pl
fr = pl.read_parquet("cache/test_source1.parquet", columns=["entity_id", "country"]).filter(pl.col("country") == "France") \
    .select(pl.col("entity_id").alias("s1"))
pl.scan_parquet(sys.argv[1]).join(fr.lazy(), on="s1", how="semi").collect().write_parquet("cache/team_fr_test.parquet")
PYEOF
# 2. House-number veto + filler rescue on sub-7 -> france_numveto_dropped.parquet (the 9,547-pair veto list)
$PY -m er.france_post --base "$SUB7/matching_results.tsv" --cand-file "$SUB7/candidate_pairs.tsv" \
    --scores cache/team_fr_test.parquet --rescue --out "$OUT7"
# 3. sub11_plus: blend 0.7*logit(e5base2) + 0.3*logit(CPU matcher), t 0.6/0.75, applied as changes vs the team's 0.4/0.7 cut
#    on sub-11's US/India rows; France = sub-11 minus the veto-list pairs it still had (reads $OUT7/france_numveto_dropped.parquet)
$PY -m er.sub11plus --out "$OUT11" --veto "$OUT7/france_numveto_dropped.parquet"
sha256sum "$OUT7/matching_results.tsv" "$OUT11/matching_results.tsv"
