#!/bin/bash
# Reproduce the scored #27-11 output (SHA-256 186f4d1f...) from supplied records.
# Run at the extracted package root after the README's model/data preparation.
set -euo pipefail
export BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2

# France stage 3 and its category/street/conflict rules: scored #27-7.
bash code/business_entity_resolution/reproduce_final_s3.sh
cp output/matching_results.tsv cache/v6/final_2707_matching.tsv

# The two-Qwen stage 3 v4 US/India output: #27-9's US/India rows.
bash code/business_entity_resolution/reproduce_final_v4.sh

fr_artifacts=code/business_entity_resolution/artifacts
drop_list=${BER_FR_DROPS:-$fr_artifacts/france_numveto_2711_drops.parquet}
add_list=${BER_FR_ADDS:-$fr_artifacts/france_filler_2711_adds.parquet}
python -m ber.experiments.france_numveto \
  --france-from cache/v6/final_2707_matching.tsv \
  --us-base output/matching_results.tsv \
  --drops "$drop_list" --adds "$add_list" \
  --out output/matching_results.tsv

printf '%s  %s\n' '186f4d1f85deba7ab6b5bbffdc2e535cd2cf26318cc2dabfda6e4a2d9d95e8de' 'output/matching_results.tsv' | sha256sum -c -
printf '%s  %s\n' '582ac1171c81b021a18e99219b818596c634055770074ad285d6e4d2a938f9c8' 'output/candidate_pairs.tsv' | sha256sum -c -
