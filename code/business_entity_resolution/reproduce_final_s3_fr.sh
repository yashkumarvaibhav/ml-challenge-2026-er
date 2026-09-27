#!/bin/bash
# sub-2026-09-27-8: #27-7 (reproduce_final_s3.sh) with France-only stage-3 add threshold 0.7 and the France LLM veto.
set -e
bash code/business_entity_resolution/reproduce_final_s3.sh
cp output/matching_results.tsv cache/v6/final_2707_matching.tsv
export BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2 BER_LLM_TAG=qwen7b_r3 BER_LLM_BASE=Qwen2.5-7B
BER_TEST_PRED=cache/v6/s2/test_pred_e5base2_llm3_qwen7b_r3_all.parquet BER_DECISION=experiments/v6/decision_params_s3_fr070.json \
  python -m ber.pipeline test --steps write
python -m ber.experiments.france_category_swap --write output
python -m ber.experiments.france_street_diff --write output --only street
python -m ber.experiments.france_category_conflict --write output
python -m ber.final_s3 --others-from cache/v6/final_2707_matching.tsv      # US/India rows = #27-7 (0.5/0.75)
python -m ber.experiments.france_llm_veto --write output
sha256sum output/matching_results.tsv output/candidate_pairs.tsv   # 462b003e... / 582ac117...
