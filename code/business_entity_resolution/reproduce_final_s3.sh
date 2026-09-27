#!/bin/bash
# Stage-3 variant (sub-2026-09-27-5): US/India decisions from the Qwen2.5-7B stage 3, France rows = sub-2026-09-27-2.
#   1. reproduce_final.sh -> output/ (= sub-2026-09-27-2); keep a copy for its France rows
#   2. stage 3 on the US/India stage-2 band 0.05-0.95 (ber.llm_stage3 --steps test; LightGBM fit on holdout half A, world 0)
#   3. pipeline write at 0.5/0.75 from the stage-3 scores, then France rows restored from step 1
# Prerequisites: README steps 0-6 and the three Qwen LoRA rounds (README "Stage 3").
set -e
bash code/business_entity_resolution/reproduce_final.sh
cp output/matching_results.tsv cache/v6/final_2702_matching.tsv
export BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2 BER_LLM_TAG=qwen7b_r3 BER_LLM_BASE=Qwen2.5-7B BER_S3_COUNTRIES=US,India
python -m ber.llm_stage3 --steps test
BER_TEST_PRED=cache/v6/s2/test_pred_e5base2_llm3_qwen7b_r3_India_US.parquet BER_DECISION=experiments/v6/decision_params_s3_qwen7b_r3.json \
  python -m ber.pipeline test --steps write
python -m ber.final_s3 --france-from cache/v6/final_2702_matching.tsv
sha256sum output/matching_results.tsv output/candidate_pairs.tsv
