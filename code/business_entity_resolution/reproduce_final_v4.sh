#!/bin/bash
# Stage 3 v4 final candidates:
#   F1v4 = sub-2026-09-27-10: stage 3 v4 in every country, then France category-swap, street, category-conflict rules + France LLM veto
#   F2v4 = sub-2026-09-27-9 : stage 3 v4 for US/India; France rows = sub-2026-09-27-8 (reproduce_final_s3_fr.sh)
# Stage 3 v4 = LightGBM on the stage-2 band 0.01 <= p <= 0.995 with: stage-2 p, S1/record context, the stage-2 context
# features (stage-1 p, bi-encoder cosine, CE, ranks, competition counts), the BER_S2X stage-2 challenger score and two
# LoRA LLM pair classifiers (Qwen2.5-7B r3, Qwen2.5-1.5B r3; scored on 0.05-0.95 and 0.95-0.995); fit on the world-0
# band of holdout half A; decision 0.5 / 0.75. Holdout half B vs production: +0.00052/+0.00044 (A->B w0/test-like),
# +0.00066/+0.00051 (B->A).
# Prerequisites: README steps 0-6, README "Stage 3" (7B rounds) and "Stage 3 v4" (1.5B rounds, S2X challenger).
set -e
export BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2
bash code/business_entity_resolution/reproduce_final_s3_fr.sh                       # -> #27-8 in output/
cp output/matching_results.tsv cache/v6/final_2708_matching.tsv
# LLM band scores (7B band scores were produced by reproduce_final_s3.sh through ber.llm_stage3 --steps test)
BER_LLM_TAG=qwen15_r3 BER_LLM_BASE=Qwen2.5-1.5B python -m ber.llm_stage3 --steps holdout
BER_LLM_TAG=qwen15_r3 BER_LLM_BASE=Qwen2.5-1.5B BER_S3_COUNTRIES=France,India,US python -m ber.llm_stage3 --steps test
BER_LLM_TAG=qwen7b_r3 BER_LLM_BASE=Qwen2.5-7B   python -m ber.llm_band_scores
BER_LLM_TAG=qwen15_r3 BER_LLM_BASE=Qwen2.5-1.5B python -m ber.llm_band_scores
BER_LLM_TAG=qwen7b_r3 BER_LLM_BASE=Qwen2.5-7B python -m ber.llm_stage3 --steps test_v4
# F1v4
BER_TEST_PRED=cache/v6/s2/test_pred_e5base2_llm3v4_qwen7b_r3.parquet BER_DECISION=experiments/v6/decision_params_s3_qwen7b_r3.json \
  python -m ber.pipeline test --steps write
python -m ber.experiments.france_category_swap --write output
python -m ber.experiments.france_street_diff --write output --only street
python -m ber.experiments.france_category_conflict --write output
BER_LLM_TAG=qwen7b_r3 BER_LLM_BASE=Qwen2.5-7B python -m ber.experiments.france_llm_veto --write output
cp output/matching_results.tsv cache/v6/final_F1v4_matching.tsv
sha256sum output/matching_results.tsv          # aed2b972... (sub-2026-09-27-10)
# F2v4
python -m ber.final_s3 --france-from cache/v6/final_2708_matching.tsv
sha256sum output/matching_results.tsv output/candidate_pairs.tsv   # 04ed2ac2... (sub-2026-09-27-9) / 582ac117...
