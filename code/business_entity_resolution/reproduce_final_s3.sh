#!/bin/bash
# Final submission sub-2026-09-27-7 (public 0.988599):
#   stage 3 (Qwen2.5-7B LoRA r3 + LightGBM fit on holdout half A, world 0) on the stage-2 band 0.05-0.95 for every
#   country, decision 0.5/0.75, then the France rules: category swap, different street, broad category conflict.
#   Public path: 0.98806 (#27-2, stage 2 + two France rules) -> 0.98827 (#27-5, + US/India stage 3)
#                -> 0.988451 (#27-6, + France category conflict) -> 0.988599 (#27-7, + France stage 3).
# Prerequisites: README steps 0-6 (cache/v6/s2/test_pred_e5base2.parquet) and the Qwen LoRA rounds (README "Stage 3").
set -e
export BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2 BER_LLM_TAG=qwen7b_r3 BER_LLM_BASE=Qwen2.5-7B
BER_S3_COUNTRIES=US,India python -m ber.llm_stage3 --steps test
BER_S3_COUNTRIES=France python -m ber.llm_stage3 --steps test
python - <<'PY'
import polars as pl
S = "cache/v6/s2/"
a = pl.read_parquet(S + "test_pred_e5base2_llm3_qwen7b_r3_India_US.parquet")
f = pl.read_parquet(S + "test_pred_e5base2_llm3_qwen7b_r3_France.parquet")
fr = pl.read_parquet("cache/test_source1.parquet").filter(pl.col("country") == "France").select(pl.col("entity_id").alias("s1"))
m = pl.concat([a.join(fr, on="s1", how="anti"), f.join(fr, on="s1")])
assert m.height == a.height
m.write_parquet(S + "test_pred_e5base2_llm3_qwen7b_r3_all.parquet")
PY
BER_TEST_PRED=cache/v6/s2/test_pred_e5base2_llm3_qwen7b_r3_all.parquet BER_DECISION=experiments/v6/decision_params_s3_qwen7b_r3.json \
  python -m ber.pipeline test --steps write
BER_TEST_PRED=cache/v6/s2/test_pred_e5base2.parquet python -m ber.experiments.france_category_swap   # category words (mined from stage-2 decisions)
python -m ber.experiments.france_category_swap --write output
python -m ber.experiments.france_street_diff --write output --only street
python -m ber.experiments.france_category_conflict --write output
sha256sum output/matching_results.tsv output/candidate_pairs.tsv   # cbacd9f4... / 582ac117...
