#!/bin/bash
# Put the shipped learned artifacts where the code reads them. Run from the package root (the folder that contains
# output/ and code/) once, before the README's reproduction steps. Retraining steps overwrite these files together with
# caches that match the retrained files, so do not re-stage in the middle of a retraining run.
set -euo pipefail
A=code/business_entity_resolution/artifacts
# Public source releases exclude this private artifact kit.
required=(
  maps.json self_maps_test.json lgb_v4.txt lgb_v6.txt lgb_s2_v6_e5base2.txt
  lgb_s3v4_llm_qwen7b_r3.txt lgb_s3v4ab_llm_qwen7b_r3.txt
  decision_params_s2_e5base2.json decision_params_s3_qwen7b_r3.json
  decision_params_s3_fr070.json france_category_swap.json
  llm_ce_qwen7b_r3/adapter_config.json llm_ce_qwen7b_r3/adapter_model.safetensors
  llm_ce_qwen15_r3/adapter_config.json llm_ce_qwen15_r3/adapter_model.safetensors
  france_numveto_2711_drops.parquet france_filler_2711_adds.parquet
)
missing=0
for file in "${required[@]}"; do
  if [ ! -f "$A/$file" ]; then
    printf 'Missing private artifact: %s\n' "$A/$file" >&2
    missing=1
  fi
done
if [ -f "$A/models__dense_e5small__model.pt" ] && [ ! -f "$A/models__ce_e5base2__model.pt" ]; then
  printf 'Missing private cross-encoder weights for the supplied encoder kit.\n' >&2
  missing=1
fi
if [ "$missing" = 1 ]; then
  printf 'Public source excludes trained artifacts. See docs/reproduction.md; nothing was staged.\n' >&2
  exit 1
fi
mkdir -p cache/maps cache/v6/s2 experiments/v6 models/llm_ce_qwen7b_r3 models/llm_ce_qwen15_r3
cp $A/maps.json $A/self_maps_test.json cache/maps/
cp $A/lgb_v4.txt $A/lgb_v6.txt $A/lgb_s2_v6_e5base2.txt $A/lgb_s3v4_llm_qwen7b_r3.txt $A/lgb_s3v4ab_llm_qwen7b_r3.txt models/
cp $A/decision_params_s2_e5base2.json $A/decision_params_s3_qwen7b_r3.json $A/decision_params_s3_fr070.json $A/france_category_swap.json experiments/v6/
cp $A/llm_ce_qwen7b_r3/adapter_config.json $A/llm_ce_qwen7b_r3/adapter_model.safetensors models/llm_ce_qwen7b_r3/
cp $A/llm_ce_qwen15_r3/adapter_config.json $A/llm_ce_qwen15_r3/adapter_model.safetensors models/llm_ce_qwen15_r3/
cp $A/france_numveto_2711_drops.parquet $A/france_filler_2711_adds.parquet cache/v6/s2/
if [ -f $A/models__dense_e5small__model.pt ]; then   # only if the fine-tuned encoder weights were added to artifacts/
  mkdir -p models/dense_e5small models/ce_e5base2
  cp $A/models__dense_e5small__model.pt models/dense_e5small/model.pt
  cp $A/models__ce_e5base2__model.pt models/ce_e5base2/model.pt
  if [ -d $A/ce_e5base2_tokenizer ]; then cp $A/ce_e5base2_tokenizer/* models/ce_e5base2/; fi   # ber.cross_encoder.score reads the tokenizer here
fi
if [ ! -d student_resource/dataset/train ] || [ ! -d student_resource/dataset/test ]; then
  echo "NOTE: put the supplied dataset at student_resource/dataset/train/ and student_resource/dataset/test/"
fi
echo "artifacts staged into cache/, experiments/v6/ and models/"
