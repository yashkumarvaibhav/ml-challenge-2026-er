# Reproduction and private inputs

This source release preserves the implementation from the final #27-19 submission package. It does not ship data, learned artifacts, outputs, or record-specific correction lists. The commands below are historical training instructions, not evidence of a new full training run.

## Data and environment

Run the main pipeline from the repository root. Use Python 3.10, the main requirements file, and a CUDA-capable NVIDIA GPU. Set `BER_PROCS` to a suitable number for your hardware. The archived pins are preserved rather than upgraded for this release; a new installation and full GPU run have not been validated.

Place your authorized private files at:

```text
student_resource/dataset/train/
  train_source1.tsv
  train_source2.tsv
  train_source3.tsv
  train_ground_truth.tsv
student_resource/dataset/test/
  test_source1.tsv
  test_source2.tsv
  test_source3.tsv
```

These locations and all generated files are ignored by Git. This repository grants no access to or redistribution rights for the dataset.

## Main pipeline training

The following stages generate local training artifacts. The intermediate stage-2 output is not the final #27-19 submission. Base checkpoints are downloaded once at pinned revisions; later processing runs offline. The original 96/64-thread examples below should be reduced on smaller machines.

```bash
python3.10 -m venv .venv        # Python 3.10
# torch 2.4.1+cu121 comes from the PyTorch CUDA 12.1 wheel index
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
source .venv/bin/activate                            # every `python` below is this environment
export PYTHONPATH=code/business_entity_resolution/src BER_PROCS=96

# 0. one-time download of the four pinned base models (MIT / Apache-2.0); every later step runs offline
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('intfloat/multilingual-e5-small', revision='614241f622f53c4eeff9890bdc4f31cfecc418b3', local_dir='models/hf/multilingual-e5-small')"
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('intfloat/multilingual-e5-base', revision='d128750597153bb5987e10b1c3493a34e5a4502a', local_dir='models/hf/multilingual-e5-base')"
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-7B', revision='d149729398750b98c0af14eb82c78cfe92750796', local_dir='models/hf/Qwen2.5-7B', allow_patterns=['*.json','*.safetensors','*.txt','LICENSE'])"
.venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen2.5-1.5B', revision='8faed761d45a263340a0528343f099c05c9a4323', local_dir='models/hf/Qwen2.5-1.5B', allow_patterns=['*.json','*.safetensors','*.txt','LICENSE'])"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 1. data -> splits -> normalization (training-portion pairs only for the learned maps)
.venv/bin/python -m ber.splits
.venv/bin/python -m ber.normalize --stage raw --split train
.venv/bin/python -m ber.learn_maps
.venv/bin/python -m ber.normalize --stage final --split train
.venv/bin/python -m ber.normalize --stage raw --split test
.venv/bin/python -m ber.normalize --stage final --split test

# 2. v4 prerequisite: rule-based blocking + v4 matcher (hard negatives for the encoders; France pseudo-pairs)
export BER_RUN=v4
.venv/bin/python -m ber.pipeline train --steps block,features,fit,eval
.venv/bin/python -m ber.run_blocking --split train --full --worlds 20
.venv/bin/python -m ber.pipeline test --steps block,score
.venv/bin/python -m ber.self_maps mine                              # France maps from v4 test pseudo-pairs (p >= 0.99)
.venv/bin/python -m ber.normalize --stage final --split test        # re-normalize test with the self-trained maps

# 3. bi-encoder + first cross-encoder (GPU; both use the v4 candidates of step 2)
BER_PROCS=64 .venv/bin/python -m ber.dense --steps data,tokenize,train,encode          # bi-encoder + embeddings
BER_CE=e5base .venv/bin/python -m ber.cross_encoder --steps data,train                # cross-encoder (evalband is an optional report)

# 4. v6 blocking + stage 1 (artifacts in cache/v6/, reports in experiments/v6/)
export BER_RUN=v6 BER_WORK=v6
.venv/bin/python -m ber.pipeline train --steps block,features,fit,eval
.venv/bin/python -m ber.run_blocking --split train --full --worlds 20            # cache/v6/cand/train_w20.parquet
.venv/bin/python -m ber.run_blocking --split test
.venv/bin/python -m ber.pipeline test --steps score

# 5. cross-encoder continued on the v6 hard negatives (needs cache/v6/cand/train_w20.parquet from step 4)
BER_CE=e5base2 .venv/bin/python -m ber.cross_encoder --steps data,train

# 6. stage 2 with the e5base2 cross-encoder
export BER_CE=e5base2
.venv/bin/python -m ber.stage2 --steps feats,oof,ce,s2feats,fit,eval --world 20   # writes experiments/v6/decision_params_s2_e5base2.json
.venv/bin/python -m ber.stage2 --steps cetest,test                                # cache/v6/s2/test_pred_e5base2.parquet
# 6b. world-0 blocking + stage-2 features and holdout scores (read by stage 3: s2feats_w0_e5base2, holdout_pred_w0_e5base2)
.venv/bin/python -m ber.run_blocking --split train --full --worlds 0
.venv/bin/python -m ber.stage2 --steps feats,oof,ce,s2feats,eval --world 0

# 7. outputs (stage 2 + France rules) -> output/matching_results.tsv (SHA-256 81cf71fe..., intermediate submission #27-2;
#    the final output is produced by reproduce_final_2719.sh, see "Final submission #27-19")
PATH=$PWD/.venv/bin:$PATH bash code/business_entity_resolution/reproduce_final.sh
```

## Qwen-7B pair classifier

The classifier is fine-tuned on the cross-encoder training pairs. Its combiner fits holdout half A and evaluates half B before the final A+B refit. These commands require the preceding caches.

```bash
# base model: downloaded in step 0
export BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2 BER_LLM_BASE=Qwen2.5-7B
# three LoRA rounds (A6000; ~28 pairs/s): 80k pairs, then +160k and +160k new pairs continuing from the previous adapter
BER_LLM_TAG=qwen7b    BER_LLM_N=80000  python -m ber.llm_ce --steps train
BER_LLM_TAG=qwen7b_r2 BER_LLM_N=160000 BER_LLM_INIT=qwen7b    BER_LLM_SKIP_N=80000 BER_LLM_SEED_OFF=8 python -m ber.llm_ce --steps train
BER_LLM_TAG=qwen7b_r3 BER_LLM_N=160000 BER_LLM_INIT=qwen7b_r2 BER_LLM_SKIP_N=80000 BER_LLM_SEED_OFF=9 python -m ber.llm_ce --steps train
BER_LLM_TAG=qwen7b_r3 python -m ber.llm_stage3 --steps holdout      # half-B report: experiments/v6/llm_stage3_qwen7b_r3_holdout.json
PATH=$PWD/.venv/bin:$PATH bash code/business_entity_resolution/reproduce_final_s3.sh   # -> output/matching_results.tsv (SHA-256 460623d7...)
```

## Qwen-1.5B and stage-3 v4

The second classifier and stage-2 challenger supply additional features to the combiner.

```bash
export BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2 BER_LLM_BASE=Qwen2.5-1.5B   # base model downloaded in step 0
# 1.5B LoRA rounds (A4000; ~45 pairs/s): 150k, then +300k and +300k new pairs continuing from the previous adapter
BER_LLM_TAG=qwen15    BER_LLM_N=150000 python -m ber.llm_ce --steps train
BER_LLM_TAG=qwen15_r2 BER_LLM_N=300000 BER_LLM_INIT=qwen15    BER_LLM_SKIP_N=150000 BER_LLM_SEED_OFF=8 python -m ber.llm_ce --steps train
BER_LLM_TAG=qwen15_r3 BER_LLM_N=300000 BER_LLM_INIT=qwen15_r2 BER_LLM_SKIP_N=150000 BER_LLM_SEED_OFF=9 python -m ber.llm_ce --steps train
# BER_S2X stage-2 challenger scores (holdout + test)
BER_S2X=1 python -m ber.stage2 --steps fit,eval --world 20 && BER_S2X=1 python -m ber.stage2 --steps eval --world 0
BER_S2X=1 python -m ber.stage2 --steps testfeats,test
PATH=$PWD/.venv/bin:$PATH bash code/business_entity_resolution/reproduce_final_v4.sh
```

## Exact final assembly requires a private artifact kit

`reproduce_final_2719.sh` supports the historical final modes:

- `BER_FINAL_MODE=exact` rebuilds #27-11 from the pipeline and applies the frozen net corrections.
- `BER_FINAL_MODE=replay` starts from the private frozen #27-11 output and applies those corrections. It needs no GPU for the assembly itself.
- `BER_FINAL_MODE=regenerate` runs the original assembly scripts with the prepared pipeline caches, teammate outputs, and audit inputs.

All modes need files excluded from this repository. The staging script checks for its required private files before copying anything. Without them, it exits with an explanation. The original encoder weights are optional staging inputs; model training is the alternative.

Place the privately retained kit under `code/business_entity_resolution/artifacts/`, following the relative paths referenced by the scripts. Stage it only before the intended run; staging during retraining can mix incompatible models and caches. It contains learned maps, LightGBM weights, LoRA adapters, decision files, France lists, and the final assembly lists. Replay also requires the frozen #27-11 matching TSV and the final candidate TSV. Place that candidate file at `output/candidate_pairs.tsv`.

```bash
export PYTHONPATH="$PWD/code/business_entity_resolution/src"
export BER_ROOT="$PWD"
bash code/business_entity_resolution/stage_artifacts.sh
# After placing the required private replay files:
BER_FINAL_MODE=replay bash code/business_entity_resolution/reproduce_final_2719.sh
```

The historical final output SHA-256 is `720dce1652fc585ed3569dcddd6013e5d016f8b59dd725e13ebab8319ee301e0`; the candidate-file SHA-256 is `582ac1171c81b021a18e99219b818596c634055770074ad285d6e4d2a938f9c8`. These identify privately held files; they do not distribute any record IDs. Exact modes compare against those hashes. A new training run may differ because of numerical and threshold effects.

The original organizer validator is also private and must be obtained through authorized access. No claim of current organizer approval or a newly evaluated leaderboard score is made by this source release.

## Independent CPU matcher

See [the CPU matcher README](../code/business_entity_resolution/src/teammate_reference/cpu_pipeline/README.md). It needs a separate environment, dataset placement, training caches, and the main pipeline inputs for the final blend. The public copy contains its code and training entry points, but not its trained `models2/` or `models3/` directories.
