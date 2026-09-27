#!/bin/bash
# FINAL SUBMISSION: sub-2026-09-27-19 (best public score 0.989275; the submission used for the private ranking).
#   matching_results.tsv SHA-256 720dce1652fc585ed3569dcddd6013e5d016f8b59dd725e13ebab8319ee301e0
#   candidate_pairs.tsv  SHA-256 582ac1171c81b021a18e99219b818596c634055770074ad285d6e4d2a938f9c8
# Run at the extracted package root after the README's model/data preparation (steps 0-6, "Stage 3", "Stage 3 v4").
#
#   1. reproduce_final_2711.sh -> scored #27-11 (public 0.989213, SHA-256 186f4d1f...).
#   2. Final assembly #27-11 -> #27-19 (src/ber/final2719/):
#        #27-12 France stage-3-v4 selective stack       #27-13 one owner per record + per-S1 source caps
#        #27-14 stage-3 v4 refit on holdout halves A+B   #27-15 France kept-back house-number drops
#        #27-16 teammate blend as US/India tie-breaker   #27-17 per-S1 expected-F0.5 restore
#        #27-19 independent-audit correction
#      BER_FINAL_MODE=exact (default): apply the frozen net pair lists (artifacts/final2719/final2719_{drops,adds}.parquet;
#        7,212 drops and 5,853 adds) and check the SHA-256.
#      BER_FINAL_MODE=replay: skip step 1 and start from the uploaded #27-11 file shipped in artifacts/final2719/ (no GPU).
#      BER_FINAL_MODE=regenerate: re-run the original step scripts s12a ... s19 (they need the pipeline caches of
#        steps 4-6 and "Stage 3 v4"); every intermediate file is compared with the SHA-256 of the uploaded version.
# BER_SKIP_2711=1 skips step 1 when output/matching_results.tsv already holds #27-11.
set -euo pipefail
export BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2
export BER_ROOT=${BER_ROOT:-$PWD}
export PYTHONPATH=${PYTHONPATH:-$BER_ROOT/code/business_entity_resolution/src}
F=code/business_entity_resolution/src/ber/final2719
A=code/business_entity_resolution/artifacts
# Fail before changing outputs if this source-only clone lacks private inputs.
case "${BER_FINAL_MODE:-exact}" in
  exact|replay)
    required=(final2719/final2719_drops.parquet final2719/final2719_adds.parquet)
    ;;
  regenerate)
    required=(teammate/france_numveto_dropped.parquet teammate/sub11_plus_matching_results.tsv
              final2719/codex_restore17.tsv final2719/codex_negative16.tsv)
    ;;
  *) printf 'BER_FINAL_MODE must be exact, replay, or regenerate.\n' >&2; exit 1 ;;
esac
if [ "${BER_FINAL_MODE:-exact}" = replay ]; then
  required+=(final2719/sub-2026-09-27-11_matching_results.tsv)
fi
missing=0
for file in "${required[@]}"; do
  if [ ! -f "$A/$file" ]; then
    printf 'Missing private input: %s\n' "$A/$file" >&2
    missing=1
  fi
done
if [ "${BER_FINAL_MODE:-exact}" = replay ] && [ ! -f output/candidate_pairs.tsv ]; then
  printf 'Replay also needs the private final candidate file at output/candidate_pairs.tsv.\n' >&2
  missing=1
fi
if [ "$missing" = 1 ]; then
  printf 'Exact final assembly needs private inputs excluded from the public source. See docs/reproduction.md.\n' >&2
  exit 1
fi
if [ "${BER_FINAL_MODE:-exact}" = replay ]; then
  printf '%s  %s\n' '582ac1171c81b021a18e99219b818596c634055770074ad285d6e4d2a938f9c8' 'output/candidate_pairs.tsv' | sha256sum -c -
fi

if [ "${BER_FINAL_MODE:-exact}" = "replay" ]; then   # final assembly only: start from the uploaded #27-11 file
  cp $A/final2719/sub-2026-09-27-11_matching_results.tsv output/matching_results.tsv
  BER_SKIP_2711=1
fi
if [ "${BER_SKIP_2711:-0}" != "1" ]; then
  bash code/business_entity_resolution/reproduce_final_2711.sh
fi
printf '%s  %s\n' '186f4d1f85deba7ab6b5bbffdc2e535cd2cf26318cc2dabfda6e4a2d9d95e8de' 'output/matching_results.tsv' | sha256sum -c -

if [ "${BER_FINAL_MODE:-exact}" != "regenerate" ]; then
  python -m ber.final2719.apply_2719 --base output/matching_results.tsv --out output/matching_results.tsv
else
  check() { printf '%s  %s\n' "$2" "$1" | sha256sum -c - || echo "WARNING: $1 differs from the uploaded file ($2)"; }
  stage() { mkdir -p "submissions/archive/$1" && cp "$2" "submissions/archive/$1/matching_results.tsv" && cp output/candidate_pairs.tsv "submissions/archive/$1/candidate_pairs.tsv"; }
  # inputs the step scripts read, at the paths they had on the build machine
  stage sub-2026-09-27-11 output/matching_results.tsv
  stage sub-2026-09-27-7 cache/v6/final_2707_matching.tsv          # written by reproduce_final_2711.sh
  stage sub-2026-09-27-10 cache/v6/final_F1v4_matching.tsv         # written by reproduce_final_v4.sh
  python -m ber.final_s3 --france-from cache/v6/final_2708_matching.tsv    # #27-9 = stage 3 v4 US/India + #27-8 France
  stage sub-2026-09-27-9 output/matching_results.tsv
  cp submissions/archive/sub-2026-09-27-11/matching_results.tsv output/matching_results.tsv
  mkdir -p pinaksh/sub7_fr pinaksh/sub11_plus/sub11_plus handoff
  cp $A/teammate/france_numveto_dropped.parquet pinaksh/sub7_fr/
  cp $A/teammate/sub11_plus_matching_results.tsv pinaksh/sub11_plus/sub11_plus/matching_results.tsv
  cp $A/final2719/codex_restore17.tsv $A/final2719/codex_negative16.tsv handoff/

  python $F/s12a_stack12_screen.py
  python $F/s12b_build.py
  cp submissions/archive/sub-2026-09-27-11/candidate_pairs.tsv submissions/archive/sub-2026-09-27-12/
  python $F/s12c_fix_number_conflicts.py
  python $F/s12d_fix_normalized_conflicts.py
  check submissions/archive/sub-2026-09-27-12/matching_results.tsv 406bcee6d734c41c82711cbd2ba2c947153674e342e6c6533cf7accaeccf8a96
  python $F/s13a_one_owner.py
  cp submissions/archive/sub-2026-09-27-12/candidate_pairs.tsv submissions/archive/sub-2026-09-27-13/
  python $F/s13b_source_caps.py
  check submissions/archive/sub-2026-09-27-13/matching_results.tsv e59f8e994ba996ab4128c57dbcafbf3b357e33dfcb87966444caca74c54c7098
  python $F/s14a_stage3v4_refit_ab.py
  python $F/s14b_s15_candidates_keptback.py
  cp submissions/archive/sub-2026-09-27-13/candidate_pairs.tsv submissions/archive/sub-2026-09-27-14/
  cp submissions/archive/sub-2026-09-27-13/candidate_pairs.tsv submissions/archive/sub-2026-09-27-15/
  check submissions/archive/sub-2026-09-27-14/matching_results.tsv 0057f29d7ed0c68b99a753b3d61d41e822427caa763da37a4b76e13ce76aab58
  check submissions/archive/sub-2026-09-27-15/matching_results.tsv 58852be0fc986f30e44ec54833024801ffc58edae031636aa85d4ba18787cd25
  python $F/s16_teammate_blend_tiebreak.py
  cp submissions/archive/sub-2026-09-27-15/candidate_pairs.tsv submissions/archive/sub-2026-09-27-16/
  check submissions/archive/sub-2026-09-27-16/matching_results.tsv 41de81e53cdec6ab5be03997dd624acf345be4fd720bf3ca2a0d0d14b602811a
  python $F/s17_expected_f_refine.py
  cp submissions/archive/sub-2026-09-27-16/candidate_pairs.tsv submissions/archive/sub-2026-09-27-17/
  check submissions/archive/sub-2026-09-27-17/matching_results.tsv 3e6a274e212349f132493e58f1efb1affe82a9338581f0c979ad35f0d5d0701f
  python $F/s19_audit_fix.py
  cp submissions/archive/sub-2026-09-27-18/matching_results.tsv output/matching_results.tsv
fi

printf '%s  %s\n' '720dce1652fc585ed3569dcddd6013e5d016f8b59dd725e13ebab8319ee301e0' 'output/matching_results.tsv' | sha256sum -c -
printf '%s  %s\n' '582ac1171c81b021a18e99219b818596c634055770074ad285d6e4d2a938f9c8' 'output/candidate_pairs.tsv' | sha256sum -c -
