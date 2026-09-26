#!/bin/bash
# Final submission (public best, sub-2026-09-27-2) from the v6 + e5base2 test scores:
#   1. v6 stage 2 with the e5base2 cross-encoder -> output/ (= sub-2026-09-26-5, public 0.984185)
#   2. France category-swap rule (generator filler vs category words)   (= sub-2026-09-26-8, public 0.987248)
#   3. France "completely different street" rule                         (= sub-2026-09-27-2, public 0.98806)
# Prerequisite: README steps 0-4 with BER_CE=e5base2 (cache/v6/s2/test_pred_e5base2.parquet).
set -e
export BER_RUN=v6 BER_WORK=v6 BER_CE=e5base2
export BER_TEST_PRED=cache/v6/s2/test_pred_e5base2.parquet BER_DECISION=experiments/v6/decision_params_s2_e5base2.json
python -m ber.pipeline test --steps write
python -m ber.experiments.france_category_swap              # category words + drops (label-free, from accepted test pairs)
python -m ber.experiments.france_category_swap --write output
python -m ber.experiments.france_street_diff --write output --only street
sha256sum output/matching_results.tsv output/candidate_pairs.tsv
