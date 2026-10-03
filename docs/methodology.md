# Methodology and final submission

## Candidate retrieval

The task maps each canonical Source-1 business to zero or more noisy records from Sources 2 and 3. Countries are processed separately, but the matcher is designed to handle an unseen country rather than discard its records. France lacks training labels, which makes its calibration a separate source of uncertainty.

Formatting normalization removes Unicode, punctuation, placeholder, and number-format differences. Value mappings such as transliteration and aliases are learned from training pairs. High-confidence pseudo-pairs supply label-free test adaptation. The historical system did not perform external business or identity lookups.

The v6 blocker unions dense forward and reverse retrieval with exact-name/address-number and character-gram channels. Dense search uses a fine-tuned multilingual E5-small bi-encoder and an inverted-file index. Each Source-1 entity receives at most 30 candidates; the recorded mean before the stage-1 filter was approximately 24.8. Stage 1 retained pairs with probability at least 0.01 for the submitted candidate file, approximately 4.73 per test entity.

## Pair matching and competition

The first LightGBM matcher uses string similarities, IDF-weighted coverage, address-number agreement, record flags, embedding cosine, retrieval ranks, and entity context. Stage 2 adds competition: a record's score at other entities, rank and gap, and the entity's strongest scores. Stage-1 scores used for fitting these features are cross-fitted. A fine-tuned E5-base cross-encoder supplies an additional feature on the uncertain stage-1 band.

Stage 3 adds two Qwen sequence-classification models fine-tuned with LoRA on provided pairs. They run locally and offline; their probabilities feed a LightGBM combiner rather than a simple average. The broader v4 combiner also includes a stage-2 challenger and competition features. Model identifiers and the recorded licenses/revisions are in [NOTICE.md](../NOTICE.md).

## Validation boundaries

The historical split held out approximately 200,000 Source-1 entities and divided them into halves A and B. Evaluation considered both a normal world and a test-like world in which 20% of reference entities were removed, leaving records that should remain unmatched. This tests open-set behavior as well as matching accuracy.

Stages 1 and 2 were fitted using the training portion. Stage-3 combiner development fitted half A and evaluated half B, with a reverse-direction check. The final combiner was later refitted on both halves A and B. Consequently, earlier half-B results describe the earlier evaluation protocol; they are not an untouched holdout evaluation of the final refitted model.

The recorded v6 two-way macro-F0.5 in the 20%-ghost world was approximately 0.98951 / 0.98941 before later final assembly. Local labeled checks cover the US and India. France's hidden labels and the public leaderboard's country mixture were unavailable, so transfer estimates for France were assumptions rather than measured validation.

## Final assembly: #27-11 to #27-19

The final package includes source for each assembly step under `src/ber/final2719/`:

1. Selective France stacking, with number-conflict checks, combines the earlier France decisions with a stage-3 variant.
2. Enforce one Source-1 owner per matched record and source-specific count caps derived from the training structure.
3. Refit the stage-3 combiner on A+B for US/India and restrict accepted matches to the candidate file.
4. Apply the remaining France house-number veto pairs held back by an earlier high-confidence threshold.
5. Use the independent CPU matcher as a tie-breaker on uncertain US/India pairs. It contributes a different error pattern despite being weaker on its own.
6. Refine decisions using per-entity expected F0.5.
7. Correct an audited restoration error in the last refinement.

The net change from #27-11 was 7,212 drops and 5,853 additions. The final output contained 5,829,875 matched pairs. Those pair lists and prediction files contain challenge identifiers and are held privately; the public source describes and implements their generation/application.

## Results and limits

The recorded public macro-F0.5 increased from 0.955067 for the first model-based pipeline to 0.989213 for #27-11 and **0.989275 for #27-19**. These are historical submission-record values, not scores recomputed for this source release. The organizers list GitBoosters under [Top 50 Teams](https://docs.google.com/spreadsheets/d/e/2PACX-1vQDMSduegekOef-4HZUTZcV-OVzn-3jymCVab6Xeuw5W934u25dSbetTJ65ikoWUxmIPraEakUYZ8QH/pubhtml?gid=1284921920); that list does not state an exact rank or private score.

The useful gains came from indexed multilingual retrieval, competition-aware matching, cross-encoder and LLM features, and targeted post-processing. Earlier experiment modules remain as implementation references, including approaches that were not selected. Exact submission reconstruction depends on the privately retained data and frozen artifacts. Full retraining may change floating-point results and threshold decisions.
