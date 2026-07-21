# Leaderboard Rescue Decision

**Status:** QA verified, Public Score not yet recorded.

## Failure diagnosis

The first two submissions scored around `0.6` while the accepted local grouped
validation was `0.837508`. This invalidates the synthetic-negative validation as
an absolute leaderboard estimate. The two uploaded files disagreed on only
`21,647` rows (`0.6443%`), so the second upload did not provide a meaningful
independent test.

The production sampler labels unobserved pairs as negatives. Its fixed mixture
contains `316,893` BM25, `814,401` category, and `496,406` random negatives.
Those labels are useful for controlled comparison but are not hidden Kaggle
ground truth. In particular, pretrained semantic similarity ranks BM25 negatives
above known positives on average, demonstrating that this negative regime
contains a different decision problem from random or category negatives.

## New evidence

The rescue stack adds signals that were absent from the uploaded model:

- multilingual static semantic similarity for combined item text, title/brand,
  and category views;
- leakage-free char-ngram nearest-query category priors;
- exact-category frequency, share, modal status, and group-relative ranks;
- group-relative semantic margins; and
- a five-fold LightGBM stack trained only on OOF base predictions.

Known positives occupy the modal exact category for `76.732%` of training query
groups. Exact-category frequency alone reaches `0.708259` synthetic grouped
Macro-F1. The complete stack improves cross-fitted Macro-F1 from `0.837508` to
`0.844334` and changes enough test rows to constitute a real model-family test.

`intfloat/multilingual-e5-small` was also evaluated on the uncertain probability
band. Its overall band AUC was only `0.515505`: it helped category and random
negative regimes but inverted BM25 ordering. A gated model produced only a
`0.000098` full-OOF gain while changing many test rows, so it is retained as a
feedback-only second candidate and is not the preferred upload.

## Submission protocol

1. Upload `submission_rescue_1_structural_semantic_stack.csv` first.
2. Record its displayed Public Score and rank before any other upload.
3. Do not upload the E5-gated candidate without comparing that observed score
   with both earlier submissions.
4. Preserve at least one daily submission for a threshold or gating response to
   the new leaderboard evidence.

The authoritative file hashes and class balances are written to
`outputs/rescue_candidates/rescue_candidate_set.json`. Public scores must be
recorded from Kaggle; no local metric may be substituted for them.

The preferred file contains `3,359,679` rows and `690,032` positive predictions
(`20.5386%`). Its SHA-256 is
`cee09771377aff5369ab47fd6a8d2cdbbb04db4ee84e4a56af57a71c0e2e3a21`.
The feedback-only E5 candidate contains `666,103` positives and disagrees on
`85,167` rows (`2.5350%`). Both files passed exact submission QA.
