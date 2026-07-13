# Local Validation and Leaderboard Risk

**Status:** Updated 13 July 2026

There is currently no approved local Macro-F1 score, fixed threshold, or
leaderboard projection. Earlier values near `0.96` were generated with
row-level validation that allowed the same query group in train and validation.
They are invalid for generalization claims.

## Current Controls

- `StratifiedGroupKFold(group=term_id)` keeps a query entirely in one fold.
- Negative sampling excludes all known positive pairs, including positives
  outside an experiment's sampled training subset.
- Attribute features parse the real flat catalog format.
- Threshold selection uses only new grouped OOF probabilities.
- Production inference accepts only full-training, hash-verified artifacts.

## Remaining Risks

Synthetic negatives may still differ from the hidden test-label distribution,
and grouped local validation cannot prove leaderboard performance. Model
selection must use repeated grouped experiments plus real leaderboard results;
leaderboard values must be recorded as observations, never projected from local
scores.

The next valid baseline is defined in [`model_status.md`](model_status.md).
