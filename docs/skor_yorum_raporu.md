# Local Validation and Leaderboard Risk

**Status:** Updated 15 July 2026

## Accepted Local Evidence

The accepted weighted LightGBM/XGBoost candidate has cross-fitted grouped
Macro-F1 `0.837508`. LightGBM alone scores `0.837304` and XGBoost alone scores
`0.836820`. The deploy threshold `0.3718157097697258` is fitted on all OOF rows
and is a deployment parameter, not an unbiased score.

Earlier values near `0.96` are historical and invalid for comparison because
they used row-level validation, incomplete positive exclusion, an obsolete
attribute parser, or test-unshaped negative ratios.

## Generalization Controls

- `StratifiedGroupKFold(group=term_id)` keeps each query in exactly one fold.
- Fold-specific weights and thresholds are selected outside the evaluated fold.
- Negative sampling excludes all 250,000 known positives.
- Candidate quotas match the observed submission group-size rule.
- BM25 share was selected using an unlabeled distribution diagnostic, then
  accepted through grouped OOF behavior.
- Source data, models, OOF arrays, decision, and final CSV are hash-bound.

## Residual Risk

Synthetic negatives can still differ from hidden labels. The final CSV predicts
19.2216% positives while the training-candidate positive prevalence is 13.3142%;
this is a covariate-shift signal, not proof of an error. A tested large-group
cardinality override reduced cross-fitted Macro-F1 from `0.837508` to `0.835422`
and was therefore rejected.

There is no observed Kaggle public score in the repository. Upload and
leaderboard recording require authorized team-account access; no projection is
substituted for that observation.
