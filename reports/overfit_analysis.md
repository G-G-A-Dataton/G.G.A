# Overfitting Analysis

**Status:** Updated 13 July 2026

Legacy local scores and leaderboard estimates are invalid because they predate
query-grouped validation, full-positive negative exclusion, and the corrected
attribute parser. No production score or fixed threshold is currently approved.

The active controls are:

- 5-fold `StratifiedGroupKFold` with `group=term_id`;
- exact, unique per-term negative quotas with complete positive exclusion;
- threshold optimization from grouped OOF predictions;
- a versioned feature schema and SHA-256 artifact manifest;
- strict submission ID, column, dtype, and row-count validation.

Residual risk remains from synthetic-negative distribution shift and repeated
model selection on the same grouped folds. New experiments should report fold
dispersion, preserve untouched comparison runs where feasible, and distinguish
real leaderboard observations from local metrics.

See [`docs/model_status.md`](../docs/model_status.md) for the acceptance gate.
