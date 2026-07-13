# Model Validation Status

**Status on 13 July 2026: production code is accepted; a full production model has not yet been trained in this workspace.**

All scores recorded before the current contracts remain historical only. They used at least one invalid condition: row-level validation, incomplete positive exclusion, inactive attributes, synthetic embeddings, or a fixed negative ratio unlike the test candidate distribution.

## Current Acceptance Contract

- 250,000 known positives and 17,968 complete training query groups.
- 1,877,700 training candidates using `max(100, ceil(2 * positives))` per query.
- Category-hard and catalog-random negatives exclude the complete positive reference.
- 33 model features: 23 base, TF-IDF cosine, and 9 candidate-relative context features.
- Five `StratifiedGroupKFold` folds grouped by `term_id`.
- Cross-fitted threshold and ensemble evaluation.
- Data, code, feature, model, OOF, and decision artifacts protected by version/hash manifests.
- Global test context computed out of core; submission query groups are not assumed contiguous.
- Atomic binary submission publishing after exact ID/row/schema QA.

## Verified Runs

- Unit/integration suite: 77 passing tests.
- Canonical training smoke: 20 complete terms, 2,396 candidates, five folds, artifact manifest generated.
- Shortlist smoke: 10 complete terms, five LightGBM + five XGBoost folds, 200 test rows, valid OOF manifest and submission.
- BM25 smoke: 20,000 catalog items in 2.71 seconds, approximately 140 MB peak RSS.

Smoke scores are pipeline acceptance evidence only. They are too small for model selection or leaderboard projection.

Run the full acceptance sequence from [`RUNBOOK.md`](../RUNBOOK.md). After the full run, record the cross-fitted scores, deploy decision, artifact hashes, runtime/RSS, and actual Kaggle score in `docs/experiment_log.md`.
