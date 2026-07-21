# Model Validation Status

**Status on 15 July 2026: the full production model, model selection, final local
submission candidate, and delivery manifest are accepted.**

All scores recorded before the current contracts remain historical only. They used at least one invalid condition: row-level validation, incomplete positive exclusion, inactive attributes, synthetic embeddings, or a fixed negative ratio unlike the test candidate distribution.

## Current Acceptance Contract

- 250,000 known positives and 17,968 complete training query groups.
- 1,877,700 training candidates using `max(100, ceil(2 * positives))` per query.
- BM25-hard, category-hard, and catalog-random negatives exclude the complete positive reference.
- 33 model features: 23 base, TF-IDF cosine, and 9 candidate-relative context features.
- Five `StratifiedGroupKFold` folds grouped by `term_id`.
- Cross-fitted threshold and ensemble evaluation.
- Data, code, feature, model, OOF, and decision artifacts protected by version/hash manifests.
- Global test context computed out of core; submission query groups are not assumed contiguous.
- Atomic binary submission publishing after exact ID/row/schema QA.

## Accepted Full Run

- 108 passing regression/integration tests.
- 17,968 complete terms, 250,000 positives, and 1,877,700 candidates.
- 316,893 BM25, 814,401 category, and 496,406 random negatives.
- Five LightGBM plus five XGBoost grouped folds with 33 features.
- Selected 0.65/0.35 weighted blend: cross-fitted Macro-F1 `0.837508`.
- Deploy threshold `0.3718157097697258`.
- 3,359,679-row QA-passing submission with 645,783 positive predictions.
- Hash-bound OOF, decision, submission, and delivery manifests.
- 48:28.11 training wall time and 3,765,812 KB peak RSS without swap.

Historical smoke scores remain pipeline evidence only and are not used for the
accepted model decision.

On 16 July, a fresh hash-locked Python environment and detached clean clone
re-ran all verification gates and rebuilt the complete submission byte for
byte. This reproducibility acceptance does not change the 15 July model or its
validation score; see
[`reproducibility_dry_run.md`](reproducibility_dry_run.md).

The 16 July final set retains the accepted blend as Candidate 1 and the
next-best cross-fitted LightGBM model as Candidate 2. Both complete files pass
exact QA; see [`final_submission_candidates.md`](final_submission_candidates.md)
and [`submission_qa_approval.md`](submission_qa_approval.md).

Reproduce or verify the accepted sequence from [`RUNBOOK.md`](../RUNBOOK.md).
Only the Kaggle upload and leaderboard observation require external team-account
access; no leaderboard score is fabricated locally.
