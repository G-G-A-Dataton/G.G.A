# Final Submission Candidate Strategy

- Status: **PASS**
- Source revision: `212b296cf9988444e7b06531e0e23b71994411f1`
- Artifact revision: `f22e1e66a1e06879d29905637ecbfe6c0cfc6604`
- Ranking signal: cross-fitted grouped Macro-F1 only
- Leaderboard signal used: no

## Ranked Candidates

| Rank | Candidate | Cross-fitted Macro-F1 | Weights (LGB/XGB) | Threshold | Positives | Positive rate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Weighted LightGBM/XGBoost blend | 0.837508 | 0.65/0.35 | 0.37181571 | 645,783 | 19.2216% |
| 2 | LightGBM | 0.837304 | 1.00/0.00 | 0.38338208 | 634,316 | 18.8803% |

## Decision

Candidate 1 remains the default upload because it has the highest
leakage-free cross-fitted score and is byte-identical to the accepted
15 July delivery. Candidate 2 is the strongest single-family fallback;
it provides model diversity without using leaderboard feedback.

The candidates disagree on `21,647` rows (`0.6443%`). This is a
controlled alternative, not evidence that the runner-up is expected to
score higher. Kaggle upload and observed scores remain account-bound.

## Integrity

- Candidate 1 SHA-256: `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c`
- Candidate 2 SHA-256: `c9407c965ad7ba047441aa44952dab4681a13e23d32fb494b1c31545c83052e7`
