# Final Submission Candidate Strategy

- Status: **PASS**
- Source revision: `3dc2578bdafb7b333f03e40309b6bf4c7fb06bee`
- Artifact revision: `ee0cf31ed7856fe5e15362f63f8c13ff9984a44c`
- Ranking signal: cross-fitted grouped Macro-F1 only
- Leaderboard signal used: no

## Ranked Candidates

| Rank | Candidate | Cross-fitted Macro-F1 | Weights (LGB/XGB) | Threshold | Positives | Positive rate |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Weighted LightGBM/XGBoost blend | 0.837374 | 0.45/0.55 | 0.38757889 | 625,585 | 18.6204% |
| 2 | XGBoost | 0.836911 | 0.00/1.00 | 0.38713759 | 622,915 | 18.5409% |

## Decision

Candidate 1 remains the default upload because it has the highest
leakage-free cross-fitted score and is byte-identical to the accepted
15 July delivery. Candidate 2 is the strongest single-family fallback;
it provides model diversity without using leaderboard feedback.

The candidates disagree on `23,144` rows (`0.6889%`). This is a
controlled alternative, not evidence that the runner-up is expected to
score higher. Kaggle upload and observed scores remain account-bound.

## Integrity

- Candidate 1 SHA-256: `d40b0338a5593e00563662a2fa5376e967c8e64f8b6490d4d5864a93a0a6281c`
- Candidate 2 SHA-256: `f1ea967ec9e63e7af4aa384434ead8996cca022a04b268482e47bb5e33f59a45`
