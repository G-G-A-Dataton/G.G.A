# Final Kaggle Selection Note

- Repository decision: **frozen**
- Kaggle upload/selection state: **pending authorized team-account action**
- Leaderboard signal used for local ordering: **none**

## Selected Upload Pair

| Priority | File | Strategy | Local grouped Macro-F1 | SHA-256 |
|---:|---|---|---:|---|
| 1 | `outputs/final_candidates/submission_candidate_1_weighted_blend.csv` | LGB/XGB 0.65/0.35, threshold 0.37181571 | 0.837508 | `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c` |
| 2 | `outputs/final_candidates/submission_candidate_2_lightgbm.csv` | LightGBM, threshold 0.38338208 | 0.837304 | `c9407c965ad7ba047441aa44952dab4681a13e23d32fb494b1c31545c83052e7` |

Both files have 3,359,679 rows, exact sample ID order, unique IDs, integer
binary predictions, and no index column. Candidate 1 is the default because it
has the strongest leakage-free grouped validation score. Candidate 2 is the
strongest locally available single-family fallback.

The account holder must upload both CSV files and manually mark both as final
submissions before the competition closes. The observed Kaggle submission IDs,
scores, and selection timestamp must only be added after that account action;
the repository does not fabricate them.

The finalist solution dataset is a later and conditional action: only a Top-20
candidate creates the private dataset `GGA_Trendyol2026_Solution` between
18 and 25 July and shares it with `trendyoldatascience` and `coderspacetr`.
