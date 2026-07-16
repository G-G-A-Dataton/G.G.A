# Final Data And Negative Sampling Freeze

- Freeze date: 17 July 2026
- Data version: `data_v1_2026-07-13`
- Random seed: `42`
- External/synthetic/pseudo-labeled data: **none**
- Accepted training rows: `1,877,700`
- Positive rows: `250,000`
- Negative rows: `1,627,700`
- Complete training terms: `17,968`

## Source Data

| File | Rows | SHA-256 |
|---|---:|---|
| `terms.csv` | 50,153 | `9f94b15f225049bca882c50d8ee86f726d889456826d57a791bd3c4f33006812` |
| `items.csv` | 962,873 | `e420c87572fe1c4169d2021cb43a35813fe59f4d2cfe55e90eaf61a6fd477f21` |
| `training_pairs.csv` | 250,000 | `3e015fd88e0900729a9444227a21bcefaa7be661f8319f3f1afec5eb5cdc9463` |
| `submission_pairs.csv` | 3,359,679 | `2d89e1622156776d37185c7f8d1fd7c34eaf5f475f07fb6adf89f64314b60b56` |
| `sample_submission.csv` | 3,359,679 | `6e75f1dcf4dab4b127a1ea988041a36f6cb1ec0b190cd9461c4cb144fb30adeb` |

## Candidate Sampling Contract

Each query keeps every known positive and receives
`max(100, ceil(2 * positive_count))` candidates. The negative quota is filled
in this fixed order:

1. compact BM25 hard negatives: `20%`, `top_n=200`, `max_df_ratio=0.15`;
2. level-2 category hard negatives: `50%`;
3. deterministic catalog-random fill for the remaining quota.

All 250,000 known positives are excluded from every negative source. Full-run
source counts are 316,893 BM25, 814,401 category, and 496,406 random negatives.
The implementation and configuration are locked by
`src/candidate_sampling.py`, `configs/final_v1.json`, and
`outputs/ensemble_artifacts/oof_manifest.json`.
