# Final Submission QA Approval

- Status: **PASS**
- Approved candidates: `2/2`
- Reference rows: `3,359,679`
- Sample submission SHA-256: `6e75f1dcf4dab4b127a1ea988041a36f6cb1ec0b190cd9461c4cb144fb30adeb`

## Mandatory Checks

Both candidates passed bounded-memory validation for:

- exact `id,prediction` column order;
- exactly 3,359,679 data rows;
- integer predictions restricted to `{0, 1}` with no nulls;
- exact sample-submission ID order and global ID uniqueness;
- no accidental index column;
- recorded row counts, class balance, and SHA-256.

## Approved Files

| Rank | File | Rows | Positives | SHA-256 |
|---:|---|---:|---:|---|
| 1 | `outputs/final_candidates/submission_candidate_1_weighted_blend.csv` | 3,359,679 | 645,783 | `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c` |
| 2 | `outputs/final_candidates/submission_candidate_2_lightgbm.csv` | 3,359,679 | 634,316 | `c9407c965ad7ba047441aa44952dab4681a13e23d32fb494b1c31545c83052e7` |

Candidate 1 additionally matches the accepted delivery SHA-256 byte for
byte. Any later modification invalidates this approval and requires the
candidate-set command to be rerun before upload.
