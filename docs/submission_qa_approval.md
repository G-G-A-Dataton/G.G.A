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
| 1 | `outputs/final_candidates/submission_candidate_1_weighted_blend.csv` | 3,359,679 | 625,585 | `d40b0338a5593e00563662a2fa5376e967c8e64f8b6490d4d5864a93a0a6281c` |
| 2 | `outputs/final_candidates/submission_candidate_2_xgboost.csv` | 3,359,679 | 622,915 | `f1ea967ec9e63e7af4aa384434ead8996cca022a04b268482e47bb5e33f59a45` |

Candidate 1 additionally matches the accepted delivery SHA-256 byte for
byte. Any later modification invalidates this approval and requires the
candidate-set command to be rerun before upload.
