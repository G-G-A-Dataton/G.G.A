# July 16 Team Task Audit

This record accepts all four team-member tasks planned for 16 July 2026. It
uses repository evidence only and does not infer Kaggle account activity.

## Progress

- July 16 member tasks: **4/4 complete (100%)**.
- Team plan through 16 July: **62/64 complete or formally closed (96.9%)**.
- Repository-verifiable/local engineering cells: **62/62 complete (100%)**.
- Remaining cells: two pre-existing Kaggle account actions from 1-15 July.

## Member Deliverables

| Member | Planned output | Status | Acceptance evidence |
|---|---|---|---|
| Ömer Faruk Kara | Two final submission candidates and strategy | Complete | [`final_submission_candidates.md`](final_submission_candidates.md); full OOF ranking; primary blend and LightGBM fallback |
| Ahmet Emin Işın | Final report draft v2 and cleaned experiment table | Complete | [`final_solution_report_v2.md`](final_solution_report_v2.md) and [`experiment_log.md`](experiment_log.md) |
| Mustafa Mert Çevik | Column, row, ID, and binary submission QA approval | Complete | [`submission_qa_approval.md`](submission_qa_approval.md); 2/2 full candidates passed |
| Muhammed Köseoğlu | Clean-environment reproducibility dry-run | Complete | [`reproducibility_dry_run.md`](reproducibility_dry_run.md); accepted CSV reproduced byte for byte |

## Final Candidate Evidence

Candidate packaging executed from clean revision
`212b296cf9988444e7b06531e0e23b71994411f1` using full artifacts from revision
`f22e1e66a1e06879d29905637ecbfe6c0cfc6604`.

| Rank | Candidate | Cross-fitted Macro-F1 | Rows | Positives | SHA-256 |
|---:|---|---:|---:|---:|---|
| 1 | LGB/XGB 0.65/0.35 blend | 0.837508 | 3,359,679 | 645,783 | `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c` |
| 2 | LightGBM | 0.837304 | 3,359,679 | 634,316 | `c9407c965ad7ba047441aa44952dab4681a13e23d32fb494b1c31545c83052e7` |

Candidate 1 is byte-identical to the accepted 15 July delivery. The candidates
disagree on 21,647 rows (`0.6443%`). Candidate 2 is a controlled model-family
fallback; the grouped local evidence continues to select Candidate 1.

## QA Contract

Both final CSV files passed bounded-memory checks for:

- exact `id,prediction` column order;
- exactly 3,359,679 rows;
- integer `{0,1}` values without nulls;
- exact sample-submission ID order and global uniqueness;
- no accidental index column;
- stable SHA-256 and recorded class balance.

The current regression/integration suite has **114/114 passing tests**. The
configuration test also ensures `configs/final_v1.json` cannot silently drift
from the accepted BM25/category/random production candidate contract.

## Reproducibility Evidence

Muhammed's dry-run used a fresh Python 3.13.5 environment with all 158
hash-locked packages, a detached clean clone, disabled user-site and online
model access, and full data/artifact/delivery verification. It rebuilt the
accepted 3,359,679-row submission with the same SHA-256.

Full model retraining is the separately accepted 48-minute production run. The
dry-run validates its immutable artifacts and completely re-executes delivery
inference.

## Remaining External Actions

Kaggle team/access confirmation and authorized upload/public score recording
remain external. They cannot be truthfully completed by repository code and do
not reduce the 4/4 completion of the 16 July plan cells.
