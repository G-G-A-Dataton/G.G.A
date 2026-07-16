# Reproducibility Freeze

- Freeze date: 17 July 2026
- Python: `3.13.5`
- Locked distributions: `158`
- Regression/integration tests: `114/114 PASS`
- Accepted model artifact revision: `f22e1e66a1e06879d29905637ecbfe6c0cfc6604`

## Integrity Surface

| Surface | Frozen evidence | Status |
|---|---|---|
| Source data | `configs/final_v1.json` schema/size/row/SHA contract | PASS |
| Negative sampling | deterministic seed/quota and full distribution manifest | PASS |
| Model files | five LightGBM + five XGBoost fold files | PASS |
| TF-IDF | serialized 10,000-term vectorizer with manifest hash | PASS |
| Feature cache | OOF and complete test arrays bound by `oof_manifest.json` | PASS |
| Deploy decision | recomputed from grouped OOF arrays before inference | PASS |
| Candidate CSVs | exact row/ID/binary/hash QA for both files | PASS |
| Runbook | canonical verify/train/predict and finalist steps documented | PASS |
| Offline entrypoint | model-loading `step3.sh`, network-independent | PASS |
| Full offline re-inference | 3,359,679 rows, 645,783 positives, accepted SHA-256 | PASS |

`step3.sh` does not copy cached labels into the output. It validates the model
bundle, loads all ten models, reconstructs base and group-relative features
from the supplied competition CSV files, predicts every row, applies the
frozen blend/threshold, and runs exact submission QA.

The full model-loading run reproduced SHA-256
`2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c`.
It is byte-identical to both `outputs/submission_v2.csv` and the primary final
candidate; no cached test probability file was read by the inference routine.

The historical clean-environment dry-run remains recorded in
`docs/reproducibility_dry_run.md`. The 17 July freeze extends that gate with
the official three-step finalist interface and explicit generated-negative
data export required by the organizer's final-delivery specification.
