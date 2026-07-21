# Candidate Artifact Reconciliation

## Decision

The upload-eligible set is the pair recorded in the local
`outputs/final_candidates/candidate_set.json` and bound to the accepted
delivery manifest:

1. 65% LightGBM / 35% XGBoost blend, SHA-256 `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c`.
2. LightGBM fallback, SHA-256 `c9407c965ad7ba047441aa44952dab4681a13e23d32fb494b1c31545c83052e7`.

## Reconciliation Record

The 16 July repository synchronization introduced reports for a separate
fresh run with a 45%/55% blend and XGBoost fallback. Its reported CSV and model
hashes are not present in this workspace because production outputs are
gitignored. A report without its corresponding files cannot pass local QA,
cannot be reproduced by `step3.sh`, and must not be selected on Kaggle.

This is an artifact-availability decision, not a claim about hidden-set model
quality. The shared workspace retains the earlier accepted set because both
CSV files are present, their complete model bundle is present, all source and
artifact hashes validate, and the primary candidate is reproducible from the
offline model-loading path.
