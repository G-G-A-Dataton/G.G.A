# July 16 Reproducibility Dry-Run

- Status: **PASS**
- Source revision: `ec619e29702503bbe4db8147a4c30a0a10daecda`
- Artifact revision: `f22e1e66a1e06879d29905637ecbfe6c0cfc6604`
- Python: `3.13.5`
- Hash-locked packages: `158`
- Tests: `102/102` passed
- Network during dry-run: `disabled`
- Asset materialization: `copy`

## Deterministic Delivery Check

- Accepted submission SHA-256: `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c`
- Reproduced submission SHA-256: `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c`
- Byte-identical: **true**
- Rows: `3,359,679`
- Positive rows: `645,783`

## Executed Gates

| Gate | Seconds |
|---|---:|
| `compile` | 0.285 |
| `tests` | 9.067 |
| `environment` | 0.183 |
| `data_freeze` | 20.482 |
| `data_pipeline` | 25.066 |
| `accepted_delivery` | 8.170 |
| `submission_reproduction` | 90.489 |
| `submission_qa` | 10.849 |

The dry-run used a detached local clone, disabled Python user-site
packages and online model access, revalidated frozen data and all
delivery hashes, and rebuilt the final CSV from the accepted OOF/test
probabilities. Full model retraining remains a separate long-running
operation documented in `acceptance_runs.md`.
