# July 16 Muhammed Task Audit

This record accepts Muhammed Koseoglu's 16 July task: verify that the repository
is reproducible from a clean environment and publish the dry-run result.

## Progress

- Muhammed's 16 July task: **1/1 complete (100%)**.
- Team plan through 16 July: **59/64 complete or formally closed (92.2%)**.
- The denominator includes all four 16 July member cells. The other three cells
  are outside this acceptance and remain unclaimed here.
- The two pre-existing Kaggle account actions through 15 July remain external.

## Acceptance Evidence

| Contract | Result |
|---|---|
| Source revision | `ec619e29702503bbe4db8147a4c30a0a10daecda` |
| Source isolation | Clean detached local clone |
| Python | 3.13.5 in a fresh virtual environment |
| Dependency closure | 158/158 exact versions from hash-locked `requirements.lock` |
| User-site / online model access | disabled / disabled |
| Repository compile | passed for `src`, `scripts`, `pipeline`, `notebooks`, and `tests` |
| Unit/integration suite | 102/102 passed |
| Data freeze / data pipeline | passed / passed |
| Accepted delivery manifest | all source and artifact hashes passed |
| Submission reproduction | 3,359,679 rows; 645,783 positives; full QA passed |
| Determinism | reproduced CSV is byte-identical to accepted CSV |

Both accepted and reproduced submissions have SHA-256
`2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c`.
The measured gate timings and artifact revision are recorded in
[`reproducibility_dry_run.md`](reproducibility_dry_run.md).

## Reproduction Command

```bash
PYTHONNOUSERSITE=1 venv/bin/python \
  scripts/run_reproducibility_dry_run.py --python venv/bin/python
```

The runner fails on a dirty source tree, package drift, stale data or artifacts,
relationship violations, submission QA errors, or any byte difference in the
regenerated final CSV. Full model retraining is the separately measured
48-minute production operation documented in `acceptance_runs.md`; this dry-run
validates its immutable artifacts and completely re-executes delivery inference.
