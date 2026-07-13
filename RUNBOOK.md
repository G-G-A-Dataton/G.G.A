# G.G.A Production Runbook

This document is the canonical training, inference, and submission QA procedure.
Run every command from the repository root with the project virtual environment
activated.

## 1. Verify Code and Data

```bash
python -m unittest discover -s tests -v
python scripts/data/verify_pipeline.py
```

The data contract is:

| File | Rows |
|---|---:|
| `terms.csv` | 50,153 |
| `items.csv` | 962,873 |
| `training_pairs.csv` | 250,000 |
| `submission_pairs.csv` | 3,359,679 |
| `sample_submission.csv` | 3,359,679 |

## 2. Train Production Artifacts

```bash
python scripts/training/run_train_full_v2.py
```

Training uses leakage-free BM25/random negative sampling and 5-fold
`StratifiedGroupKFold` with `group=term_id`. A successful full run writes five
LightGBM models, the TF-IDF vectorizer, the optimized threshold, and
`outputs/model_manifest_v2.json`. The manifest is written after all required
inference artifacts and records the feature schema, validation contract, and
SHA-256 hashes.

Quick training tests are isolated from production artifacts:

```bash
python scripts/training/run_train_full_v2.py --sample 10000 --no-error-analysis
```

Sample artifacts go to `outputs/sample_artifacts_v2/` and production inference
rejects them.

## 3. Run Inference

Use sample mode before the full run:

```bash
python scripts/submission/run_pipeline.py --mode predict --sample 10000 --batch-size 5000
python scripts/submission/run_pipeline.py --mode predict
```

The pipeline validates artifact hashes and feature schema, loads the threshold
from the production artifact set, and streams batch predictions through a
temporary CSV. Only a QA-passing file atomically replaces
`outputs/submission_v2.csv`; a failed run preserves the previous output. Sample
mode writes a separate sample output.

## 4. Submission Contract

The final CSV must have exactly these columns in this order:

```text
id,prediction
```

It must contain 3,359,679 rows, preserve the exact ID order from
`datasets/sample_submission.csv`, contain integer predictions only, and contain
no null or duplicate IDs. The inference pipeline runs this QA automatically.
It can also be run explicitly:

```bash
python -m src.validate_submission \
  outputs/submission_v2.csv datasets/sample_submission.csv
```

Logs are written to `outputs/pipeline.log`. A failed manifest, missing artifact,
unresolved ID, feature mismatch, or QA check stops the pipeline with a non-zero
exit status.
