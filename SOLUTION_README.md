# G.G.A Finalist Solution Workflow

This repository contains the complete data preparation, grouped validation,
training, model serialization, and offline inference workflow for the accepted
G.G.A ensemble. The Kaggle competition data is not redistributed.

## Required Competition Files

Place the original files in one directory without renaming them:

- `terms.csv`
- `items.csv`
- `training_pairs.csv`
- `submission_pairs.csv`
- `sample_submission.csv`

Their schemas, row counts, byte sizes, and SHA-256 values are frozen in
`configs/final_v1.json`. Every production run rejects changed source data.

## Step 1: Environment

```bash
bash step1.sh
```

This creates `.solution_venv` with Python 3.13.5, installs all 158 transitive
packages from the hash-locked `requirements.lock`, and verifies every version.
Use `--env-path PATH` or `GGA_ENV_PATH` to choose another environment path.

## Step 2: Data Generation And Training

```bash
bash step2.sh \
  --competition_data_path competition_data/ \
  --extra_data_path extra_generated_data/ \
  --model_dump_path new_trained_models/
```

The model directory must be empty. This step deterministically generates the
exact training candidate matrix, writes it as
`generated_training_candidates.csv`, trains five LightGBM and five XGBoost
grouped folds, saves the TF-IDF vectorizer and OOF/test arrays, and recomputes
the deploy decision. SHA-256 manifests bind all generated data and artifacts.

No external, synthetic, pseudo-labeled, paid-service, generative-model, or
embedding data is used by the accepted solution. All negative examples are
generated locally from the original catalog with seed 42.

## Step 3: Offline Inference

Disconnecting the machine from the network does not affect this step:

```bash
bash step3.sh \
  --model_dump_path models/ \
  --competition_data_path competition_data/ \
  --out_path submission.csv
```

The routine verifies the data/model hashes, loads all ten fold models and the
TF-IDF vectorizer, rebuilds test features out of core, applies the frozen
ensemble and threshold, and validates the final `id,prediction` CSV. It does
not consume cached final labels or make network calls.

## Accepted Model

- Validation: five-fold `StratifiedGroupKFold`, grouped by `term_id`
- Candidate selection: cross-fitted Macro-F1
- Model: 65% LightGBM / 35% XGBoost
- Threshold: `0.3718157097697258`
- Local cross-fitted Macro-F1: `0.8375076143`
- Accepted submission SHA-256:
  `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c`

See `RUNBOOK.md` for acceptance gates and `docs/hardware_environment.md` for
the development hardware and measured resource envelope.
