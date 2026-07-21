# G.G.A Production Runbook

This is the canonical offline-capable workflow. Run commands from the repository root with `venv` activated.

## 1. Acceptance Gate

```bash
python scripts/run_production.py --stage verify
```

The gate runs 114 regression/integration tests, verifies all 158 hash-locked
packages from `requirements.lock`, checks the versioned data freeze in
`configs/final_v1.json`, and validates all CSV relationships. Any mismatch
stops the run.

## 2. Production Training

```bash
python scripts/run_production.py --stage train
```

The default shortlist trainer builds 1,877,700 test-shaped candidates from all
250,000 positives. Each query receives `max(100, ceil(2 * positives))`
candidates. The negative quota uses 20% compact-BM25 hard negatives, 50%
positive-product L2 category negatives, and deterministic catalog-random fill.
All known positives are excluded before any source is sampled.

Five LightGBM and five XGBoost models use the same `StratifiedGroupKFold(group=term_id)` assignment and feature matrix. Test predictions are streamed through disk-backed feature stores because submission term groups are not contiguous.

The accepted production configuration permits 3,000 boosting rounds with 200
rounds of early stopping. All ten accepted folds stopped naturally between
iterations 1,901 and 2,526; the ceiling is therefore not truncating the final
models.

Production artifacts in `outputs/ensemble_artifacts/` include ten models, TF-IDF vectorizer, OOF labels/folds/predictions, full test predictions, and `oof_manifest.json`. The manifest binds feature schemas, a clean Git revision, source-data hashes, candidate distribution, and every artifact hash.

Quick runs use complete query groups and cannot overwrite production artifacts:

```bash
python scripts/training/run_model_shortlist.py \
  --sample-terms 300 --test-sample 50000 --num-boost-round 200
```

The single-LightGBM fallback remains available explicitly:

```bash
python scripts/run_production.py --stage train --pipeline lightgbm
```

## 3. Model Selection

```bash
python scripts/run_production.py --stage predict
```

LightGBM, XGBoost, and weighted blend candidates are compared by cross-fitted Macro-F1. Each validation fold uses weights and a threshold selected only on the other folds. A blend is promoted only when it beats both single models. The deploy threshold is selected on all OOF rows and is labeled as a deployment parameter, not an unbiased score.

## 4. Inference And QA

```bash
python scripts/run_production.py --stage predict
```

Selection verifies artifact and current source-data hashes, streams the selected full-test probabilities, and atomically publishes `outputs/submission_v2.csv` only after QA. The output must contain 3,359,679 unique IDs in exact sample order and integer predictions in `{0, 1}`.

The accepted 15 July delivery selects a 65% LightGBM / 35% XGBoost blend at
threshold `0.3718157097697258`. Its cross-fitted grouped Macro-F1 is
`0.8375076143`. The published local candidate has 645,783 positive predictions
and is bound to the OOF manifest and ensemble decision by
`outputs/submission_v2.manifest.json`.

The LightGBM fallback inference independently recomputes global candidate-relative features out of core:

```bash
python scripts/run_production.py --stage predict --pipeline lightgbm
```

```bash
python -m src.validate_submission \
  outputs/submission_v2.csv datasets/sample_submission.csv
```

## 5. Embeddings

Embeddings are optional until a grouped comparison proves a gain. They are never silently replaced with synthetic or zero vectors.

```bash
python src/embedding_batch.py --target both \
  --model models/paraphrase-multilingual-MiniLM-L12-v2 --offline
python scripts/embedding/run_embedding_score_comparison.py
```

Each embedding matrix requires its hash manifest. Missing models, stale checkpoints, missing IDs, or model mismatches stop the experiment.

The local sentence-transformer checkpoint is not part of the accepted 15 July
delivery. Do not describe embeddings as a production feature unless both full
matrices and a positive grouped ablation are present. The final accepted model
uses lexical, parsed attribute, TF-IDF, and candidate-relative features.

## 6. Clean-Environment Reproducibility

Install `requirements.lock` into a fresh Python 3.13.5 environment, then run:

```bash
PYTHONNOUSERSITE=1 venv/bin/python \
  scripts/run_reproducibility_dry_run.py --python venv/bin/python
```

The runner requires a clean source worktree and creates a detached local clone.
It disables Python user-site and online Hugging Face/Transformers access, then
executes compile, tests, strict environment verification, frozen-data checks,
pipeline checks, accepted-delivery validation, complete submission
reproduction, and submission QA. The run passes only when the reproduced CSV
is byte-identical to the accepted CSV.

The accepted 16 July run passed 102 tests and reproduced SHA-256
`2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c`.
See [`docs/reproducibility_dry_run.md`](docs/reproducibility_dry_run.md).

## 7. Final Candidate Packaging

Build the two highest-ranked full submission candidates from the verified OOF
and test probabilities:

```bash
python scripts/submission/run_final_candidate_set.py
```

The command requires a clean Git revision, revalidates the accepted delivery
and full OOF artifacts, ranks candidates by cross-fitted grouped Macro-F1,
streams both complete CSV files, and runs exact submission QA. It publishes
only when Candidate 1 is byte-identical to the accepted delivery.

Outputs are written under `outputs/final_candidates/`. The approved strategy,
hashes, and QA evidence are recorded in
[`docs/final_submission_candidates.md`](docs/final_submission_candidates.md)
and [`docs/submission_qa_approval.md`](docs/submission_qa_approval.md).

## 8. Official Finalist Entry Points

The organizer-facing interface is independent of repository-relative dataset
and output paths:

```bash
bash step1.sh
bash step2.sh \
  --competition_data_path competition_data/ \
  --extra_data_path extra_generated_data/ \
  --model_dump_path new_trained_models/
bash step3.sh \
  --model_dump_path models/ \
  --competition_data_path competition_data/ \
  --out_path submission.csv
```

Step 2 exports every generated positive/negative candidate and its manifest.
Step 3 loads all ten fold models and rebuilds features from the supplied data;
it does not turn cached test probabilities into a nominal inference result.
See [`SOLUTION_README.md`](SOLUTION_README.md) for the complete contract.
