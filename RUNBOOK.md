# G.G.A Production Runbook

This is the canonical offline-capable workflow. Run commands from the repository root with `venv` activated.

## 1. Acceptance Gate

```bash
python scripts/run_production.py --stage verify
```

The gate runs 77 regression tests, verifies every pinned package, checks the versioned data freeze in `configs/final_v1.json`, and validates all CSV relationships. Any mismatch stops the run.

## 2. Production Training

```bash
python scripts/run_production.py --stage train
```

The default shortlist trainer builds 1,877,700 test-shaped candidates from all 250,000 positives. Each query receives `max(100, ceil(2 * positives))` candidates; half of the negative quota targets the positive products' L2 categories and the remainder comes from the catalog. All known positives are excluded.

Five LightGBM and five XGBoost models use the same `StratifiedGroupKFold(group=term_id)` assignment and feature matrix. Test predictions are streamed through disk-backed feature stores because submission term groups are not contiguous.

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
