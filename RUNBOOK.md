# G.G.A Production Runbook

This is the canonical offline-capable workflow. Run commands from the repository root with `venv` activated.

## 1. Acceptance Gate

```bash
python scripts/run_production.py --stage verify
```

The gate runs 66 regression tests, verifies every pinned package, checks the versioned data freeze in `configs/final_v1.json`, and validates all CSV relationships. Any mismatch stops the run.

## 2. Production Training

```bash
python scripts/run_production.py --stage train
```

The trainer builds 1,877,700 test-shaped candidates from all 250,000 positives. Each query receives `max(100, ceil(2 * positives))` candidates; half of the negative quota targets the positive products' L2 categories and the remainder comes from the catalog. All known positives are excluded.

Five LightGBM models use `StratifiedGroupKFold(group=term_id)`. Threshold selection is cross-fitted: each validation fold is evaluated with a threshold selected only on the other folds. The deploy threshold is then selected from all OOF predictions and is explicitly labeled as a deployment parameter, not an unbiased score.

Production artifacts in `outputs/` include models, TF-IDF vectorizer, OOF predictions, threshold report, feature importance, and `model_manifest_v2.json`. The manifest binds feature schemas, Git revision, source-data hashes, candidate distribution, metrics, and artifact hashes.

Quick runs use complete query groups and cannot overwrite production artifacts:

```bash
python scripts/training/run_train_full_v2.py \
  --sample-terms 300 --num-boost-round 200 --no-error-analysis
```

## 3. Shortlist And Ensemble

```bash
python scripts/training/run_model_shortlist.py
python scripts/analysis/run_ensemble_optimization.py
```

The shortlist trains five LightGBM and five XGBoost folds on the same matrix. Test features are written to disk-backed stores because `submission_pairs.csv` does not keep query groups contiguous. Ensemble weight and threshold are selected outside each evaluated fold. A blend is promoted only when its cross-fitted Macro-F1 exceeds both single models.

## 4. Inference And QA

```bash
python scripts/run_production.py --stage predict
```

Inference verifies artifact and source hashes, computes global candidate-relative features out of core, streams predictions, and atomically publishes `outputs/submission_v2.csv` only after QA. The output must contain 3,359,679 unique IDs in exact sample order and integer predictions in `{0, 1}`.

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
