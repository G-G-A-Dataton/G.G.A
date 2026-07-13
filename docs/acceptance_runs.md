# Acceptance Runs

Only reproducible local evidence is recorded here. Sample scores validate pipeline behavior and must not be used as leaderboard projections.

## Regression And Contracts

| Check | Result |
|---|---|
| Unit/integration suite | 78/78 passed |
| Python | 3.13.5 verified |
| Direct package pins | 17/17 exact |
| Frozen competition files | 5/5 row/schema/size/SHA-256 checks passed |
| Full submission input | 3,359,679 rows and sample ID order verified |

## Canonical LightGBM Smoke

| Field | Result |
|---|---|
| Complete terms | 20 |
| Test-shaped candidates | 2,396 |
| Group folds | 5 |
| Output | hash-manifested sample artifacts |

## Dual-Model Smoke

| Field | Result |
|---|---|
| Complete terms | 10 |
| Candidates | 1,000 |
| Models | 5 LightGBM + 5 XGBoost |
| Test rows | 200 |
| Selected candidate | XGBoost |
| Cross-fitted Macro-F1 | 0.884613 |
| Deploy threshold | 0.13285412 |
| Submission QA | passed |

## Medium Dual-Model Acceptance

Command executed from clean Git revision `3cae90d`:

```bash
python scripts/training/run_model_shortlist.py \
  --sample-terms 300 --test-sample 50000 --batch-size 25000 \
  --num-boost-round 200 --early-stopping-rounds 30 \
  --artifact-dir outputs/ensemble_medium_artifacts
```

| Field | Result |
|---|---|
| Complete terms | 300 |
| Positive rows | 4,955 |
| Test-shaped candidates | 33,048 |
| Category negatives | 14,052 |
| Catalog-random negatives | 14,041 |
| Test predictions | 50,000 per model family |
| LightGBM cross-fitted Macro-F1 | 0.939451 |
| XGBoost cross-fitted Macro-F1 | 0.937968 |
| Blend cross-fitted Macro-F1 | 0.937536 |
| Selected candidate | LightGBM |
| Deploy threshold | 0.41054133 |
| Submission QA | 50,000/50,000 rows passed |
| Wall time | 2:08.17 |
| Peak RSS | 1,462,884 KB |

The feature-importance consumer also loaded all five verified LightGBM folds and produced a 33-feature aggregate without contract errors.

## BM25 Resource Smoke

| Field | Result |
|---|---|
| Real catalog products | 20,000 |
| Index wall time | 2.71 seconds |
| Peak RSS | approximately 140 MB |
| Indexed tokens | 33,751 |

## Not Yet Evidence

- Full 17,968-term dual-model training and 3,359,679-row final submission
- Full optional embedding generation and grouped full-data ablation
- Kaggle public/private leaderboard scores

These items cannot be described as complete until their manifests, logs, and external scores exist.
