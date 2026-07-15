# Acceptance Runs

Only reproducible local evidence is recorded here. Sample scores validate pipeline behavior and must not be used as leaderboard projections.

## Regression And Contracts

| Check | Result |
|---|---|
| Unit/integration suite | 95/95 passed |
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

## Full Production Acceptance

Training and inference ran from clean revision
`f22e1e66a1e06879d29905637ecbfe6c0cfc6604`.

| Field | Result |
|---|---|
| Complete training terms | 17,968 |
| Positive / negative rows | 250,000 / 1,627,700 |
| Total training candidates | 1,877,700 |
| Negative sources | 316,893 BM25 / 814,401 category / 496,406 random |
| Models | 5 LightGBM + 5 XGBoost |
| Feature contract | 33 features |
| LightGBM cross-fitted Macro-F1 | 0.837304 |
| XGBoost cross-fitted Macro-F1 | 0.836820 |
| Blend cross-fitted Macro-F1 | **0.837508** |
| Selected deploy rule | 0.65 LightGBM + 0.35 XGBoost; threshold 0.37181571 |
| Training wall time | 48:28.11 |
| Training peak RSS | 3,765,812 KB; no swap |
| Submission rows | 3,359,679 |
| Positive predictions | 645,783 (19.2216%) |
| Submission QA | schema, binary values, unique/exact ID order, row count, and atomic publication passed |
| Selection/inference wall time | 1:11.70 |
| Selection/inference peak RSS | 701,084 KB |

Every accepted fold stopped before the 3,000-round ceiling. LightGBM best
iterations were 2,526, 2,056, 2,071, 1,981, and 2,464; XGBoost best iterations
were 2,253, 1,901, 2,070, 1,963, and 2,232.

## Delivery Integrity

| Artifact | SHA-256 |
|---|---|
| `outputs/submission_v2.csv` | `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c` |
| `outputs/submission_v2.manifest.json` | `972c0fbada3be885d993e13010c501c338ba86b8d8e62eccc8d3d51423c4a94e` |
| `outputs/ensemble_artifacts/oof_manifest.json` | `fdb9582e4cef39572059b7731b44e473d78adbcfe75d08398c8c39e3258f9eb9` |
| `outputs/ensemble_artifacts/ensemble_decision.json` | `7aab740498895f6b60df182cef5c2401c9b0173a152049224b7159395cdfccc4` |

The full OOF consumer independently reloaded and verified all hashes, source
data, feature columns, fold IDs, array shapes, and full-mode row counts.

## External Or Conditional Evidence

- Full sentence embeddings and their grouped ablation require a local model
  checkpoint and are not part of the accepted feature contract.
- Kaggle public/private leaderboard scores require team-account access.

Neither item may be inferred from local validation or represented as complete
until its own artifacts or external record exists.
