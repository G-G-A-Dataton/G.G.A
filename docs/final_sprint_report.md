# Final Kaggle Sprint Report

- Archive date: 17 July 2026
- Metric: Macro-F1
- Evidence boundary: local grouped validation and repository integrity

## Current Result

The final solution uses deterministic test-shaped negative sampling, 33 fixed
lexical/TF-IDF/context features, five grouped LightGBM folds, and five grouped
XGBoost folds. Cross-fitted selection promotes the 65%/35% blend at threshold
`0.3718157097697258` with local Macro-F1 `0.8375076143`.

| Candidate | Cross-fitted Macro-F1 | Positive rows | Decision |
|---|---:|---:|---|
| Weighted LGB/XGB blend | 0.837508 | 645,783 | Primary final submission |
| LightGBM | 0.837304 | 634,316 | Final fallback submission |
| XGBoost | 0.836820 | 642,738 | Not selected |

## Experiment Decision Summary

- Grouped validation, not Kaggle feedback, is the model-selection authority.
- BM25/category/random sampling is frozen at 20%/50%/remaining fill.
- Sentence embeddings remain excluded because no accepted grouped ablation
  demonstrated an improvement and the local checkpoint is not delivery-bound.
- Large-query cardinality overrides remain rejected after reducing grouped
  Macro-F1 from 0.837508 to 0.835422.
- The report-only fresh retrain received during synchronization is excluded
  from upload because its ignored CSV/model artifacts are absent locally.

## Delivery State

All local engineering work through 17 July is complete: data and sampling are
frozen, two upload candidates pass exact QA, the complete model bundle is hash
verified, official `step1.sh`/`step2.sh`/`step3.sh` entrypoints exist, and
offline model-loading inference is covered by automated tests. Authorized
Kaggle upload and manual final-selection state remain external.

The final full offline run also reconstructed all 3,359,679 predictions from
the serialized models and reproduced the primary candidate byte for byte.
