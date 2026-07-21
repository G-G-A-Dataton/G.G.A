# Ensemble Candidate Comparison

Candidate selection uses cross-fitted Macro-F1. All-OOF selection scores and deploy thresholds are included only for reproducibility.

Artifact mode: `full` training / `full` test.

| Candidate | Selected | Cross-fitted Macro-F1 | All-OOF selection score | LGB weight | XGB weight | Deploy threshold |
|---|---|---:|---:|---:|---:|---:|
| LightGBM | no | 0.837304 | 0.837345 | 1.0000 | 0.0000 | 0.38338208 |
| XGBoost | no | 0.836820 | 0.836960 | 0.0000 | 1.0000 | 0.38769966 |
| Weighted blend | yes | 0.837508 | 0.837702 | 0.6500 | 0.3500 | 0.37181571 |

The selected row is the only candidate eligible for production thresholding.
