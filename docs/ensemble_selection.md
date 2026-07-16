# Ensemble Model Selection Report

This report separates held-out model-selection performance from deploy parameters.

| Candidate | Cross-fitted Macro-F1 |
|---|---:|
| LightGBM | 0.836825 |
| XGBoost | 0.836911 |
| Weighted blend | 0.837374 |

## Deploy Parameters

- Selected candidate: `weighted_blend`
- LightGBM weight: `0.4500`
- XGBoost weight: `0.5500`
- Threshold: `0.38757889`
- Candidate positive rate: `18.6204%`

The all-OOF selection score is recorded for reproducibility only; it is not an unbiased validation estimate.
