# Ensemble Model Selection Report

This report separates held-out model-selection performance from deploy parameters.

| Candidate | Cross-fitted Macro-F1 |
|---|---:|
| LightGBM | 0.837304 |
| XGBoost | 0.836820 |
| Weighted blend | 0.837508 |

## Deploy Parameters

- Selected candidate: `weighted_blend`
- LightGBM weight: `0.6500`
- XGBoost weight: `0.3500`
- Threshold: `0.37181571`
- Candidate positive rate: `19.2216%`

The all-OOF selection score is recorded for reproducibility only; it is not an unbiased validation estimate.
