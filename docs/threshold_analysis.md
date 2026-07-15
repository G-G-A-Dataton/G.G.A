# Threshold Analysis

The primary score uses fold-specific thresholds selected without the evaluated fold. The deploy threshold is then fitted on all OOF rows and is not reported as an unbiased validation score.

## Validation Result

- Artifact mode: `full` training / `full` test
- Selected candidate: `weighted_blend`
- Cross-fitted Macro-F1: `0.837508`
- Fold count: `5` grouped by `term_id`

## Deploy Parameters

- LightGBM weight: `0.6500`
- XGBoost weight: `0.3500`
- Threshold: `0.37181571`
- All-OOF diagnostic Macro-F1 at deploy threshold: `0.837701`
- All-OOF diagnostic Macro-F1 at 0.5: `0.830207`

## Diagnostic Curve

| Threshold | Macro-F1 | F1 positive | F1 negative | Precision | Recall | Positive rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.33000000 | 0.836042 | 0.717615 | 0.954470 | 0.689286 | 0.748372 | 14.4555% |
| 0.34000000 | 0.836522 | 0.717959 | 0.955086 | 0.696516 | 0.740764 | 14.1600% |
| 0.35000000 | 0.836965 | 0.718259 | 0.955671 | 0.703726 | 0.733404 | 13.8756% |
| 0.36000000 | 0.837341 | 0.718470 | 0.956211 | 0.710754 | 0.726356 | 13.6064% |
| 0.37000000 | 0.837646 | 0.718580 | 0.956712 | 0.717658 | 0.719504 | 13.3484% |
| 0.37181571 | 0.837701 | 0.718600 | 0.956802 | 0.718928 | 0.718272 | 13.3020% |
| 0.38000000 | 0.837568 | 0.718023 | 0.957113 | 0.724206 | 0.711944 | 13.0887% |
| 0.39000000 | 0.837470 | 0.717461 | 0.957478 | 0.730473 | 0.704904 | 12.8481% |
| 0.40000000 | 0.837339 | 0.716836 | 0.957842 | 0.737037 | 0.697712 | 12.6038% |
| 0.41000000 | 0.837165 | 0.716152 | 0.958178 | 0.743505 | 0.690740 | 12.3693% |
| 0.42000000 | 0.836796 | 0.715136 | 0.958456 | 0.749680 | 0.683636 | 12.1412% |

The complete descriptive curve is stored in `outputs/threshold_analysis.csv`.
