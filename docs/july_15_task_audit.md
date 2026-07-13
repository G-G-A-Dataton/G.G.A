# July 13-15 Task Audit

This audit maps the team plan to executable evidence as of 13 July 2026.

| Date / owner deliverable | Status | Canonical evidence |
|---|---|---|
| 13 Jul: LGBM/XGBoost/ensemble candidate table | Code complete, medium run passed | `run_model_shortlist.py`, `run_ensemble_comparison.py`, `acceptance_runs.md` |
| 13 Jul: report method section | Complete | `docs/rapor_yontem_v1.md` |
| 13 Jul: submission feature dry-run | Complete at 50K rows | out-of-core stores and medium QA run |
| 13 Jul: offline dependency list | Complete | `docs/offline_dependency.md`, exact environment verifier |
| 14 Jul: model shortlist and OOF export | Code complete, medium run passed | ten models, OOF labels/folds, test arrays, hash manifest |
| 14 Jul: threshold scan | Complete without fold leakage | `run_threshold_analysis.py`, shared cross-fitted selection |
| 14 Jul: memory/time risk measurement | Complete at medium scale | 2:08.17 wall time, 1.46 GB peak RSS |
| 14 Jul: one-command runbook | Complete | `scripts/run_production.py`, `RUNBOOK.md` |
| 15 Jul: joint ensemble/threshold optimization | Complete | `run_ensemble_optimization.py`; single models and blend compete on cross-fitted Macro-F1 |
| 15 Jul: local/public score interpretation | Local methodology complete; external score absent | local score boundary documented; no Kaggle score is fabricated |
| 15 Jul: final training data freeze | Complete | `configs/final_v1.json`, five SHA-256 hashes |
| 15 Jul: environment/reproducibility notes | Complete | `.python-version`, exact requirements, offline guide, clean-worktree artifact rule |

## Remaining Runtime Or External Actions

1. Execute the full default production run and archive `oof_manifest.json`, `ensemble_decision.json`, runtime/RSS, and `submission_v2.csv` QA result.
2. Upload approved candidates to Kaggle and record public scores. This requires team account access and cannot be inferred locally.
3. Generate full sentence embeddings only if a local model checkpoint and suitable compute are available; promote them only after a positive grouped full-data ablation.

All historical scores produced by row-level CV, partial positive sampling, fixed negative ratios unlike test, synthetic embeddings, or same-OOF threshold reporting remain invalid for model selection.
