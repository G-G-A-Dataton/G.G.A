# July 15 Delivery Record

## Accepted Engineering Deliverables

| Area | Acceptance evidence |
|---|---|
| Data QA and final freeze | Five CSVs bound by row count, schema, bytes, and SHA-256 in `configs/final_v1.json` |
| Candidate generation | Exact test-shaped quotas; complete-positive exclusion; deterministic BM25/category/random sources |
| Feature matrix | 23 base, 1 TF-IDF, and 9 global candidate-relative features |
| Validation | Five grouped folds; cross-fitted threshold and ensemble selection |
| Model shortlist | Five LightGBM and five XGBoost models with shared folds and features |
| Model decision | Weighted blend selected only because it beats both single models cross-fitted |
| Submission | Bounded-memory inference, exact QA, atomic publication, delivery manifest |
| Error and feature analysis | Full hash-verified feature importance, threshold, comparison, and fold-external error taxonomy reports |
| Embedding safety | Offline checkpoint/manifests, strict coverage, and no synthetic/zero fallback; not promoted without positive grouped evidence |
| Reproducibility | Exact Python/package pins, clean Git revision, data/artifact hashes, one-command runner |

## Full Production Result

- Training revision: `f22e1e66a1e06879d29905637ecbfe6c0cfc6604`.
- Training data: 17,968 terms, 250,000 positives, 1,627,700 negatives,
  1,877,700 total candidates.
- Negative sources: 316,893 BM25, 814,401 category, 496,406 random.
- Cross-fitted Macro-F1: LightGBM `0.837304`, XGBoost `0.836820`,
  weighted blend `0.837508`.
- Deploy decision: LightGBM `0.65`, XGBoost `0.35`, threshold
  `0.3718157097697258`.
- Training performance: `48:28.11`, peak RSS `3,765,812 KB`, no swap.
- Final CSV: 3,359,679 rows, 645,783 positives (`19.2216%`), exact QA passed.
- Selection/inference: `1:11.70`, peak RSS `701,084 KB`.
- Current regression/integration suite: 102/102 passed.

## Integrity Chain

| Artifact | SHA-256 |
|---|---|
| Final CSV | `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c` |
| Delivery manifest | `972c0fbada3be885d993e13010c501c338ba86b8d8e62eccc8d3d51423c4a94e` |
| OOF manifest | `fdb9582e4cef39572059b7731b44e473d78adbcfe75d08398c8c39e3258f9eb9` |
| Ensemble decision | `7aab740498895f6b60df182cef5c2401c9b0173a152049224b7159395cdfccc4` |

Generated current reports are `ensemble_selection.md`,
`ensemble_comparison.md`, `threshold_analysis.md`, `feature_importance.md`, and
`error_taxonomy.md`. Older similarly named Turkish reports are retained only as
explicitly invalidated historical records.

The clean-environment acceptance performed on 16 July is recorded separately
in [`reproducibility_dry_run.md`](reproducibility_dry_run.md); it reproduced the
final CSV byte for byte without changing this delivery decision.

## Evidence Boundary

The repository contains a complete local 15 July candidate, not a Kaggle
leaderboard observation. Upload and public/private scores require the authorized
team account. Full sentence embeddings are optional and must not be described
as part of this model unless a real checkpoint, complete matrices, and a
positive grouped ablation are later archived.
