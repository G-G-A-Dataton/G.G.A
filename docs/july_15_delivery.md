# July 15 Delivery Record

## Completed Engineering Deliverables

| Area | Acceptance evidence |
|---|---|
| Data QA and final freeze | Five CSVs bound by row count, column order, bytes, and SHA-256 in `configs/final_v1.json` |
| Negative generation | Exact per-query test-shaped quotas; complete positive exclusion; deterministic category/random sources |
| Feature matrix | Unicode-safe exact matching, model-token signals, parsed attributes, TF-IDF, and global context ranks/gaps |
| Validation | Five grouped folds; cross-fitted threshold and ensemble reports |
| Model shortlist | Shared LightGBM/XGBoost OOF and out-of-core test prediction pipeline |
| Ensemble decision | Promote the best held-out candidate; blend only when it beats both single models |
| Submission | Disk-backed global features, bounded-memory inference, exact QA, atomic publication |
| Embeddings | Verified checkpoints/manifests, memmap assembly, strict coverage, no synthetic fallback |
| BM25 | Two-pass document-frequency build and compact NumPy posting lists |
| Offline reproducibility | Exact Python/package pins, environment verifier, local-only optional embedding model |
| One-command operation | `scripts/run_production.py` and canonical `RUNBOOK.md` |

## Measured Acceptance Runs

- 77 regression/integration tests passed.
- Full data freeze verification passed for 5 files and 3,359,679 submission rows.
- Environment verification passed for Python 3.13.5 and 17 pinned packages.
- Canonical 20-term training smoke generated 2,396 candidates and five model folds.
- Ten-term dual-model shortlist generated all 10 models, OOF/fold arrays, 200 predictions, and a QA-passing sample submission.
- Compact BM25 indexed 20,000 real products in 2.71 seconds at about 140 MB peak RSS.

## Explicitly Pending Runtime Evidence

The full 1,877,700-row production training, full 3,359,679-row inference, optional 962,873-item sentence embeddings, and Kaggle leaderboard score are runtime jobs, not completed evidence in this repository state. They must not be represented as completed until their manifests and logs exist. Historical reports remain for provenance but are invalid for current model selection.
