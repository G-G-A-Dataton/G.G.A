# Solution Method v1

> This historical method snapshot is superseded by
> [`final_solution_report_v2.md`](final_solution_report_v2.md) for the complete
> 16 July solution, candidate strategy, QA, and reproducibility evidence.

**Contract date:** 15 July 2026
**Scope:** final candidate generation, features, validation, model selection, and inference

## 1. Problem And Data

The task is binary relevance classification for `(term_id, item_id)` pairs and is evaluated with Macro-F1. `training_pairs.csv` contains 250,000 positive pairs across 17,968 training terms; no labeled negatives are provided.

All five competition CSV files are frozen in `configs/final_v1.json` by byte size, ordered schema, row count, and SHA-256. Training and prediction artifacts also record the source hashes and consumers recompute them before use.

## 2. Candidate Generation

The submission distribution is highly asymmetric: each test term has approximately `max(100, 2 * known-positive-count)` candidate rows. Training therefore uses the same per-term quota instead of a global fixed negative ratio.

For each complete training term:

1. Keep every known positive.
2. Set the target candidate count to `max(100, ceil(2 * positives))`.
3. Fill 20% of the negative quota from the compact BM25 index.
4. Fill 50% from catalog products sharing the positive products' normalized L2 category.
5. Fill the remainder from deterministic catalog-random candidates.
6. Exclude every known positive pair, including positives outside an experiment sample.
7. Assert exact per-term quotas, source counts, uniqueness, and reproducibility.

The 20% BM25 share was selected by an unlabeled candidate-distribution shift
study and then accepted only after grouped OOF validation. It is not justified
by semantic intuition alone.

## 3. Feature Contract

The production matrix contains 33 numeric features in a fixed order:

- 23 base features: normalized title/category overlap and coverage, phrase match, model-code match/conflict, exact brand match, text lengths, category hierarchy, demographic conflict, and parsed color/size/material signals.
- 1 TF-IDF cosine feature: 10,000 unigram vocabulary with `min_df=2`, fitted once and stored with the models.
- 9 candidate-relative context features: per-term ranks, top-score gaps, mean deltas, and candidate-count scale.

Unicode normalization and complete-token matching prevent substring false positives. Context features are computed over complete term groups. Because submission terms are not contiguous, production test context is generated in two disk-backed passes rather than assuming chunk-local groups.

Sentence embeddings are optional. Only hash-manifested matrices with exact model, dimension, source, ID coverage, and row-order metadata are accepted. Missing embeddings are never replaced by zeros or synthetic values.

## 4. Validation And Thresholds

All candidates for one `term_id` remain in exactly one of five `StratifiedGroupKFold` folds (`seed=42`). LightGBM and XGBoost share the same fold IDs and feature matrix.

Threshold reporting is cross-fitted. For each evaluated fold, the threshold is selected using only the other four OOF folds. The resulting concatenated predictions produce the model-selection Macro-F1. The final deploy threshold is then selected on all OOF rows and is explicitly recorded as a deployment parameter, not as an unbiased validation score.

Blend weight selection follows the same rule: each validation fold receives a weight and threshold selected outside that fold. The weighted blend is promoted only if its cross-fitted Macro-F1 beats both single models.

## 5. Models And Selection

The final shortlist contains five deterministic LightGBM folds and five XGBoost histogram-tree folds. Selection candidates are:

- LightGBM mean probability
- XGBoost mean probability
- weighted LightGBM/XGBoost probability

`scripts/analysis/run_ensemble_optimization.py` evaluates all three through the shared contract in `src/modeling.py`, selects the highest cross-fitted score, and records weights and threshold in `ensemble_decision.json`.

## 6. Reproducibility And Inference

Production training refuses to create versioned artifacts from a dirty Git worktree. The OOF manifest binds the clean Git revision, data hashes, feature/candidate schema versions, fold contract, row counts, model files, TF-IDF vectorizer, OOF arrays, test arrays, and SHA-256 of every artifact.

The selected test probabilities are streamed into `outputs/submission_v2.csv`. Publication is atomic and occurs only after verifying:

- exactly 3,359,679 rows,
- exact `id,prediction` column order,
- exact sample-submission ID order,
- unique IDs across chunks,
- integer predictions in `{0, 1}`,
- no accidental index column.

The canonical commands are documented in `RUNBOOK.md`; pinned packages and local-only optional embedding dependencies are documented in `docs/offline_dependency.md`.

## 7. Accepted Evidence Boundary

The full run trained on 1,877,700 candidates and generated all 3,359,679 test
predictions. Cross-fitted Macro-F1 was 0.837304 for LightGBM, 0.836820 for
XGBoost, and 0.837508 for the weighted blend. The selected deploy rule uses
65% LightGBM, 35% XGBoost, and threshold 0.37181571.

This is valid grouped local evidence, not a leaderboard projection. The final
CSV and its decision/OOF lineage are hash-bound by the delivery manifest. A
Kaggle score remains external evidence. Full sentence embeddings remain a
conditional experiment and cannot enter the feature contract without a real
local checkpoint, complete manifests, and a positive grouped ablation.
