# Solution Method v1

**Contract date:** 13 July 2026
**Scope:** final candidate generation, features, validation, model selection, and inference

## 1. Problem And Data

The task is binary relevance classification for `(term_id, item_id)` pairs and is evaluated with Macro-F1. `training_pairs.csv` contains 250,000 positive pairs across 17,968 training terms; no labeled negatives are provided.

All five competition CSV files are frozen in `configs/final_v1.json` by byte size, ordered schema, row count, and SHA-256. Training and prediction artifacts also record the source hashes and consumers recompute them before use.

## 2. Candidate Generation

The submission distribution is highly asymmetric: each test term has approximately `max(100, 2 * known-positive-count)` candidate rows. Training therefore uses the same per-term quota instead of a global fixed negative ratio.

For each complete training term:

1. Keep every known positive.
2. Set the target candidate count to `max(100, ceil(2 * positives))`.
3. Fill half of the negative quota from catalog products sharing the positive products' normalized L2 category.
4. Fill the remainder from deterministic catalog-random candidates.
5. Exclude every known positive pair, including positives outside an experiment sample.
6. Assert exact per-term quotas, uniqueness, and reproducibility.

BM25 remains an ablation source. Its compact two-pass index is suitable for hard-negative experiments, but BM25 is not promoted merely because it is semantically closer; promotion requires a grouped comparison.

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

## 7. Current Evidence Boundary

The 300-term acceptance run produced 33,048 training candidates, ten model folds, and 50,000 QA-verified test predictions in 2 minutes 8 seconds at approximately 1.46 GB peak RSS. Cross-fitted Macro-F1 was 0.939451 for LightGBM, 0.937968 for XGBoost, and 0.937536 for the blend; LightGBM was correctly selected at deploy threshold 0.41054133.

This run validates behavior and scale, not leaderboard quality. Full 1,877,700-row training, full 3,359,679-row prediction, optional full embedding ablation, and Kaggle scores must be recorded separately when those runtime jobs complete.
