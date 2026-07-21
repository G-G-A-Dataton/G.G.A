# July 1-15 Task Audit

This is the authoritative mapping from every team-member task through 15 July
2026 to repository evidence. Historical sprint documents preserve their
original claims but do not override this audit.

Muhammed's subsequent 16 July clean-environment task is tracked separately in
[`july_16_task_audit.md`](july_16_task_audit.md); the historical counts below
remain scoped to 1-15 July.

## Progress

- Repository-verifiable or formally closed plan cells: **58/60 (96.7%)**.
- Local engineering acceptance: **100%**.
- External team-account actions: **2/60** (`Kaggle team/access confirmation`
  and `Kaggle upload/public score record`). These cannot be inferred locally.
- Full sentence embeddings are a conditional experiment, not an accepted model
  dependency. The v1 batch/cosine/checkpoint contract is complete; promotion
  requires a real local checkpoint and a positive grouped ablation.

## Day-By-Day Evidence

| Day | Status | Team-member deliverables and evidence |
|---|---|---|
| 1 Jul | Local complete; one external check | Ömer: Kaggle team/access is external. Ahmet: `EDA_notlari_v0.md`. Mustafa: validated merge/data-loading contract. Muhammed: `REPO_CALISMA_STANDARDI.md`, repository layout, and experiment log. |
| 2 Jul | Complete | Ömer: Macro-F1 and grouped validation contracts in `src/metrics.py`. Ahmet: `EDA_on_raporu.md`. Mustafa: duplicate/reference/schema checks in the data verifier. Muhammed: exact submission validator and tests. |
| 3 Jul | Complete | Ömer: LightGBM baseline path. Ahmet: lexical/brand/category features. Mustafa: deterministic random negative sampling. Muhammed: TF-IDF cosine PoC and production module. |
| 4 Jul | Complete | Ömer: model training upgraded to grouped folds. Ahmet: demographic signals. Mustafa: complete-positive exclusion and leakage assertions. Muhammed: TF-IDF integrated into the fixed feature contract. |
| 5 Jul | Local complete; upload external | Ömer: locally QA-approved submission workflow; Kaggle upload needs account access. Ahmet: `sprint1_raporu.md`. Mustafa: seeded sampling. Muhammed: atomic submission QA gate. |
| 6 Jul | Complete | Ömer: error-analysis module. Ahmet: L1/L2/L3 category features. Mustafa: compact two-pass BM25 index. Muhammed: TF-IDF parameter tooling and persisted vectorizer. |
| 7 Jul | Complete | Ömer: BM25 candidate-shift comparison. Ahmet: attribute signal notes. Mustafa: BM25 hard-negative source with full exclusion. Muhammed: disk-backed full-submission feature strategy. |
| 8 Jul | Complete at v1/PoC scope | Ömer: tuned tree configurations. Ahmet: `attribute_dogrulama_notlari.md`. Mustafa: flat catalog attribute parser and tests. Muhammed: strict offline embedding batch/checkpoint implementation; no synthetic fallback. |
| 9 Jul | Complete | Ömer: LightGBM/XGBoost comparison. Ahmet: color/material/model conflict features. Mustafa: deterministic BM25/category/random mix. Muhammed: canonical item-text builder. |
| 10 Jul | Complete; embedding promotion conditional | Ömer: full five-fold `feature_importance.md`. Ahmet: technical/method drafts. Mustafa: sampling-distribution study supersedes fixed-ratio datasets. Muhammed: production batch embedding runner and manifests; full matrix is not promoted without a local checkpoint. |
| 11 Jul | Complete | Ömer: test-shaped candidate contract supersedes the legacy ratio matrix. Ahmet: threshold diagnostics. Mustafa: parameterized manifest-backed data builder. Muhammed: term embedding runner/checkpoint contract. |
| 12 Jul | Complete; embedding promotion rejected pending evidence | Ömer: embedding comparison path refuses synthetic evidence. Ahmet: full fold-external `error_taxonomy.md`. Mustafa: frozen-data/pipeline QA. Muhammed: strict cosine feature and coverage validation. |
| 13 Jul | Complete | Ömer: full LGBM/XGBoost/blend candidate table. Ahmet: current `rapor_yontem_v1.md`. Mustafa: full bounded-memory feature/inference path. Muhammed: `offline_dependency.md` and environment verifier. |
| 14 Jul | Complete | Ömer: hash-verified full OOF shortlist. Ahmet: leakage-free `threshold_analysis.md`. Mustafa: measured full runtime/RSS with out-of-core inference. Muhammed: canonical one-command `RUNBOOK.md`. |
| 15 Jul | Local complete; leaderboard external | Ömer: joint ensemble/threshold decision. Ahmet: local/leaderboard risk report without fabricated public score. Mustafa: five-file SHA-256 data freeze. Muhammed: exact requirements, clean-revision artifact lineage, and delivery manifest. |

## Accepted 15 July Decision

| Field | Accepted value |
|---|---|
| Training terms / candidates | 17,968 / 1,877,700 |
| Candidate sources | 250,000 positive; 316,893 BM25; 814,401 category; 496,406 random |
| Validation | 5-fold `StratifiedGroupKFold`, grouped by `term_id` |
| Models | 5 LightGBM + 5 XGBoost |
| Selected candidate | 65% LightGBM + 35% XGBoost |
| Cross-fitted Macro-F1 | 0.837508 |
| Deploy threshold | 0.3718157097697258 |
| Final local CSV | 3,359,679 rows; 645,783 positives; full QA passed |

## Remaining External Action

Upload the hash-approved `outputs/submission_v2.csv` through the authorized team
account and record the observed public score. No local script or document can
truthfully complete that account-bound action.
