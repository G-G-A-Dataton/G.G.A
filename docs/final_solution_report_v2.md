# G.G.A Final Solution Report v2

- **Report date:** 16 July 2026
- **Competition stage:** TEKNOFEST 2026 E-Commerce Datathon
- **Metric:** Macro-F1
- **Evidence boundary:** grouped local validation; no Kaggle score is claimed

## 1. Executive Summary

G.G.A models relevance for each `(term_id, item_id)` pair as binary
classification. The central challenge is that the 250,000 labeled training
pairs are positive only. The accepted solution therefore builds deterministic,
test-shaped negative candidates, extracts lexical and candidate-relative
features, and trains five LightGBM plus five XGBoost grouped folds.

The selected 65% LightGBM / 35% XGBoost blend reaches cross-fitted grouped
Macro-F1 `0.837508`, ahead of LightGBM (`0.837304`) and XGBoost (`0.836820`).
Its deploy threshold is `0.3718157097697258`. The final 3,359,679-row candidate
contains 645,783 positive predictions and passes exact submission QA.

Two final candidates are retained. The blend is the default because it has the
best leakage-free local result. LightGBM is the runner-up and model-family
fallback. Their predictions differ on 21,647 rows (`0.6443%`). Neither choice
uses leaderboard feedback.

## 2. Data Contract

| Dataset | Rows | Purpose |
|---|---:|---|
| `terms.csv` | 50,153 | normalized query text by `term_id` |
| `items.csv` | 962,873 | catalog text, hierarchy, brand, demographics, attributes |
| `training_pairs.csv` | 250,000 | known positive query-product pairs |
| `submission_pairs.csv` | 3,359,679 | pairs requiring binary predictions |
| `sample_submission.csv` | 3,359,679 | exact output IDs and order |

`configs/final_v1.json` freezes all five files by schema, byte size, row count,
and SHA-256. Every production consumer recomputes the relevant hashes before
using model or prediction artifacts.

## 3. Candidate Generation

Training terms receive the same candidate-density rule observed in the test
set: `max(100, ceil(2 * known positives))`. Every known positive is retained.
The remaining quota is filled in deterministic order from:

1. 20% compact-BM25 hard negatives;
2. 50% products from the positive products' normalized L2 categories;
3. catalog-random products for the remaining quota.

All 250,000 known positives are excluded from every negative source, including
positives outside a sampled experiment. The accepted full matrix contains
1,877,700 rows across 17,968 complete training terms:

| Source | Rows |
|---|---:|
| Positive | 250,000 |
| BM25 hard negative | 316,893 |
| Category hard negative | 814,401 |
| Catalog-random negative | 496,406 |

The 20% BM25 fraction first minimized an unlabeled train/test feature-distance
diagnostic and was promoted only after grouped OOF validation. It is not based
on hidden labels or leaderboard probing.

## 4. Feature Contract

The fixed model matrix contains 33 ordered features:

- 23 base features covering normalized lexical overlap and coverage, phrase
  evidence, model-code match/conflict, exact brand evidence, text length,
  category hierarchy, demographics, and parsed color/size/material signals;
- one TF-IDF cosine feature from a 10,000-term unigram vocabulary with
  `min_df=2`;
- nine per-query context features covering candidate counts, ranks, gaps, and
  distance from the candidate-group mean.

The five-fold LightGBM gain summary is descriptive, not causal. The largest
shares are `tfidf_cosine_delta_mean` (46.17%), `candidate_count_log1p` (8.45%),
`query_title_overlap_rank` (6.95%), and `tfidf_cosine_rank` (5.32%). A feature
is removed only after grouped ablation; zero gain alone is insufficient.

Sentence embeddings are not part of the accepted matrix. The code requires a
real local checkpoint, complete ID coverage, and hash manifests, and it never
substitutes zero or synthetic vectors. Promotion remains conditional on a
positive grouped ablation.

## 5. Validation And Selection

All rows for one `term_id` remain in exactly one of five
`StratifiedGroupKFold` folds (`seed=42`). LightGBM and XGBoost share the same
fold IDs and feature matrix.

Weights and thresholds are evaluated cross-fitted: for each held-out fold, the
parameters are selected using only the other four folds. The final deploy
threshold is then fitted on all OOF rows and is recorded as a deployment
parameter, not an unbiased validation estimate.

| Candidate | Cross-fitted Macro-F1 | Deploy threshold |
|---|---:|---:|
| Weighted blend, LGB/XGB 0.65/0.35 | **0.837508** | 0.37181571 |
| LightGBM | 0.837304 | 0.38338208 |
| XGBoost | 0.836820 | 0.38769966 |

The full run used 3,000-round ceilings and 200-round early stopping. Every one
of the ten models stopped naturally between iterations 1,901 and 2,526.
Training took 48:28.11 with peak RSS 3,765,812 KB and no swap.

## 6. Final Candidate Set

| Rank | Strategy | Rows | Positives | Positive rate | SHA-256 |
|---:|---|---:|---:|---:|---|
| 1 | 65% LGB / 35% XGB blend | 3,359,679 | 645,783 | 19.2216% | `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c` |
| 2 | LightGBM | 3,359,679 | 634,316 | 18.8803% | `c9407c965ad7ba047441aa44952dab4681a13e23d32fb494b1c31545c83052e7` |

Candidate 1 is byte-identical to the accepted 15 July delivery. Candidate 2 is
retained because it is the strongest single-family model and has independently
selected cross-fitted/deploy thresholds. It is a fallback, not a claim of
superior hidden-test performance.

Both files pass exact `id,prediction` order, 3,359,679 rows, global unique IDs,
integer `{0,1}` predictions, null rejection, accidental-index rejection, and
SHA-256 recording. See [`submission_qa_approval.md`](submission_qa_approval.md).

## 7. Error And Risk Analysis

Fold-external predictions contain 68,740 false positives and 71,467 false
negatives. The dominant observable false-positive signal is lexical decoy
evidence (45,185 rows); 9,105 false negatives have no lexical evidence. These
labels describe feature evidence and are not asserted as root causes.

Residual risks remain:

- synthetic negative relevance can differ from the hidden-label policy;
- final-candidate positive rates differ from training-candidate prevalence;
- very close model scores do not establish hidden-set ordering;
- attributes are sparse and `query_size_match` has zero LightGBM gain;
- the repository has no authorized Kaggle public/private score observation.

A large-group cardinality override was rejected because cross-fitted Macro-F1
fell from `0.837508` to `0.835422`. No leaderboard-only heuristic enters the
accepted decision.

## 8. Reproducibility And Integrity

The accepted environment is Python 3.13.5 with 158 transitive packages pinned
and hashed in `requirements.lock`. Model, TF-IDF, OOF, decision, source-data,
and submission artifacts are protected by versioned SHA-256 manifests.

The 16 July clean-environment dry-run used a detached clone, disabled Python
user-site and online model access, passed all then-current tests and data gates,
and rebuilt Candidate 1 byte for byte. The current suite contains 108 tests,
including a contract that keeps `configs/final_v1.json` synchronized with the
accepted candidate sampler.

Canonical commands:

```bash
python scripts/run_production.py --stage verify
python scripts/run_production.py --stage train
python scripts/run_production.py --stage predict
python scripts/submission/run_final_candidate_set.py
```

Detailed evidence is available in [`acceptance_runs.md`](acceptance_runs.md),
[`reproducibility_dry_run.md`](reproducibility_dry_run.md),
[`final_submission_candidates.md`](final_submission_candidates.md), and
[`experiment_log.md`](experiment_log.md).

## 9. Evidence Boundary And Final Decision

The repository provides a complete, reproducible local solution and two
hash-approved upload candidates. It does not prove a Kaggle score. Authorized
team-account upload, public score recording, and final private leaderboard
observation remain external actions and must be recorded only after they occur.

- **Default upload:** Candidate 1, weighted blend.
- **Controlled fallback:** Candidate 2, LightGBM.
- **Reason:** ordered cross-fitted grouped Macro-F1 with no leaderboard probing.
