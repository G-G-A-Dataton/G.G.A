# G.G.A Experiment Register

This register separates decision-grade experiments from historical development
runs. Scores are comparable only when they use the accepted test-shaped
candidate distribution, complete-positive exclusion, grouped validation, and
cross-fitted parameter selection.

## Decision-Grade Evidence

| ID | Date | Scope | Validation | Result | Decision |
|---|---|---|---|---|---|
| EXP-010 | 15 Jul | Full LightGBM/XGBoost shortlist | 5-fold grouped, cross-fitted | Blend 0.837508; LGB 0.837304; XGB 0.836820 | Accept 0.65/0.35 blend at threshold 0.37181571 |
| ACC-011 | 16 Jul | Clean-environment delivery reproduction | 102 tests plus data/artifact/delivery gates | Final CSV reproduced byte for byte | Reproducibility accepted |
| REL-012 | 16 Jul | Top-two final candidate packaging | Full OOF ranking and exact QA | 2/2 candidates passed; 21,647-row disagreement | Blend primary; LightGBM fallback |

Kaggle public/private scores are deliberately blank because the repository has
no authorized account observation.

## EXP-010: Accepted Full Production Run

| Field | Accepted value |
|---|---|
| Artifact revision | `f22e1e66a1e06879d29905637ecbfe6c0cfc6604` |
| Complete terms | 17,968 |
| Positive / negative candidates | 250,000 / 1,627,700 |
| Candidate sources | 316,893 BM25; 814,401 category; 496,406 random |
| Validation | 5-fold `StratifiedGroupKFold`, group=`term_id` |
| Models | 5 LightGBM + 5 XGBoost |
| Features | 33 ordered production features |
| Cross-fitted scores | blend 0.837508; LGB 0.837304; XGB 0.836820 |
| Deploy rule | LGB 0.65; XGB 0.35; threshold 0.3718157097697258 |
| Submission | 3,359,679 rows; 645,783 positives; exact QA passed |
| Runtime | training 48:28.11; selection/inference 1:11.70 |

## REL-012: Final Candidate Set

| Rank | Candidate | Cross-fitted Macro-F1 | Threshold | Positives | SHA-256 |
|---:|---|---:|---:|---:|---|
| 1 | LGB/XGB 0.65/0.35 blend | 0.837508 | 0.37181571 | 645,783 | `2ecfcb051291582e025f303a9e1e16c985c297b0c4ec8cf15f47716892e7fe4c` |
| 2 | LightGBM | 0.837304 | 0.38338208 | 634,316 | `c9407c965ad7ba047441aa44952dab4681a13e23d32fb494b1c31545c83052e7` |

The candidates disagree on 21,647 rows (`0.6443%`). The runner-up is retained
for model-family diversity; the local evidence still selects Candidate 1.

## Supporting Decision Records

| Record | Evidence | Outcome |
|---|---|---|
| Candidate-distribution study | 1,000 training and 1,000 submission terms; unlabeled normalized-quantile distance | 20% BM25 fraction selected, then validated grouped |
| Threshold study | Fold-specific selection outside each evaluated fold | Deploy threshold 0.37181571 fitted on all OOF only after validation |
| Large-group override | Cross-fitted Macro-F1 0.835422 versus 0.837508 baseline | Rejected |
| Sentence embeddings | Production-safe code exists; no complete local checkpoint/manifests and positive grouped ablation | Not promoted |
| Error taxonomy | 68,740 FP and 71,467 FN from fold-external predictions | Lexical decoys and no-lexical-evidence cases prioritized |

## Historical Development Runs

EXP-001 through EXP-009 are retained as engineering history only. At least one
accepted contract was absent: query-group isolation, complete-positive
exclusion, flat-catalog attribute parsing, test-shaped negative quotas, or
cross-fitted thresholds. Their values must not be compared with EXP-010.

| ID | Historical scope | Recorded value | Status / limitation |
|---|---|---|---|
| EXP-001 | LightGBM random-negative baseline | 0.9613 row-level CV | Invalidated: row-level split and obsolete candidate distribution |
| EXP-002 | Demographic additions | no isolated accepted score | Closed without a decision-grade result |
| EXP-003 | TF-IDF development run | 0.9699 row-level CV | Invalidated validation; TF-IDF retained through EXP-010 evidence |
| EXP-004 | Category hierarchy additions | no isolated accepted score | Closed without a decision-grade result |
| EXP-005 | Early hard-negative baseline | 0.9622 row-level CV | Invalidated validation and negative contract |
| EXP-006 | Planned BM25 comparison | no accepted full result under that ID | Superseded by distribution study and EXP-010 |
| EXP-007 | Planned mixed full run | no accepted result under that ID | Superseded by EXP-010 |
| EXP-008 | Early tuning grid | 0.9631 row-level diagnostic | Invalidated validation and feature contract |
| EXP-009 | Early LGB/XGB comparison | LGB 0.9613; XGB 0.9597 | Invalidated validation; model families re-evaluated in EXP-010 |

## Registration Rules

Every new decision-grade row must record:

- immutable experiment ID, date, owner, and clean Git revision;
- data/candidate/feature schema versions and hashes;
- group-aware validation and leakage controls;
- cross-fitted score plus separately labeled deploy parameters;
- artifact paths and SHA-256;
- measured runtime/resources and an explicit accept/reject decision;
- leaderboard value only when directly observed through the authorized account.

The accepted integrity chain is detailed in
[`july_15_delivery.md`](july_15_delivery.md). Current model status and the full
v2 narrative are in [`model_status.md`](model_status.md) and
[`final_solution_report_v2.md`](final_solution_report_v2.md).
