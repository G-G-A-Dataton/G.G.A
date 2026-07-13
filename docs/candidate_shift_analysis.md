# Candidate Distribution Shift

This analysis uses no submission labels. It compares retrieval-sensitive feature marginals from deterministic training candidates with unseen submission-term candidates.

- Sampled training terms: `1,000`
- Sampled submission terms: `1,000`
- Fixed category-hard fraction: `0.50`
- Selected BM25 fraction by mean normalized quantile distance: `0.20`

| BM25 fraction | Mean distance | Max distance | BM25 rows | Category rows | Random rows |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 0.094050 | 0.146292 | 17,376 | 44,844 | 27,408 |
| 0.15 | 0.101211 | 0.114118 | 13,110 | 44,844 | 31,674 |
| 0.25 | 0.126410 | 0.217622 | 21,601 | 44,844 | 23,183 |
| 0.10 | 0.137726 | 0.168319 | 8,795 | 44,844 | 35,989 |
| 0.30 | 0.192488 | 0.296342 | 25,857 | 44,844 | 18,927 |
| 0.00 | 0.282542 | 0.339182 | 0 | 44,844 | 44,784 |

The result is a covariate-shift diagnostic, not a label-based model score. Final promotion also requires grouped OOF behavior and submission sanity checks.
