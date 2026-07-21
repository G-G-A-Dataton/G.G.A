# Archived preliminary benchmark — not a valid model evaluation

This report is retained only for auditability. It was generated before the
evaluation contract was enforced:

- The golden set contains unverified BM25-negative candidates.
- The benchmark synthesized reranker scores from labels instead of invoking a
  trained cross-encoder.
- The reported calibration was not Platt calibration on an independent split.

Therefore its ranking, calibration, throughput, and "100% validation" claims
must not be used for model selection, release approval, or comparison.

The replacement benchmark requires
`datasets/golden_testset_verified_v1.parquet`, human-verified labels, and real
`bm25_score`, `dense_score`, `hybrid_score`, `rerank_score`, and
`calibrated_score` columns.
