# Archived preliminary ablation — not a valid model comparison

This report is retained only for auditability. Its Dense, Hybrid, Reranker, and
Calibration scores were synthesized from evaluation labels rather than emitted
by the real model stages. Do not use those numbers for release or architecture
decisions.

The current ablation runner accepts only human-verified labels and materialized
stage outputs: `bm25_score`, `dense_score`, `rrf_score`, `linear_score`,
`rerank_score`, and `calibrated_score`.
