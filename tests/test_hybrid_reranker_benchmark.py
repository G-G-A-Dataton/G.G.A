"""
tests/test_hybrid_reranker_benchmark.py
=========================================
G.G.A Takımı — E2E Benchmark Runner Unit Tests
"""

import os
import pandas as pd
import pytest
from scripts.training.run_hybrid_reranker_benchmark import run_benchmark, write_benchmark_report


def test_run_benchmark_with_dummy_dataset(tmp_path):
    dataset_file = tmp_path / "dummy_golden.csv"
    df = pd.DataFrame({
        "term_id": ["q1", "q1", "q2", "q2"],
        "item_id": ["i1", "i2", "i3", "i4"],
        "bm25_score": [0.9, 0.1, 0.8, 0.2],
        "dense_score": [0.8, 0.2, 0.9, 0.1],
        "hybrid_score": [0.9, 0.1, 0.9, 0.1],
        "rerank_score": [0.95, 0.05, 0.95, 0.05],
        "calibrated_score": [0.9, 0.1, 0.9, 0.1],
        "label": [1, 0, 1, 0],
        "label_status": ["verified_positive", "verified_negative"] * 2,
        "bm25_rank": [1, 2, 1, 2],
        "query_text": ["siyah bot", "siyah bot", "elbise", "elbise"],
        "item_title": ["Bot 1", "Bot 2", "Elbise 1", "Elbise 2"],
    })
    df.to_csv(dataset_file, index=False)

    results = run_benchmark(
        config_path="configs/benchmark_e2e.yaml",
        dataset_override=str(dataset_file),
    )

    assert "metrics" in results
    metrics = results["metrics"]
    assert "retrieval_recall@10" in metrics
    assert "rerank_ndcg@10" in metrics
    assert "ece_raw" in metrics
    assert results["n_queries"] == 2


def test_run_benchmark_rejects_unverified_candidates(tmp_path):
    dataset_file = tmp_path / "unverified.csv"
    pd.DataFrame({
        "term_id": ["q1", "q1"], "item_id": ["i1", "i2"], "label": [1, 0],
        "label_status": ["known_positive", "unverified_candidate"],
        "bm25_score": [0.9, 0.1], "dense_score": [0.9, 0.1],
        "hybrid_score": [0.9, 0.1], "rerank_score": [0.9, 0.1],
        "calibrated_score": [0.9, 0.1],
    }).to_csv(dataset_file, index=False)
    with pytest.raises(ValueError, match="not fully verified"):
        run_benchmark("configs/benchmark_e2e.yaml", str(dataset_file))


def test_write_benchmark_report(tmp_path):
    report_md = tmp_path / "test_report.md"
    metrics_csv = tmp_path / "test_metrics.csv"

    dummy_results = {
        "metrics": {"retrieval_recall@10": 1.0, "rerank_ndcg@10": 0.9, "ece_raw": 0.02},
        "n_queries": 5,
        "n_rows": 20,
        "dataset_path": "dummy.parquet",
    }

    write_benchmark_report(dummy_results, str(report_md), str(metrics_csv))

    assert report_md.exists()
    assert metrics_csv.exists()
    assert "Recall@10" in report_md.read_text(encoding="utf-8")
