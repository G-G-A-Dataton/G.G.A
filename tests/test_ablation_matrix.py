"""
tests/test_ablation_matrix.py
==============================
G.G.A Takımı — Ablation Matrix Runner Unit Tests
"""

import pandas as pd
import pytest
from scripts.analysis.run_ablation_matrix import run_ablation_suite, write_ablation_reports


def test_run_ablation_suite(tmp_path):
    dataset_file = tmp_path / "dummy_ablation.csv"
    df = pd.DataFrame({
        "term_id": ["q1", "q1", "q2", "q2"],
        "item_id": ["i1", "i2", "i3", "i4"],
        "bm25_score": [0.9, 0.1, 0.8, 0.2],
        "dense_score": [0.8, 0.2, 0.9, 0.1],
        "rrf_score": [0.9, 0.1, 0.9, 0.1],
        "linear_score": [0.9, 0.1, 0.9, 0.1],
        "rerank_score": [0.95, 0.05, 0.95, 0.05],
        "calibrated_score": [0.9, 0.1, 0.9, 0.1],
        "label": [1, 0, 1, 0],
        "label_status": ["verified_positive", "verified_negative"] * 2,
        "bm25_rank": [1, 2, 1, 2],
        "query_text": ["ayakkabı", "ayakkabı", "bot", "bot"],
        "item_title": ["Ayakkabı 1", "Ayakkabı 2", "Bot 1", "Bot 2"],
    })
    df.to_csv(dataset_file, index=False)

    res_df = run_ablation_suite(str(dataset_file))

    assert len(res_df) == 6
    assert set(res_df["code"]) == {"A", "B", "C", "D", "E", "F"}
    assert "recall@50" in res_df.columns
    assert "ndcg@10" in res_df.columns
    assert "ece" in res_df.columns


def test_write_ablation_reports(tmp_path):
    output_csv = tmp_path / "ablation.csv"
    output_md = tmp_path / "ablation.md"

    dummy_df = pd.DataFrame([{
        "code": "A",
        "variant": "BM25 Baseline",
        "recall@50": 0.8,
        "recall@100": 0.9,
        "precision@10": 0.5,
        "ndcg@10": 0.7,
        "mrr": 0.75,
        "ece": 0.05,
        "latency_ms": 1.2,
        "throughput_qps": 500.0,
    }])

    write_ablation_reports(dummy_df, str(output_csv), str(output_md))

    assert output_csv.exists()
    assert output_md.exists()
    assert "BM25 Baseline" in output_md.read_text(encoding="utf-8")
