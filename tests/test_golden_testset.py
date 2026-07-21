"""
tests/test_golden_testset.py
=============================
G.G.A Takımı — Golden Test Set Unit Tests
"""

import pandas as pd
import pytest
from src.golden_testset import build_golden_testset, write_golden_testset_manifest


@pytest.fixture
def dummy_data():
    terms = pd.DataFrame({"term_id": ["q1", "q2"], "query": ["ayakkabı", "elbise"]})
    items = pd.DataFrame({
        "item_id": ["i1", "i2", "i3", "i4"],
        "title": ["Siyah Ayakkabı", "Kırmızı Elbise", "Deray Ayakkabı", "Spor Elbise"],
        "category": ["Ayakkabı", "Elbise", "Ayakkabı", "Elbise"],
        "brand": ["Nike", "Zara", "Puma", "Mango"],
    })
    pairs = pd.DataFrame({
        "term_id": ["q1", "q2"],
        "item_id": ["i1", "i2"],
        "label": [1, 1],
    })
    return terms, items, pairs


def test_build_golden_testset(dummy_data):
    terms, items, pairs = dummy_data
    df = build_golden_testset(
        pairs, items, terms, n_queries=2, negatives_per_query=1, seed=42, verbose=False
    )
    assert not df.empty
    assert set(df.columns) >= {
        "term_id", "item_id", "label", "bm25_rank", "source",
        "query_text", "item_title", "item_category", "item_brand"
    }
    assert (df["label"] == 1).sum() == 2


def test_write_golden_testset_manifest(tmp_path):
    parquet_path = tmp_path / "test.parquet"
    # Create empty dummy file
    parquet_path.write_bytes(b"dummy parquet content")

    manifest_path = write_golden_testset_manifest(
        str(parquet_path), n_queries=10, negatives_per_query=5, seed=42, source_hashes={"a.csv": "hash1"}
    )
    assert manifest_path.endswith("_manifest.json")
