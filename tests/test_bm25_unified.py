"""
tests/test_bm25_unified.py
===========================
G.G.A Takımı — Unified BM25 Unit Tests
"""

import pandas as pd
import pytest
from src.retrieval.bm25 import BM25Index, BM25Retriever


def test_bm25_unified_retriever():
    items = pd.DataFrame({
        "item_id": ["i1", "i2", "i3"],
        "title": ["Siyah Kadın Ayakkabı", "Kırmızı Elbise", "Siyah Spor Ayakkabı"],
        "category": ["Ayakkabı", "Giyim", "Ayakkabı"],
        "brand": ["Nike", "Zara", "Puma"],
    })

    retriever = BM25Retriever(items, max_df_ratio=1.0)
    res = retriever.retrieve("siyah ayakkabı", k=2)

    assert len(res) == 2
    res_ids = [item_id for item_id, score in res]
    assert "i1" in res_ids or "i3" in res_ids
