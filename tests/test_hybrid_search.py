"""
tests/test_hybrid_search.py
============================
G.G.A Takımı — Hybrid Search Unit Tests
"""

import pandas as pd
import pytest
from src.retrieval.hybrid_search import linear_fusion, reciprocal_rank_fusion


def test_reciprocal_rank_fusion():
    bm25 = [("item_1", 10.0), ("item_2", 8.0), ("item_3", 5.0)]
    dense = [("item_2", 0.95), ("item_1", 0.85), ("item_4", 0.70)]

    # item_2 is rank 2 in bm25, rank 1 in dense
    # item_1 is rank 1 in bm25, rank 2 in dense
    # with equal weights, item_1 and item_2 should get equal RRF score

    res = reciprocal_rank_fusion(
        bm25_results=bm25,
        dense_results=dense,
        k=60,
        bm25_weight=0.5,
        dense_weight=0.5,
    )

    ids = [item_id for item_id, score in res]
    assert "item_1" in ids[:2]
    assert "item_2" in ids[:2]
    assert ids[2:] == ["item_3", "item_4"] or ids[2:] == ["item_4", "item_3"]


def test_linear_fusion():
    bm25 = [("item_1", 10.0), ("item_2", 5.0)]
    dense = [("item_1", 0.8), ("item_2", 0.4)]

    res = linear_fusion(
        bm25_results=bm25,
        dense_results=dense,
        bm25_weight=0.5,
        dense_weight=0.5,
    )

    assert res[0][0] == "item_1"
    assert res[1][0] == "item_2"
