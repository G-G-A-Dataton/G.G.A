"""
tests/test_faiss_index.py
==========================
G.G.A Takımı — FAISS Index Unit Tests
"""

import numpy as np
import pytest
from src.retrieval.faiss_index import FAISSIndex
from src.retrieval.index_versioning import IndexVersion


def test_faiss_build_search_save_load(tmp_path):
    try:
        import faiss
    except ImportError:
        pytest.skip("faiss-cpu veya faiss-gpu yüklü değil")

    dim = 16
    n_items = 100
    np.random.seed(42)

    # Random L2-normalized embeddings
    embeddings = np.random.randn(n_items, dim).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    ids = np.array([f"item_{i}" for i in range(n_items)])

    # Build index with small n_lists
    idx = FAISSIndex(dimension=dim, n_lists=4, n_probes=2)
    idx.build(embeddings, ids)

    assert idx.n_items == n_items

    # Search top 3
    q_vec = embeddings[:2]  # First 2 items as query
    scores, res_ids = idx.search(q_vec, k=3)

    assert scores.shape == (2, 3)
    assert res_ids.shape == (2, 3)
    # Self-match should be top 1
    assert res_ids[0, 0] == "item_0"
    assert res_ids[1, 0] == "item_1"

    # Save & Load
    save_path = str(tmp_path / "faiss_test.index")
    manifest_path = idx.save(save_path)
    assert manifest_path.endswith(".json")

    loaded_idx = FAISSIndex.load(save_path, verify=True)
    assert loaded_idx.n_items == n_items

    # Search on loaded index
    l_scores, l_res_ids = loaded_idx.search(q_vec, k=3)
    np.testing.assert_array_equal(res_ids, l_res_ids)
