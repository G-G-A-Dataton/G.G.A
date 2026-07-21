"""
tests/test_hard_negative_pool.py
================================
G.G.A Takımı — Hard Negative Pool Unit Tests
"""

import pandas as pd
import pytest
from src.hard_negative_pool import HardNegativePool


def test_hard_negative_pool_lifecycle(tmp_path):
    pool_file = tmp_path / "test_pool.parquet"
    pool = HardNegativePool(pool_path=str(pool_file))

    # 1. Add candidates
    candidates = pd.DataFrame({
        "term_id": ["q1", "q1", "q2"],
        "item_id": ["i1", "i2", "i3"],
    })
    added = pool.add_candidates(candidates, source="bm25")
    assert added == 3

    # Summary
    summ = pool.summary()
    assert summ["total_pairs"] == 3
    assert summ["by_status"]["UNVERIFIED"] == 3

    # 2. Update verification
    success = pool.update_verification("q1", "i1", status=HardNegativePool.STATUS_VERIFIED_NEGATIVE)
    assert success is True

    success_pos = pool.update_verification("q1", "i2", status=HardNegativePool.STATUS_FALSE_NEGATIVE_POS)
    assert success_pos is True

    # 3. Check filters
    verified_negs = pool.get_verified_negatives()
    assert len(verified_negs) == 1
    assert verified_negs.iloc[0]["item_id"] == "i1"

    false_negs = pool.get_false_negatives()
    assert len(false_negs) == 1
    assert false_negs.iloc[0]["item_id"] == "i2"

    # 4. Save and reload
    pool.save()
    assert pool_file.exists() or (tmp_path / "test_pool.csv").exists()

    reloaded = HardNegativePool(pool_path=str(pool_file))
    assert len(reloaded.get_verified_negatives()) == 1
