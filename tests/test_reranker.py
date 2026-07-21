"""
tests/test_reranker.py
======================
G.G.A Takımı — Reranker Unit Tests
"""

import numpy as np
import pandas as pd
import pytest
from src.reranker.listwise import (
    listnet_loss,
    pairwise_margin_loss,
    rerank_by_combination,
)


def test_pairwise_margin_loss():
    pos = np.array([0.9, 0.8, 0.95])
    neg = np.array([0.1, 0.2, 0.05])
    # pos is much larger than neg + margin -> loss should be 0.0
    loss = pairwise_margin_loss(pos, neg, margin=0.2)
    assert loss == pytest.approx(0.0)

    # Bad predictions
    bad_pos = np.array([0.1, 0.2])
    bad_neg = np.array([0.9, 0.8])
    bad_loss = pairwise_margin_loss(bad_pos, bad_neg, margin=0.2)
    assert bad_loss > 0.5


def test_listnet_loss():
    y_true = np.array([1, 0, 0])
    y_pred_good = np.array([0.9, 0.1, 0.05])
    y_pred_bad = np.array([0.05, 0.9, 0.8])

    loss_good = listnet_loss(y_true, y_pred_good)
    loss_bad = listnet_loss(y_true, y_pred_bad)

    assert loss_good < loss_bad


def test_rerank_by_combination():
    df = pd.DataFrame({
        "term_id": ["q1", "q1", "q1"],
        "item_id": ["i1", "i2", "i3"],
        "score": [0.8, 0.9, 0.1],          # Stage 1: i2 > i1 > i3
        "rerank_score": [0.95, 0.3, 0.05],  # Stage 2: i1 > i2 > i3
    })

    res = rerank_by_combination(df, weight_first=0.3, weight_rerank=0.7)
    # i1 should rank top because rerank score is heavily weighted (0.7)
    assert res.iloc[0]["item_id"] == "i1"
    assert res.iloc[1]["item_id"] == "i2"
