"""
tests/test_active_learning.py
==============================
G.G.A Takımı — Active Learning Unit Tests
"""

import pandas as pd
import pytest
from src.active_learning import compute_uncertainty_scores, sample_uncertain_pairs


def test_compute_uncertainty_scores():
    df = pd.DataFrame({
        "term_id": ["q1", "q1", "q2"],
        "item_id": ["i1", "i2", "i3"],
        "proba": [0.5, 0.99, 0.01],
    })
    res = compute_uncertainty_scores(df)

    assert "uncertainty_margin" in res.columns
    assert "uncertainty_entropy" in res.columns
    assert "uncertainty_least_conf" in res.columns

    # proba=0.5 should have max uncertainty
    assert res.loc[0, "uncertainty_margin"] > res.loc[1, "uncertainty_margin"]
    assert res.loc[0, "uncertainty_entropy"] > res.loc[1, "uncertainty_entropy"]


def test_sample_uncertain_pairs():
    df = pd.DataFrame({
        "term_id": ["q1", "q1", "q1", "q2", "q2"],
        "item_id": ["i1", "i2", "i3", "i4", "i5"],
        "proba": [0.49, 0.51, 0.99, 0.50, 0.01],
    })

    sampled = sample_uncertain_pairs(df, n_samples=3, strategy="entropy", max_per_query=2)

    assert len(sampled) == 3
    # Top uncertain items (0.50, 0.49, 0.51) should be chosen
    # But q1 can have max 2 items
    q1_count = (sampled["term_id"] == "q1").sum()
    assert q1_count <= 2
