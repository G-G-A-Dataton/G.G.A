"""
tests/test_segment_analysis.py
===============================
G.G.A Takımı — Segment Analysis Unit Tests
"""

import pandas as pd
import pytest
from src.segment_analysis import classify_query_segments, compute_segment_breakdown


def test_classify_query_segments():
    res = classify_query_segments("siyah bot 42 beden")
    assert res["length_segment"] == "Tail (4+ words)"
    assert res["has_color"] is True
    assert res["has_size"] is True
    assert res["has_model_code"] is True  # "42" matches model code regex


def test_compute_segment_breakdown():
    df = pd.DataFrame({
        "term_id": ["q1", "q1", "q2", "q2"],
        "item_id": ["i1", "i2", "i3", "i4"],
        "score": [0.9, 0.1, 0.8, 0.2],
        "label": [1, 0, 1, 0],
        "query_text": ["siyah bot", "siyah bot", "kırmızı elbise 38 beden", "kırmızı elbise 38 beden"],
        "item_category": ["Ayakkabı/Kadın", "Ayakkabı/Kadın", "Giyim/Elbise", "Giyim/Elbise"],
    })

    breakdowns = compute_segment_breakdown(df)
    assert "by_category" in breakdowns
    assert "by_query_length" in breakdowns
    assert "by_attributes" in breakdowns

    cat_df = breakdowns["by_category"]
    assert len(cat_df) == 2
