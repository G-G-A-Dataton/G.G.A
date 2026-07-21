"""
tests/test_data_quality.py
===========================
G.G.A Takımı — Data Quality Unit Tests
"""

import pandas as pd
import pytest
from src.data_quality import (
    detect_attribute_coverage,
    detect_duplicate_queries,
    metadata_normalization_report,
    run_full_quality_report,
    validate_label_consistency,
)


@pytest.fixture
def sample_terms():
    return pd.DataFrame({
        "term_id": ["q1", "q2", "q3", "q4"],
        "query": ["ayakkabı", "ayakkabı ", "SİYAH AYAKKABI", "elbise"],
    })


@pytest.fixture
def sample_items():
    return pd.DataFrame({
        "item_id": ["i1", "i2", "i3"],
        "title": ["Siyah Kadın Ayakkabı", "Erkek Elbise", "Çocuk Bot"],
        "category": ["Ayakkabı/Kadın", "Giyim/Elbise", "Ayakkabı/Çocuk"],
        "attributes": ["renk: siyah, beden: 38", "materyal: pamuk", None],
        "gender": ["kadın", "erkek", "unisex"],
        "age_group": ["yetişkin", "yetişkin", "çocuk"],
        "brand": ["Nike", "Adidas", "Puma"],
    })


def test_detect_duplicate_queries(sample_terms):
    res = detect_duplicate_queries(sample_terms)
    assert res["count"] == 1  # "ayakkabı" and "ayakkabı " match normalized
    assert res["total_affected_terms"] == 2
    assert len(res["examples"]) == 1


def test_detect_attribute_coverage(sample_items):
    res = detect_attribute_coverage(sample_items)
    assert "overall_non_null_rate" in res
    assert res["renk_coverage"] > 0
    assert "by_category_l1" in res


def test_validate_label_consistency():
    df_clean = pd.DataFrame({
        "term_id": ["q1", "q1", "q2"],
        "item_id": ["i1", "i2", "i1"],
        "label": [1, 0, 1],
    })
    res_clean = validate_label_consistency(df_clean)
    assert res_clean["is_consistent"] is True
    assert res_clean["violations"] == 0

    df_conflict = pd.DataFrame({
        "term_id": ["q1", "q1"],
        "item_id": ["i1", "i1"],
        "label": [1, 0],
    })
    res_conflict = validate_label_consistency(df_conflict)
    assert res_conflict["is_consistent"] is False
    assert res_conflict["violations"] == 1


def test_metadata_normalization_report(sample_items):
    res = metadata_normalization_report(sample_items)
    assert "gender_dist" in res
    assert "brand_stats" in res
    assert res["brand_stats"]["unique_count"] == 3


def test_run_full_quality_report(sample_terms, sample_items):
    report = run_full_quality_report(sample_terms, sample_items, verbose=False)
    assert "duplicate_queries" in report
    assert "attribute_coverage" in report
    assert "metadata_normalization" in report
