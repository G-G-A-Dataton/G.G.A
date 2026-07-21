"""
tests/test_retrieval_metrics.py
================================
G.G.A Takımı — Retrieval metrik testleri

src/retrieval_metrics.py için unit testler.
78 mevcut testle aynı pytest uyumlu format.
"""

import numpy as np
import pandas as pd
import pytest

from src.retrieval_metrics import (
    build_eval_dataframe,
    evaluation_report,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

def _make_df(rows):
    """Helper: [(term_id, item_id, score, label)] → DataFrame"""
    return pd.DataFrame(rows, columns=["term_id", "item_id", "score", "label"])


@pytest.fixture()
def perfect_df():
    """Pozitif her zaman negatiften yüksek skorda → mükemmel sıralama."""
    return _make_df([
        ("q1", "i1", 0.9, 1),
        ("q1", "i2", 0.4, 0),
        ("q1", "i3", 0.2, 0),
        ("q2", "i4", 0.8, 1),
        ("q2", "i5", 0.3, 0),
    ])


@pytest.fixture()
def worst_df():
    """Pozitif her zaman en altta → en kötü sıralama."""
    return _make_df([
        ("q1", "i1", 0.1, 1),
        ("q1", "i2", 0.9, 0),
        ("q1", "i3", 0.8, 0),
    ])


@pytest.fixture()
def multi_positive_df():
    """Her query için birden fazla pozitif."""
    return _make_df([
        ("q1", "i1", 0.95, 1),
        ("q1", "i2", 0.80, 1),
        ("q1", "i3", 0.60, 0),
        ("q1", "i4", 0.40, 0),
        ("q1", "i5", 0.20, 0),
        ("q2", "i6", 0.70, 1),
        ("q2", "i7", 0.30, 0),
        ("q2", "i8", 0.10, 1),
    ])


# ---------------------------------------------------------------------------
# 1. recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_perfect_ranking_recall_is_one(self, perfect_df):
        """Top-1'de pozitif varsa recall@1 = 1.0."""
        score = recall_at_k(perfect_df, k=1)
        assert score == pytest.approx(1.0)

    def test_worst_ranking_recall_at_1_is_zero(self, worst_df):
        """Pozitif en altta, k=1 → recall = 0."""
        score = recall_at_k(worst_df, k=1)
        assert score == pytest.approx(0.0)

    def test_full_recall_at_large_k(self, worst_df):
        """k ≥ katalog büyüklüğü → tüm pozitifler yakalanır."""
        score = recall_at_k(worst_df, k=100)
        assert score == pytest.approx(1.0)

    def test_multi_positive_partial_recall(self, multi_positive_df):
        """q1: 2 pozitif, top-1'de sadece 1 var → q1 recall@1 = 0.5.
           q2: 2 pozitif, top-1'de 1 var → q2 recall@1 = 0.5.
           Ortalama = 0.5."""
        score = recall_at_k(multi_positive_df, k=1)
        assert score == pytest.approx(0.5)

    def test_all_positives_in_top_k(self, multi_positive_df):
        """k=8, tüm pozitifler kapsanıyor → recall = 1.0."""
        score = recall_at_k(multi_positive_df, k=8)
        assert score == pytest.approx(1.0)

    def test_invalid_k_raises(self, perfect_df):
        with pytest.raises(ValueError, match="k pozitif"):
            recall_at_k(perfect_df, k=0)

    def test_invalid_k_negative_raises(self, perfect_df):
        with pytest.raises(ValueError):
            recall_at_k(perfect_df, k=-5)

    def test_missing_column_raises(self, perfect_df):
        with pytest.raises(ValueError, match="eksik sütunlar"):
            recall_at_k(perfect_df.drop(columns=["score"]), k=10)

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="boş"):
            recall_at_k(pd.DataFrame(columns=["term_id", "item_id", "score", "label"]), k=10)

    def test_invalid_label_raises(self, perfect_df):
        bad = perfect_df.copy()
        bad.loc[0, "label"] = 2
        with pytest.raises(ValueError, match="0 ve 1"):
            recall_at_k(bad, k=10)


# ---------------------------------------------------------------------------
# 2. precision_at_k
# ---------------------------------------------------------------------------

class TestPrecisionAtK:
    def test_perfect_precision_at_1(self, perfect_df):
        score = precision_at_k(perfect_df, k=1)
        assert score == pytest.approx(1.0)

    def test_zero_precision_when_no_positives_in_top_k(self, worst_df):
        """k=1, pozitif en altta → precision@1 = 0."""
        score = precision_at_k(worst_df, k=1)
        assert score == pytest.approx(0.0)

    def test_precision_decreases_with_k(self, perfect_df):
        """Mükemmel sıralamada k büyüdükçe precision azalır."""
        p1 = precision_at_k(perfect_df, k=1)
        p3 = precision_at_k(perfect_df, k=3)
        # q1: 1 pos / 3 items = 0.333, q2: 1/3 = 0.333 → mean = 0.333
        assert p1 >= p3

    def test_precision_formula(self):
        """Basit formül doğrulaması: k=2, 2 item, 1 pozitif → precision = 0.5."""
        df = _make_df([("q1", "i1", 0.9, 1), ("q1", "i2", 0.5, 0)])
        assert precision_at_k(df, k=2) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3. ndcg_at_k
# ---------------------------------------------------------------------------

class TestNdcgAtK:
    def test_perfect_ndcg_is_one(self, perfect_df):
        score = ndcg_at_k(perfect_df, k=5)
        assert score == pytest.approx(1.0)

    def test_worst_ndcg_is_less_than_perfect(self, worst_df):
        score = ndcg_at_k(worst_df, k=3)
        assert score < 1.0
        assert score >= 0.0

    def test_ndcg_at_1_equals_precision_at_1(self, perfect_df):
        """NDCG@1 = Precision@1 (binary relevance, single position)."""
        ndcg = ndcg_at_k(perfect_df, k=1)
        prec = precision_at_k(perfect_df, k=1)
        assert ndcg == pytest.approx(prec)

    def test_ndcg_monotone_in_rank(self):
        """İdeal sıralamada NDCG@K büyük K'da küçük K'dan yüksek veya eşit."""
        df = _make_df([
            ("q1", "i1", 0.9, 1),
            ("q1", "i2", 0.7, 1),
            ("q1", "i3", 0.3, 0),
        ])
        n5 = ndcg_at_k(df, k=5)
        n3 = ndcg_at_k(df, k=3)
        assert n5 == pytest.approx(n3)  # Tüm pozitifler top-3'te, k>3 fark yok

    def test_invalid_k_raises(self, perfect_df):
        with pytest.raises(ValueError):
            ndcg_at_k(perfect_df, k=-1)


# ---------------------------------------------------------------------------
# 4. mean_reciprocal_rank
# ---------------------------------------------------------------------------

class TestMeanReciprocalRank:
    def test_first_result_positive(self, perfect_df):
        """İlk sonuç pozitif → RR = 1.0 → MRR = 1.0."""
        score = mean_reciprocal_rank(perfect_df)
        assert score == pytest.approx(1.0)

    def test_positive_at_second_rank(self):
        df = _make_df([
            ("q1", "i1", 0.9, 0),   # rank 1: negatif
            ("q1", "i2", 0.7, 1),   # rank 2: pozitif → RR = 0.5
        ])
        score = mean_reciprocal_rank(df)
        assert score == pytest.approx(0.5)

    def test_positive_at_third_rank(self):
        df = _make_df([
            ("q1", "i1", 0.9, 0),
            ("q1", "i2", 0.7, 0),
            ("q1", "i3", 0.5, 1),   # rank 3 → RR = 1/3
        ])
        score = mean_reciprocal_rank(df)
        assert score == pytest.approx(1.0 / 3.0, rel=1e-6)

    def test_no_positive_rr_is_zero(self):
        df = _make_df([("q1", "i1", 0.9, 0), ("q1", "i2", 0.5, 0)])
        score = mean_reciprocal_rank(df)
        assert score == pytest.approx(0.0)

    def test_mrr_multiple_queries(self):
        """q1: rank-1 pozitif → RR=1.0; q2: rank-2 pozitif → RR=0.5. Ort=0.75."""
        df = _make_df([
            ("q1", "i1", 0.9, 1),
            ("q1", "i2", 0.5, 0),
            ("q2", "i3", 0.8, 0),
            ("q2", "i4", 0.6, 1),
        ])
        score = mean_reciprocal_rank(df)
        assert score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 5. evaluation_report
# ---------------------------------------------------------------------------

class TestEvaluationReport:
    def test_report_keys(self, perfect_df):
        report = evaluation_report(perfect_df, ks=[1, 10], verbose=False)
        for k in [1, 10]:
            assert f"recall@{k}" in report
            assert f"precision@{k}" in report
            assert f"ndcg@{k}" in report
        assert "mrr" in report
        assert "n_queries" in report

    def test_report_values_in_range(self, multi_positive_df):
        report = evaluation_report(multi_positive_df, ks=[1, 5, 10], verbose=False)
        for key, val in report.items():
            assert 0.0 <= val <= (len(multi_positive_df) if key == "n_queries" else 1.0), \
                f"{key}={val} aralık dışı"

    def test_perfect_ranking_all_ones(self, perfect_df):
        report = evaluation_report(perfect_df, ks=[1, 5], verbose=False)
        assert report["recall@1"] == pytest.approx(1.0)
        assert report["ndcg@1"] == pytest.approx(1.0)
        assert report["mrr"] == pytest.approx(1.0)

    def test_n_queries_correct(self, multi_positive_df):
        report = evaluation_report(multi_positive_df, ks=[1], verbose=False)
        # q1 ve q2 → 2 sorgu
        assert report["n_queries"] == pytest.approx(2.0)

    def test_default_ks(self, perfect_df):
        """K parametresi belirtilmezse varsayılan [1,5,10,20,50,100] kullanılır."""
        report = evaluation_report(perfect_df, verbose=False)
        for k in [1, 5, 10, 20, 50, 100]:
            assert f"recall@{k}" in report

    def test_missing_column_raises(self, perfect_df):
        with pytest.raises(ValueError, match="eksik sütunlar"):
            evaluation_report(perfect_df.drop(columns=["label"]), verbose=False)


# ---------------------------------------------------------------------------
# 6. build_eval_dataframe
# ---------------------------------------------------------------------------

class TestBuildEvalDataframe:
    def test_basic(self):
        meta = pd.DataFrame({
            "term_id": ["q1", "q1", "q2"],
            "item_id": ["i1", "i2", "i3"],
        })
        scores = np.array([0.9, 0.4, 0.7])
        labels = np.array([1, 0, 1])
        df = build_eval_dataframe(meta, scores, labels)
        assert list(df.columns) == ["term_id", "item_id", "score", "label"]
        assert len(df) == 3
        assert df["score"].dtype == np.float64
        assert df["label"].dtype == np.int8

    def test_length_mismatch_raises(self):
        meta = pd.DataFrame({"term_id": ["q1"], "item_id": ["i1"]})
        with pytest.raises(ValueError, match="aynı uzunlukta"):
            build_eval_dataframe(meta, np.array([0.5, 0.3]), np.array([1]))

    def test_roundtrip_with_recall(self):
        """build_eval_dataframe çıktısı recall_at_k'ya beslenebilmeli."""
        meta = pd.DataFrame({
            "term_id": ["q1", "q1"],
            "item_id": ["i1", "i2"],
        })
        df = build_eval_dataframe(meta, np.array([0.9, 0.1]), np.array([1, 0]))
        score = recall_at_k(df, k=1)
        assert score == pytest.approx(1.0)
