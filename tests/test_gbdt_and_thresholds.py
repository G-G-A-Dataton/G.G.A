"""
tests/test_gbdt_and_thresholds.py
==================================
GBDT LambdaMART, Feature Extractor & Dynamic Threshold Optimizer Birim Testleri
"""

import unittest
import numpy as np
import pandas as pd

from src.features.feature_extractor import (
    levenshtein_similarity,
    char_ngram_jaccard,
    category_hierarchy_overlap,
    TabularFeatureExtractor,
)
from src.models.train_gbdt_ensemble import EnsembleRanker
from src.optimization.threshold_optimizer import (
    get_term_length_threshold,
    compute_macro_f1,
    CategoryThresholdOptimizer,
)


class TestGBDTAndThresholds(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame([
            {
                "query": "derimod 38 numara siyah deri kadın babet",
                "title": "Siyah Deri Kadın Babet Ayakkabı",
                "brand": "Derimod",
                "category": "Ayakkabı / Kadın Ayakkabı / Babet",
                "attributes": "Renk: Siyah | Beden: 38 | Materyal: Deri",
                "bm25_score": 8.5,
                "dense_score": 0.91,
                "bm25_rank": 1,
                "dense_rank": 1,
                "cross_encoder_score": 4.2
            },
            {
                "query": "38 numara siyah deri kadın babet",
                "title": "Erkek Spor Ayakkabı",
                "brand": "Nike",
                "category": "Ayakkabı / Erkek Ayakkabı",
                "attributes": "Renk: Beyaz | Beden: 42",
                "bm25_score": 1.2,
                "dense_score": 0.40,
                "bm25_rank": 15,
                "dense_rank": 20,
                "cross_encoder_score": -3.1
            }
        ])

    def test_feature_extractor(self):
        extractor = TabularFeatureExtractor()
        feat_df = extractor.transform(self.sample_df)

        self.assertEqual(len(feat_df), 2)
        self.assertTrue(feat_df.shape[1] >= 40)
        self.assertEqual(feat_df.iloc[0]["exact_brand_match"], 1.0)
        self.assertGreater(feat_df.iloc[0]["word_overlap_ratio"], 0.5)

    def test_ensemble_ranker_blending(self):
        ranker = EnsembleRanker(weight_ce=0.40, weight_lgb=0.30, weight_xgb=0.30)
        ce_scores = np.array([4.2, -3.1])
        
        # Test without fitted GBDT models (uses normalized CE fallback)
        blended = ranker.predict_blend(self.sample_df, ce_scores)
        self.assertEqual(len(blended), 2)
        self.assertTrue(blended[0] > blended[1])

    def test_term_length_threshold(self):
        short_q = "çim halı"
        long_q = "38 numara siyah rugan kadın babet"
        
        self.assertEqual(get_term_length_threshold(short_q), 0.65)
        self.assertEqual(get_term_length_threshold(long_q), 0.35)

    def test_category_threshold_optimizer(self):
        oof_df = pd.DataFrame({
            "category": ["Ayakkabı"] * 10,
            "label": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "predicted_prob": [0.9, 0.8, 0.85, 0.7, 0.2, 0.15, 0.1, 0.3, 0.4, 0.05]
        })

        optimizer = CategoryThresholdOptimizer(default_threshold=0.50)
        thresh_map = optimizer.fit(oof_df)

        self.assertIn("Ayakkabı", thresh_map)
        self.assertTrue(0.10 <= thresh_map["Ayakkabı"] <= 0.90)

        # Dynamic threshold prediction
        pred_thresh = optimizer.predict_threshold("siyah babet", "Ayakkabı")
        self.assertTrue(0.15 <= pred_thresh <= 0.85)


if __name__ == "__main__":
    unittest.main()
