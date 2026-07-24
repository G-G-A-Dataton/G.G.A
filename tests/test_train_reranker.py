"""
tests/test_train_reranker.py
=============================
Hard Negative Miner & Reranker Training Pipeline Birim Testleri
"""

import os
import json
import unittest
import pandas as pd
import numpy as np

from src.dataset.hard_negative_miner import (
    clean_text,
    extract_attribute_value,
    format_cross_encoder_input,
    HardNegativeMiner,
)
from src.models.train_reranker import (
    compute_mrr_at_k,
    compute_ndcg_at_k,
)


class TestTrainRerankerPipeline(unittest.TestCase):

    def setUp(self):
        self.sample_items = pd.DataFrame([
            {
                "item_id": "ITEM_001",
                "title": "Siyah Deri Kadın Babet Ayakkabı",
                "category": "Ayakkabı / Kadın Ayakkabı / Babet",
                "brand": "Derimod",
                "attributes": "Renk: Siyah | Beden: 38 | Materyal: Deri"
            },
            {
                "item_id": "ITEM_002",
                "title": "Kırmızı Süet Kadın Babet Ayakkabı",
                "category": "Ayakkabı / Kadın Ayakkabı / Babet",
                "brand": "Derimod",
                "attributes": "Renk: Kırmızı | Beden: 38 | Materyal: Süet"
            },
            {
                "item_id": "ITEM_003",
                "title": "Erkek Spor Ayakkabı",
                "category": "Ayakkabı / Erkek Ayakkabı",
                "brand": "Nike",
                "attributes": "Renk: Beyaz | Beden: 42"
            },
        ])
        self.miner = HardNegativeMiner(self.sample_items)

    def test_format_cross_encoder_input(self):
        q = "38 numara siyah deri kadın babet"
        formatted = format_cross_encoder_input(q, self.sample_items.iloc[0])
        
        self.assertEqual(formatted["query"], "38 numara siyah deri kadın babet")
        self.assertIn("[TITLE] Siyah Deri Kadın Babet Ayakkabı", formatted["product_document"])
        self.assertIn("[CAT] Ayakkabı / Kadın Ayakkabı / Babet", formatted["product_document"])
        self.assertIn("[ATTR] Marka: Derimod | Renk: Siyah | Beden: 38 | Materyal: Deri", formatted["product_document"])

    def test_extract_attribute_value(self):
        attr_str = "Renk: Kırmızı | Beden: 38 | Materyal: Süet"
        self.assertEqual(extract_attribute_value(attr_str, "renk"), "Kırmızı")
        self.assertEqual(extract_attribute_value(attr_str, "beden"), "38")
        self.assertEqual(extract_attribute_value(attr_str, "materyal"), "Süet")

    def test_hard_negative_mining(self):
        q = "siyah deri kadın babet"
        pos_id = "ITEM_001"
        retrieval_cands = ["ITEM_001", "ITEM_003", "ITEM_002"]

        triplet = self.miner.create_triplet(q, pos_id, retrieval_candidates=retrieval_cands)
        self.assertIsNotNone(triplet)
        self.assertEqual(triplet["query"], "siyah deri kadın babet")
        self.assertIn("[TITLE] Siyah Deri Kadın Babet", triplet["positive_doc"])
        self.assertTrue(len(triplet["hard_negatives"]) > 0)

    def test_export_triplets_jsonl(self):
        triplets = [
            {
                "query": "siyah deri babet",
                "positive_doc": "[TITLE] Siyah Deri Babet",
                "hard_negatives": ["[TITLE] Kırmızı Süet Babet"]
            }
        ]
        out_file = "outputs/test_scratch/test_triplets.jsonl"
        saved_path = self.miner.export_triplets_jsonl(triplets, out_file)
        self.assertTrue(os.path.exists(saved_path))
        
        with open(saved_path, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)
            self.assertEqual(data["query"], "siyah deri babet")

        # Cleanup
        if os.path.exists(out_file):
            os.remove(out_file)

    def test_evaluation_metrics(self):
        eval_samples = [
            {
                "query": "siyah babet",
                "candidates": [
                    ("doc1", 0, 0.2),
                    ("doc2", 1, 0.9),  # Rank 1 (highest score) -> MRR = 1.0
                    ("doc3", 0, 0.1)
                ]
            },
            {
                "query": "spor ayakkabı",
                "candidates": [
                    ("doc1", 0, 0.8),  # Rank 1
                    ("doc2", 1, 0.6),  # Rank 2 (relevant) -> MRR = 1/2 = 0.5
                ]
            }
        ]
        
        mrr = compute_mrr_at_k(eval_samples, k=10)
        # Expected MRR = (1.0 + 0.5) / 2 = 0.75
        self.assertAlmostEqual(mrr, 0.75, places=4)

        ndcg = compute_ndcg_at_k(eval_samples, k=10)
        self.assertTrue(0.0 <= ndcg <= 1.0)


if __name__ == "__main__":
    unittest.main()
