"""
tests/test_bge_bm25_hybrid.py
==============================
BM25 + BGE-M3 Hybrid Retrieval ve RRF Modülü Birim Testleri
"""

import unittest
import pandas as pd
import numpy as np

from src.retrieval.bge_bm25_hybrid import (
    tr_lower,
    clean_text,
    build_search_text,
    reciprocal_rank_fusion,
    BM25Engine,
    DenseBGEEngine,
)


class TestBGEBM25Hybrid(unittest.TestCase):

    def test_tr_lower(self):
        self.assertEqual(tr_lower("İSTANBUL IŞIKLARI"), "istanbul ışıkları")
        self.assertEqual(tr_lower("İlahi Şiir"), "ilahi şiir")

    def test_clean_text(self):
        raw = "  İmparatorluk!!   ---  Çim Halı... 100cmx200cm??  "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "imparatorluk çim halı 100cmx200cm")

    def test_build_search_text(self):
        row = {
            "title": "Kedi Çimi Kiti",
            "category": "Pet Shop/Kedi",
            "brand": "Exelox",
            "attributes": "Organik 120gr"
        }
        stext = build_search_text(row)
        self.assertIn("kedi çimi kiti", stext)
        self.assertIn("pet shop kedi", stext)
        self.assertIn("exelox", stext)
        self.assertIn("organik 120gr", stext)

    def test_reciprocal_rank_fusion(self):
        bm25_list = ["ITEM_A", "ITEM_B", "ITEM_C"]
        dense_list = ["ITEM_B", "ITEM_D", "ITEM_A"]
        
        # k=60
        # ITEM_A: rank_bm25=1 -> 1/61, rank_dense=3 -> 1/63. Total = 1/61 + 1/63 = ~0.03226
        # ITEM_B: rank_bm25=2 -> 1/62, rank_dense=1 -> 1/61. Total = 1/62 + 1/61 = ~0.03252
        # ITEM_B should be #1!
        
        results = reciprocal_rank_fusion(bm25_list, dense_list, k=60, top_n=10)
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0][0], "ITEM_B")
        self.assertEqual(results[1][0], "ITEM_A")
        
        # Check formula accuracy
        expected_score_b = (1.0 / 62.0) + (1.0 / 61.0)
        self.assertAlmostEqual(results[0][1], expected_score_b, places=6)

    def test_bm25_engine(self):
        items_df = pd.DataFrame([
            {"item_id": "ITEM_1", "title": "Siyah Deri Kadın Ayakkabı", "category": "Ayakkabı", "brand": "Derimod", "attributes": "Beden:38"},
            {"item_id": "ITEM_2", "title": "Spor Çim Tohumu 7 Mix", "category": "Bahçe", "brand": "GrassSeed", "attributes": "Ağırlık:10kg"},
        ])
        engine = BM25Engine(k1=1.2, b=0.75)
        engine.index(items_df)
        
        results = engine.search("çim tohumu", top_k=10)
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0][0], "ITEM_2")

    def test_dense_engine_precomputed(self):
        item_ids = ["ITEM_1", "ITEM_2"]
        # Dummy 3-dimensional embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=np.float32)

        engine = DenseBGEEngine()
        engine.set_precomputed_embeddings(item_ids, embeddings)
        self.assertEqual(len(engine.item_ids), 2)
        self.assertEqual(engine.embeddings.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
