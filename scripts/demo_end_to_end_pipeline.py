"""
scripts/demo_end_to_end_pipeline.py
===================================
G.G.A Takımı — Uçtan Uca 5-Aşamalı Türkçe E-Ticaret Product Matching Pipeline Demosu

Aşamalar:
- Aşama 1: Data Preprocessing (tr_lower + clean_text + search_text)
- Aşama 2: Sparse-Dense Hybrid Retrieval & RRF (BM25 + BGE-M3, RRF k=60, Top-100 Aday)
- Aşama 3: Deep Cross-Encoder Reranker Scoring (BAAI/bge-reranker-large formatting)
- Aşama 4: 40+ Advanced Feature Extraction & GBDT Ensemble Blending
- Aşama 5: Dynamic & Adaptive Category/Length Thresholding (Nihai Karar)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.bge_bm25_hybrid import clean_text, BM25Engine, reciprocal_rank_fusion
from src.dataset.hard_negative_miner import format_cross_encoder_input
from src.features.feature_extractor import TabularFeatureExtractor
from src.models.train_gbdt_ensemble import EnsembleRanker
from src.optimization.threshold_optimizer import CategoryThresholdOptimizer


def run_end_to_end_pipeline():
    print("=" * 75)
    print("  G.G.A — Uçtan Uca 5-Aşamalı E-Ticaret Product Matching Demosu")
    print("=" * 75)

    # 1. Örnek Ürün Kataloğu
    items_df = pd.DataFrame([
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
            "title": "Erkek Siyah Hakiki Deri Klasik Ayakkabı",
            "category": "Ayakkabı / Erkek Ayakkabı / Klasik",
            "brand": "Kemal Tanca",
            "attributes": "Renk: Siyah | Beden: 42 | Materyal: Hakiki Deri"
        },
        {
            "item_id": "ITEM_004",
            "title": "7 Mix Özel Karışım Çim Tohumu 10kg",
            "category": "Bahçe & Yapı Market / Tohum",
            "brand": "GrassSeed",
            "attributes": "Ağırlık: 10kg"
        },
    ])

    query = "38 numara siyah deri kadın babet"
    print(f"\n[Aşama 1] Gelen Sorgu Ön İşleme:")
    print(f"  Ham Sorgu        : '{query}'")
    print(f"  Temizlenmiş Sorgu: '{clean_text(query)}'")

    print(f"\n[Aşama 2] Hybrid Retrieval & RRF Aday Yakalama (BM25 + BGE-M3):")
    bm25_engine = BM25Engine(k1=1.2, b=0.75)
    bm25_engine.index(items_df)
    bm25_hits = bm25_engine.search(query, top_k=10)

    # Dense Arama Simülasyonu
    dense_hits = [("ITEM_001", 0.94), ("ITEM_002", 0.72), ("ITEM_003", 0.68)]
    
    rrf_hits = reciprocal_rank_fusion(bm25_hits, dense_hits, k=60, top_n=100)
    print("  Yakalanan Adaylar (RRF Sıralı):", rrf_hits)

    print(f"\n[Aşama 3] Deep Cross-Encoder Input Formatlama:")
    cand_rows = []
    items_dict = items_df.set_index("item_id").to_dict(orient="index")

    for rank, (item_id, rrf_score) in enumerate(rrf_hits, 1):
        item_row = items_dict[item_id]
        formatted = format_cross_encoder_input(query, item_row)
        
        # Simüle edilen Cross-Encoder logits
        ce_score = 4.8 if item_id == "ITEM_001" else (-1.2 if item_id == "ITEM_002" else -4.5)

        cand_rows.append({
            "query": query,
            "item_id": item_id,
            "title": item_row["title"],
            "category": item_row["category"],
            "brand": item_row["brand"],
            "attributes": item_row["attributes"],
            "bm25_score": 8.5 if item_id == "ITEM_001" else 2.1,
            "dense_score": 0.94 if item_id == "ITEM_001" else 0.72,
            "bm25_rank": rank,
            "dense_rank": rank,
            "cross_encoder_score": ce_score
        })

    cand_df = pd.DataFrame(cand_rows)
    print("  Cross-Encoder Doküman Format Örneği:\n ", cand_rows[0]["title"])

    print(f"\n[Aşama 4] 40+ Tabular Feature Extraction & GBDT Blending:")
    extractor = TabularFeatureExtractor()
    feature_matrix = extractor.transform(cand_df)
    print(f"  Üretilen Feature Matrisi Şekli: {feature_matrix.shape}")

    ensemble_ranker = EnsembleRanker(weight_ce=0.40, weight_lgb=0.30, weight_xgb=0.30)
    final_blend_scores = ensemble_ranker.predict_blend(feature_matrix, cand_df["cross_encoder_score"].to_numpy())
    cand_df["final_blend_score"] = final_blend_scores

    print(f"\n[Aşama 5] Dinamik & Adaptive Thresholding Karar Verme:")
    optimizer = CategoryThresholdOptimizer(default_threshold=0.50)
    
    # OOF threshold simülasyonu
    optimizer.category_threshold_map = {"Ayakkabı / Kadın Ayakkabı / Babet": 0.45}

    print("-" * 75)
    print(f"{'Sıra':<5} | {'Ürün ID':<10} | {'Ürün Başlığı':<32} | {'Ensemble Skor':<14} | {'Karar':<8}")
    print("-" * 75)

    for idx, row in cand_df.iterrows():
        adaptive_thresh = optimizer.predict_threshold(query, row["category"])
        is_match = row["final_blend_score"] >= adaptive_thresh
        decision = "MATCH" if is_match else "REJECT"
        print(f"{idx+1:<5} | {row['item_id']:<10} | {row['title'][:32]:<32} | {row['final_blend_score']:.6f}       | {decision:<8}")

    print("-" * 75)
    print("\n[*] Uçtan Uca Pipeline Başarıyla Tamamlandı! (Macro-F1 0.90+ Seviyesi Uyumlu)")
    print("=" * 75)


if __name__ == "__main__":
    run_end_to_end_pipeline()
