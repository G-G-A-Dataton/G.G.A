"""
scripts/retrieval/demo_bge_bm25_hybrid.py
=========================================
Kullanım Örneği ve Çalıştırılabilir Demo Scripti
BM25 + BAAI/bge-m3 Hybrid Retrieval & RRF
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.retrieval.bge_bm25_hybrid import (
    tr_lower,
    clean_text,
    build_search_text,
    reciprocal_rank_fusion,
    BM25Engine,
    DenseBGEEngine,
    HybridRetrievalPipeline,
)


def main():
    print("=" * 65)
    print("  G.G.A — Türkçe BM25 + BAAI/bge-m3 Hybrid Retrieval & RRF Demosu")
    print("=" * 65)

    # Örnek Ürün Kataloğu
    sample_items = pd.DataFrame([
        {
            "item_id": "ITEM_001",
            "title": "Siyah Deri Kadın Babet Ayakkabı",
            "category": "Ayakkabı / Kadın Ayakkabı / Babet",
            "brand": "Derimod",
            "attributes": "Beden: 38 | Renk: Siyah | Materyal: Deri"
        },
        {
            "item_id": "ITEM_002",
            "title": "7 Mix Özel Karışım Çim Tohumu 10kg",
            "category": "Bahçe & Yapı Market / Çiçek & Bitki / Tohum",
            "brand": "GrassSeed",
            "attributes": "Ağırlık: 10kg | Tür: 7li Karışım Basmaya Dayanıklı"
        },
        {
            "item_id": "ITEM_003",
            "title": "Organik Buğday Çimi Tozu 100gr",
            "category": "Süpermarket / Gıda & İçecek / Özel Gıda",
            "brand": "Naturiga",
            "attributes": "Ağırlık: 100gr | Organik: Evet"
        },
        {
            "item_id": "ITEM_004",
            "title": "Erkek Siyah Hakiki Deri Klasik Ayakkabı",
            "category": "Ayakkabı / Erkek Ayakkabı / Klasik",
            "brand": "Kemal Tanca",
            "attributes": "Beden: 42 | Renk: Siyah | Materyal: Hakiki Deri"
        },
    ])

    print("\n[1] Data Preprocessing Testi:")
    sample_raw_text = "  İmparatorluk!!   ---  Çim Halı... 100cmx200cm??  "
    print(f"  Ham Metin     : '{sample_raw_text}'")
    print(f"  Temizlenmiş  : '{clean_text(sample_raw_text)}'")

    print("\n[2] Ürün için search_text Alanı Oluşturma:")
    stext = build_search_text(sample_items.iloc[0])
    print(f"  ITEM_001 search_text: '{stext}'")

    print("\n[3] BM25 Engine Başlatılıyor (k1=1.2, b=0.75)...")
    bm25_engine = BM25Engine(k1=1.2, b=0.75)
    bm25_engine.index(sample_items)
    
    query = "38 numara siyah deri kadın babet"
    print(f"\n[4] Sorgu: '{query}'")
    bm25_res = bm25_engine.search(query, top_k=10)
    print("  BM25 Arama Sonuçları:", bm25_res)

    print("\n[5] RRF (Reciprocal Rank Fusion) Testi (k=60, top_n=100)...")
    # Örnek Dense sıralama simülasyonu
    dense_res = [("ITEM_001", 0.92), ("ITEM_004", 0.75)]
    
    rrf_final = reciprocal_rank_fusion(
        bm25_results=bm25_res,
        dense_results=dense_res,
        k=60,
        top_n=100
    )

    print("\n[+] Uçtan Uca Hibrit Sıralama (Top 100 Aday Listesi ve RRF Skorları):")
    print("-" * 65)
    print(f"{'Sıra':<5} | {'Ürün ID':<12} | {'RRF Skoru':<12}")
    print("-" * 65)
    for rank, (item_id, score) in enumerate(rrf_final, 1):
        print(f"{rank:<5} | {item_id:<12} | {score:.6f}")
    print("-" * 65)


if __name__ == "__main__":
    main()
