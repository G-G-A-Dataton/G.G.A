"""
scripts/demo_reranker_inference.py
===================================
G.G.A Takımı — Hard Negative Mining & Reranker Inference Demonstration Script

Örnek Veri Seti Üzerinde:
1. Hard Negative Mining (Type-1 & Type-2)
2. Concatenation Formatlama
3. Cross-Encoder Reranking Skorlama Demosu
"""

from __future__ import annotations

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset.hard_negative_miner import (
    format_cross_encoder_input,
    HardNegativeMiner,
)


def main():
    print("=" * 70)
    print("  G.G.A — Cross-Encoder Reranker & Hard Negative Mining Demosu")
    print("=" * 70)

    # Örnek Ürün Kataloğu
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
            "title": "7 Mix Özel Karışım Çim Tohumu 10kg",
            "category": "Bahçe & Yapı Market / Tohum",
            "brand": "GrassSeed",
            "attributes": "Ağırlık: 10kg"
        },
        {
            "item_id": "ITEM_004",
            "title": "Siyah Deri Erkek Klasik Ayakkabı",
            "category": "Ayakkabı / Erkek Ayakkabı",
            "brand": "Kemal Tanca",
            "attributes": "Renk: Siyah | Beden: 42 | Materyal: Hakiki Deri"
        },
    ])

    print("\n[1] Cross-Encoder Concatenation Formatı Testi:")
    sample_formatted = format_cross_encoder_input("38 numara siyah deri kadın babet", items_df.iloc[0])
    print(f"  Query           : {sample_formatted['query']}")
    print(f"  Product Document: {sample_formatted['product_document']}")

    print("\n[2] Hard Negative Mining Çalıştırılıyor:")
    miner = HardNegativeMiner(items_df)

    query = "siyah deri kadın babet"
    pos_id = "ITEM_001"
    retrieval_candidates = ["ITEM_001", "ITEM_004", "ITEM_002", "ITEM_003"]

    triplet = miner.create_triplet(
        query=query,
        pos_item_id=pos_id,
        retrieval_candidates=retrieval_candidates,
        top_k_type1=20,
        max_type2=5
    )

    if triplet:
        print(f"\n[+] Oluşturulan Triplet Yapısı:")
        print(f"  Sorgu        : {triplet['query']}")
        print(f"  Pozitif Dok. : {triplet['positive_doc']}")
        print(f"  Hard Negatifler ({len(triplet['hard_negatives'])} adet):")
        for idx, neg in enumerate(triplet['hard_negatives'], 1):
            print(f"    {idx}. {neg}")

    # Triplet dosyasını dışa aktar
    out_path = os.path.join(PROJECT_ROOT, "outputs", "test_scratch", "demo_train_triplets.jsonl")
    miner.export_triplets_jsonl([triplet] if triplet else [], out_path)

    print("\n[3] Reranker Inference Simülasyonu:")
    print("  Cross-Encoder (BAAI/bge-reranker-large) modeline verilecek çiftler:")
    if triplet:
        candidates_to_score = [triplet['positive_doc']] + triplet['hard_negatives']
        for rank, doc in enumerate(candidates_to_score, 1):
            print(f"  - Pair {rank}: Query = '{triplet['query']}' | Doc = '{doc[:80]}...'")

    print("\n[*] Demo Basariyla Tamamlandi!")
    print("=" * 70)


if __name__ == "__main__":
    main()
