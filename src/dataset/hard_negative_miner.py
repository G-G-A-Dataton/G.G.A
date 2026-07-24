"""
src/dataset/hard_negative_miner.py
===================================
G.G.A Takımı — Hard Negative Mining & Cross-Encoder Input Formatting Modülü

Sorumluluklar:
1. Cross-Encoder Input Formatting:
   - Query: clean_text(query)
   - Product Document: [TITLE] {title} [SEP] [CAT] {category} [SEP] [ATTR] Marka: {brand} | Renk: {color} | Beden: {size} | Materyal: {material}
2. Type 1 Hard Negatives (BM25/Hybrid High-Rank Negatives): Top-20 aday arasından etiketi pozitif olmayanlar.
3. Type 2 Attribute-Conflict Hard Negatives: Aynı kategori ve markada olup Renk/Beden/Materyal çelişeni.
4. Triplet Export: train_triplets.jsonl formatında dışa aktarım.
"""

from __future__ import annotations

import json
import os
import re
import string
import pandas as pd
from typing import List, Dict, Set, Any, Union

from src.retrieval.bge_bm25_hybrid import clean_text


# =============================================================================
# 1. Attribute Extraction & Document Formatting
# =============================================================================

def extract_attribute_value(attr_source: Any, key: str) -> str:
    """
    Attributes metninden veya sözlükten belirli bir anahtara ait değeri çeker.
    Örnek: "Renk: Siyah | Beden: 38" -> key="renk" -> "siyah"
    """
    if isinstance(attr_source, dict):
        for k, v in attr_source.items():
            if clean_text(str(k)) == clean_text(key):
                return str(v).strip()
        return ""

    if not isinstance(attr_source, str) or not attr_source.strip():
        return ""

    # RegEx ile key arama (örn: Renk:\s*([^|;]+))
    pattern = re.compile(rf"{re.escape(key)}\s*:\s*([^|;]+)", re.IGNORECASE)
    match = pattern.search(attr_source)
    if match:
        return match.group(1).strip()
    return ""


def format_cross_encoder_input(query: str, item_row: pd.Series | dict) -> dict[str, str]:
    """
    Cross-Encoder model girdisini belirtilen standart concatenation yapısında formatlar:

    Query: clean_text(query)
    Product Document: [TITLE] {title} [SEP] [CAT] {category} [SEP] [ATTR] Marka: {brand} | Renk: {color} | Beden: {size} | Materyal: {material}
    """
    if isinstance(item_row, pd.Series):
        title = str(item_row.get("title", "")) if pd.notna(item_row.get("title")) else ""
        category = str(item_row.get("category", "")) if pd.notna(item_row.get("category")) else ""
        brand = str(item_row.get("brand", "")) if pd.notna(item_row.get("brand")) else ""
        attributes = item_row.get("attributes", "")
        color_attr = str(item_row.get("color", "")) if pd.notna(item_row.get("color")) else ""
        size_attr = str(item_row.get("size", "")) if pd.notna(item_row.get("size")) else ""
        material_attr = str(item_row.get("material", "")) if pd.notna(item_row.get("material")) else ""
    elif isinstance(item_row, dict):
        title = str(item_row.get("title", ""))
        category = str(item_row.get("category", ""))
        brand = str(item_row.get("brand", ""))
        attributes = item_row.get("attributes", "")
        color_attr = str(item_row.get("color", ""))
        size_attr = str(item_row.get("size", ""))
        material_attr = str(item_row.get("material", ""))
    else:
        title, category, brand, attributes, color_attr, size_attr, material_attr = "", "", "", "", "", "", ""

    # Renk, Beden ve Materyal bilgilerini özniteliklerden veya kolonlardan çıkar
    color = color_attr or extract_attribute_value(attributes, "renk") or "Belirtilmemiş"
    size = size_attr or extract_attribute_value(attributes, "beden") or extract_attribute_value(attributes, "numara") or "Belirtilmemiş"
    material = material_attr or extract_attribute_value(attributes, "materyal") or extract_attribute_value(attributes, "kumaş") or "Belirtilmemiş"
    brand_name = brand or "Belirtilmemiş"

    clean_q = clean_text(query)
    doc_str = (
        f"[TITLE] {title.strip()} [SEP] "
        f"[CAT] {category.strip()} [SEP] "
        f"[ATTR] Marka: {brand_name.strip()} | Renk: {color.strip()} | Beden: {size.strip()} | Materyal: {material.strip()}"
    )

    return {
        "query": clean_q,
        "product_document": doc_str
    }


# =============================================================================
# 2. Hard Negative Mining Engine
# =============================================================================

class HardNegativeMiner:
    """
    BM25/Hybrid sonuçları ve öznitelik çelişkileri üzerinden Type-1 ve Type-2 Hard Negative üreticisi.
    """
    def __init__(self, items_df: pd.DataFrame):
        self.items_df = items_df.copy()
        if "item_id" in self.items_df.columns:
            self.items_df["item_id_str"] = self.items_df["item_id"].astype(str)
            self.items_dict = self.items_df.set_index("item_id_str").to_dict(orient="index")
        else:
            self.items_dict = {}

    def mine_type1_retrieval_negatives(
        self,
        retrieval_candidates: list[str | tuple[str, float]],
        positive_item_ids: set[str],
        top_k: int = 20
    ) -> list[str]:
        """
        Type 1: Birinci aşama retrieval sonucunda ilk top_k içinde çıkan ancak pozitif etikete sahip olmayan ürünler.
        """
        hard_negs = []
        for cand in retrieval_candidates[:top_k]:
            item_id = str(cand[0]) if isinstance(cand, tuple) else str(cand)
            if item_id not in positive_item_ids and item_id in self.items_dict:
                hard_negs.append(item_id)
        return hard_negs

    def mine_type2_attribute_conflict_negatives(
        self,
        query: str,
        pos_item_id: str,
        max_negatives: int = 5
    ) -> list[str]:
        """
        Type 2: Aynı kategori ve markada olan ancak sorgudaki veya pozitif üründeki Renk/Beden/Materyal bilgisiyle çelişen ürünler.
        """
        if pos_item_id not in self.items_dict:
            return []

        pos_row = self.items_dict[pos_item_id]
        pos_cat = str(pos_row.get("category", ""))
        pos_brand = str(pos_row.get("brand", ""))
        pos_color = extract_attribute_value(pos_row.get("attributes", ""), "renk")

        if not pos_cat or not pos_brand:
            return []

        # Aynı kategori ve markadaki diğer ürünleri bul
        mask = (self.items_df["category"] == pos_cat) & (self.items_df["brand"] == pos_brand) & (self.items_df["item_id_str"] != pos_item_id)
        candidates = self.items_df[mask]

        conflict_negatives = []
        for _, row in candidates.iterrows():
            cand_id = row["item_id_str"]
            cand_color = extract_attribute_value(row.get("attributes", ""), "renk")
            
            # Eğer renk bilgileri mevcutsa ve birbiriyle çelişiyorsa bu ürünü hard negatif ekle
            if pos_color and cand_color and clean_text(pos_color) != clean_text(cand_color):
                conflict_negatives.append(cand_id)
                if len(conflict_negatives) >= max_negatives:
                    break

        return conflict_negatives

    def create_triplet(
        self,
        query: str,
        pos_item_id: str,
        retrieval_candidates: list[str | tuple[str, float]] | None = None,
        top_k_type1: int = 20,
        max_type2: int = 5
    ) -> dict[str, Any] | None:
        """
        Tek bir sorgu ve pozitif ürün için Type-1 ve Type-2 hard negatifleri birleştirerek triplet objesi döndürür.
        """
        if pos_item_id not in self.items_dict:
            return None

        pos_row = self.items_dict[pos_item_id]
        pos_doc = format_cross_encoder_input(query, pos_row)["product_document"]

        positive_set = {pos_item_id}
        
        # Type 1 mining
        type1_negs = []
        if retrieval_candidates:
            type1_negs = self.mine_type1_retrieval_negatives(retrieval_candidates, positive_set, top_k=top_k_type1)

        # Type 2 mining
        type2_negs = self.mine_type2_attribute_conflict_negatives(query, pos_item_id, max_negatives=max_type2)

        # Tüm benzersiz hard negatif ürün dokümanlarını formatla
        all_neg_ids = list(dict.fromkeys(type1_negs + type2_negs))
        hard_neg_docs = [
            format_cross_encoder_input(query, self.items_dict[nid])["product_document"]
            for nid in all_neg_ids if nid in self.items_dict
        ]

        if not hard_neg_docs:
            return None

        return {
            "query": clean_text(query),
            "positive_doc": pos_doc,
            "hard_negatives": hard_neg_docs
        }

    def export_triplets_jsonl(self, triplets: list[dict[str, Any]], output_path: str) -> str:
        """
        Üretilen triplet listesini train_triplets.jsonl dosyasına kaydeder.
        """
        parent_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for item in triplets:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"[+] Triplet verisi başarıyla kaydedildi: {output_path} ({len(triplets):,} örnek)")
        return output_path
