"""
src/attribute_features.py
==========================
G.G.A Takımı — Attribute Parsing ve Feature Üretimi (Gün 8 görevi)

Ahmet Emin Işın tarafından hazırlanmıştır.

Amaç:
  items.csv'deki unstructured "attributes" kolonunu parse edip:
  - renk (color)
  - materyal (material)
  - desen (pattern)
  - beden (size/taille)
  
  gibi önemli özellikleri çıkarmak, sonra sorgu ve ürün
  arasındaki eşleşmeleri ölçmek.
  
Örnek:
  Sorgu: "siyah erkek spor ayakkabı"
  Item attributes: "renk: siyah, materyal: tekstil, ..."
  → color_match = 1 (her iki tarafta "siyah" var)
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attribute Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_attributes(attr_str: str) -> Dict[str, str]:
    """
    Unstructured attribute string'ini key-value dict'ine çevir.
    
    Örnek:
        "renk: siyah, materyal: tekstil, desen: düz"
        → {"renk": "siyah", "materyal": "tekstil", "desen": "düz"}
    """
    if not attr_str or not isinstance(attr_str, str):
        return {}
    
    result = {}
    # Split by comma, then by colon
    for pair in attr_str.split(","):
        if ":" not in pair:
            continue
        key, val = pair.split(":", 1)
        key = key.strip().lower()
        val = val.strip().lower()
        result[key] = val
    
    return result


def extract_color_from_query(query: str) -> List[str]:
    """
    Sorgu metninden renk adlarını çıkar.
    
    Ön tanımlı renk listesi kullanır.
    """
    COLORS = {
        "siyah": ["siyah", "black"],
        "beyaz": ["beyaz", "white"],
        "kırmızı": ["kırmızı", "red"],
        "mavi": ["mavi", "blue"],
        "yeşil": ["yeşil", "green"],
        "sarı": ["sarı", "yellow"],
        "turuncu": ["turuncu", "orange"],
        "mor": ["mor", "purple", "viole"],
        "pembe": ["pembe", "pink"],
        "gri": ["gri", "gray", "grey"],
        "kahverengi": ["kahverengi", "brown"],
        "bej": ["bej", "beige"],
        "altın": ["altın", "gold"],
        "gümüş": ["gümüş", "silver"],
        "krem": ["krem", "cream"],
        "lacivert": ["lacivert", "navy"],
    }
    
    query_lower = query.lower()
    found_colors = []
    
    for canonical_color, aliases in COLORS.items():
        for alias in aliases:
            if alias in query_lower:
                found_colors.append(canonical_color)
                break
    
    return list(set(found_colors))  # Unique colors only


def extract_material_from_query(query: str) -> List[str]:
    """
    Sorgu metninden materyal adlarını çıkar.
    """
    MATERIALS = {
        "pamuk": ["pamuk", "cotton"],
        "keten": ["keten", "linen"],
        "polyester": ["polyester", "poly"],
        "vinil": ["vinil", "vinyl"],
        "deri": ["deri", "leather"],
        "suet": ["suet", "suede"],
        "seramik": ["seramik", "ceramic"],
        "metal": ["metal"],
        "ahşap": ["ahşap", "wood"],
        "cam": ["cam", "glass"],
        "plastik": ["plastik", "plastic"],
        "silikon": ["silikon", "silicone"],
    }
    
    query_lower = query.lower()
    found_materials = []
    
    for canonical_material, aliases in MATERIALS.items():
        for alias in aliases:
            if alias in query_lower:
                found_materials.append(canonical_material)
                break
    
    return list(set(found_materials))


def extract_pattern_from_query(query: str) -> List[str]:
    """
    Sorgu metninden desen adlarını çıkar.
    """
    PATTERNS = {
        "düz": ["düz", "solid", "plain"],
        "çizgili": ["çizgili", "striped"],
        "kareli": ["kareli", "checkered", "plaid"],
        "puantiye": ["puantiye", "dotted", "polka"],
        "çiçekli": ["çiçekli", "floral"],
        "geometrik": ["geometrik", "geometric"],
        "abstract": ["abstract"],
        "desen": ["desen"],
    }
    
    query_lower = query.lower()
    found_patterns = []
    
    for canonical_pattern, aliases in PATTERNS.items():
        for alias in aliases:
            if alias in query_lower:
                found_patterns.append(canonical_pattern)
                break
    
    return list(set(found_patterns))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature Üretimi
# ─────────────────────────────────────────────────────────────────────────────

def add_attribute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame'e attribute-based features ekle.
    
    İnput DataFrame kolonları:
      - query (sorgu metni)
      - attributes (unstructured attribute string)
    
    Çıktı Kolonları:
      - color_match (0/1): Sorgu ve ürün rengi eşleş mi?
      - material_match (0/1): Sorgu ve ürün materyali eşleş mi?
      - pattern_match (0/1): Sorgu ve ürün deseni eşleş mi?
      - has_color (0/1): Ürünün color attribute'u var mı?
      - color_mismatch (0/1): Sorgu'da renk var ama ürün'de farklı renk var mı?
    
    Not: Beden ve diğer attributes için attribute parser kurulduğunda eklenebilir.
    """
    out = df.copy()
    
    # Başlangıç: tüm features sıfır
    n = len(out)
    out["color_match"] = 0
    out["material_match"] = 0
    out["pattern_match"] = 0
    out["has_color"] = 0
    out["color_mismatch"] = 0
    
    # Her satır için features hesapla
    for idx in range(n):
        query = out.iloc[idx]["query"] if "query" in out.columns else ""
        attr_str = out.iloc[idx]["attributes"] if "attributes" in out.columns else ""
        
        if not isinstance(query, str) or not isinstance(attr_str, str):
            continue
        
        # Sorgudan özellikleri çıkar
        query_colors = set(extract_color_from_query(query))
        query_materials = set(extract_material_from_query(query))
        query_patterns = set(extract_pattern_from_query(query))
        
        # Ürün attributes'ini parse et
        attrs = parse_attributes(attr_str)
        
        # Renk eşleşmesi
        if "renk" in attrs:
            attr_color = attrs["renk"]
            out.loc[idx, "has_color"] = 1
            
            # Renk adı ile eşleş
            found_match = False
            for qc in query_colors:
                if qc in attr_color:
                    out.loc[idx, "color_match"] = 1
                    found_match = True
                    break
            
            if not found_match and query_colors:
                # Sorgu'da renk var ama ürün'de farklı renk var
                out.loc[idx, "color_mismatch"] = 1
        
        # Materyal eşleşmesi
        if "materyal" in attrs:
            attr_material = attrs["materyal"]
            for qm in query_materials:
                if qm in attr_material:
                    out.loc[idx, "material_match"] = 1
                    break
        
        # Desen eşleşmesi
        if "desen" in attrs:
            attr_pattern = attrs["desen"]
            for qp in query_patterns:
                if qp in attr_pattern:
                    out.loc[idx, "pattern_match"] = 1
                    break
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Test / PoC
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    
    from src.data import load_terms, load_items
    
    # Test
    print("[TEST] Attribute parser başlıyor...")
    
    # Sample data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")
    terms_df = load_terms(os.path.join(data_dir, "terms.csv"))
    items_df = load_items(os.path.join(data_dir, "items.csv"))
    
    # Merge pozitif çiftler
    train_df = pd.read_csv(
        os.path.join(data_dir, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"}
    )
    sample = train_df.head(1000).copy()
    sample = sample.merge(terms_df, on="term_id")
    sample = sample.merge(items_df, on="item_id")
    
    # Features ekle
    sample_with_attrs = add_attribute_features(sample)
    
    print("\nSample feature values:")
    print(sample_with_attrs[["query", "title", "color_match", "material_match", 
                             "pattern_match", "has_color", "color_mismatch"]].head(10).to_string())
    
    print("\nFeature statistics:")
    print(sample_with_attrs[["color_match", "material_match", "pattern_match", 
                             "has_color", "color_mismatch"]].describe())
