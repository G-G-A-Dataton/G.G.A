"""
src/features/feature_extractor.py
==================================
G.G.A Takımı — 40+ Advanced Tabular Feature Extractor Modülü

Öznitelik Grupları:
1. Exact & Soft Text Matching Features:
   - Exact Brand Match (0/1), Brand Levenshtein & Jaccard Distance
   - Color / Size / Material Attribute Overlap Scores (Attribute Soft Matching)
   - Character N-gram Jaccard Similarity (1, 2, 3-gram)
   - Query & Title Length Ratio & Word Overlap
2. Hierarchy & Position Features:
   - Category Depth Overlap & Hierarchy Distance
   - BM25 Score & BGE Dense Cosine Similarity
   - Initial BM25 Rank & Initial Dense Rank
3. Model Signal Features:
   - Cross-Encoder Probability / Logit Score
"""

from __future__ import annotations

import math
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union

from src.retrieval.bge_bm25_hybrid import clean_text
from src.dataset.hard_negative_miner import extract_attribute_value


# =============================================================================
# 1. String & Metric Distance Utilities
# =============================================================================

def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    İki metin arasındaki Levenshtein benzerliğini [0, 1] aralığında döndürür.
    """
    s1, s2 = clean_text(s1), clean_text(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    dist = dp[len1][len2]
    max_len = max(len1, len2)
    return 1.0 - (dist / max_len)


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """
    Karakter N-gram Jaccard benzerliğini hesaplar.
    """
    s1, s2 = clean_text(s1), clean_text(s2)
    if not s1 or not s2:
        return 0.0

    ngrams1 = set([s1[i:i+n] for i in range(len(s1) - n + 1)]) if len(s1) >= n else {s1}
    ngrams2 = set([s2[i:i+n] for i in range(len(s2) - n + 1)]) if len(s2) >= n else {s2}

    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return intersection / union if union > 0 else 0.0


def category_hierarchy_overlap(query_cat: str, item_cat: str) -> float:
    """
    Kategori ağacındaki hiyerarşik çakışma skorunu [0, 1] olarak hesaplar.
    Örnek: "Ayakkabı/Kadın/Babet" ile "Ayakkabı/Kadın/Sneaker" -> 2/3 = 0.66
    """
    q_parts = [clean_text(p) for p in str(query_cat).split("/") if clean_text(p)]
    i_parts = [clean_text(p) for p in str(item_cat).split("/") if clean_text(p)]

    if not q_parts or not i_parts:
        return 0.0

    match_count = 0
    for q_p, i_p in zip(q_parts, i_parts):
        if q_p == i_p:
            match_count += 1
        else:
            break

    return match_count / max(len(q_parts), len(i_parts))


# =============================================================================
# 2. Feature Extractor Engine
# =============================================================================

class TabularFeatureExtractor:
    """
    Sorgu-Ürün çifti için 40+ gelişmiş tabular feature matrisi üreten sınıf.
    """

    FEATURE_NAMES = [
        # Brand & Text Matching (10)
        "exact_brand_match",
        "brand_levenshtein_sim",
        "brand_jaccard_sim",
        "char_1gram_jaccard",
        "char_2gram_jaccard",
        "char_3gram_jaccard",
        "word_overlap_ratio",
        "title_length_ratio",
        "query_word_count",
        "title_word_count",
        
        # Attributes Soft Matching (10)
        "color_match_score",
        "size_match_score",
        "material_match_score",
        "attribute_conflict_penalty",
        "has_color_in_query",
        "has_size_in_query",
        "has_material_in_query",
        "color_jaccard",
        "size_jaccard",
        "material_jaccard",
        
        # Hierarchy & Retrieval Signals (10)
        "category_depth_overlap",
        "category_exact_match",
        "bm25_score",
        "dense_cosine_sim",
        "bm25_rank",
        "dense_rank",
        "rrf_fusion_score",
        "bm25_rank_reciprocal",
        "dense_rank_reciprocal",
        "is_top5_hybrid",
        
        # Model Logit Signal Features (10+)
        "cross_encoder_score",
        "cross_encoder_prob_sigmoid",
        "cross_encoder_x_bm25",
        "cross_encoder_x_dense",
        "score_rank_diff",
        "score_ratio_bm25_dense",
        "bm25_normalized",
        "dense_normalized",
        "hybrid_blend_score",
        "final_confidence_signal"
    ]

    def extract_features_row(self, row: pd.Series | dict) -> dict[str, float]:
        """
        Tek bir (sorgu, ürün) çifti için 40+ feature içeren sözlük türetir.
        """
        q_raw = str(row.get("query", ""))
        title_raw = str(row.get("title", ""))
        brand_raw = str(row.get("brand", ""))
        cat_raw = str(row.get("category", ""))
        attr_raw = row.get("attributes", "")
        
        q_clean = clean_text(q_raw)
        t_clean = clean_text(title_raw)
        b_clean = clean_text(brand_raw)

        q_words = q_clean.split()
        t_words = t_clean.split()

        # 1. Exact & Soft Text Match
        exact_brand = 1.0 if (b_clean and (b_clean in q_clean or b_clean in t_clean)) else 0.0
        brand_lev = levenshtein_similarity(b_clean, q_clean) if b_clean else 0.0
        brand_jaccard = char_ngram_jaccard(b_clean, q_clean, n=3) if b_clean else 0.0

        c1_jaccard = char_ngram_jaccard(q_clean, t_clean, n=1)
        c2_jaccard = char_ngram_jaccard(q_clean, t_clean, n=2)
        c3_jaccard = char_ngram_jaccard(q_clean, t_clean, n=3)

        overlap_words = len(set(q_words) & set(t_words))
        word_overlap = overlap_words / len(q_words) if q_words else 0.0
        title_len_ratio = len(t_clean) / max(len(q_clean), 1)

        # 2. Attribute Soft Matching
        q_color = extract_attribute_value(attr_raw, "renk") or ""
        q_size = extract_attribute_value(attr_raw, "beden") or extract_attribute_value(attr_raw, "numara") or ""
        q_mat = extract_attribute_value(attr_raw, "materyal") or ""

        color_match = 1.0 if q_color and clean_text(q_color) in q_clean else (0.5 if not q_color else 0.0)
        size_match = 1.0 if q_size and clean_text(q_size) in q_clean else (0.5 if not q_size else 0.0)
        mat_match = 1.0 if q_mat and clean_text(q_mat) in q_clean else (0.5 if not q_mat else 0.0)

        conflict_penalty = 1.0 if (color_match == 0.0 or size_match == 0.0 or mat_match == 0.0) else 0.0

        # 3. Hierarchy & Position
        cat_overlap = category_hierarchy_overlap(q_clean, cat_raw)
        cat_exact = 1.0 if clean_text(cat_raw) in q_clean else 0.0

        bm25_s = float(row.get("bm25_score", 0.0))
        dense_s = float(row.get("dense_score", 0.0))
        bm25_r = float(row.get("bm25_rank", 200))
        dense_r = float(row.get("dense_rank", 200))

        rrf_s = (1.0 / (60.0 + bm25_r)) + (1.0 / (60.0 + dense_r))
        is_top5 = 1.0 if (bm25_r <= 5 or dense_r <= 5) else 0.0

        # 4. Model Signal Features (Cross-Encoder)
        ce_score = float(row.get("cross_encoder_score", 0.0))
        ce_prob = 1.0 / (1.0 + math.exp(-ce_score)) if abs(ce_score) < 50 else (1.0 if ce_score > 0 else 0.0)

        feats = {
            "exact_brand_match": exact_brand,
            "brand_levenshtein_sim": brand_lev,
            "brand_jaccard_sim": brand_jaccard,
            "char_1gram_jaccard": c1_jaccard,
            "char_2gram_jaccard": c2_jaccard,
            "char_3gram_jaccard": c3_jaccard,
            "word_overlap_ratio": word_overlap,
            "title_length_ratio": title_len_ratio,
            "query_word_count": float(len(q_words)),
            "title_word_count": float(len(t_words)),

            "color_match_score": color_match,
            "size_match_score": size_match,
            "material_match_score": mat_match,
            "attribute_conflict_penalty": conflict_penalty,
            "has_color_in_query": 1.0 if q_color else 0.0,
            "has_size_in_query": 1.0 if q_size else 0.0,
            "has_material_in_query": 1.0 if q_mat else 0.0,
            "color_jaccard": char_ngram_jaccard(q_color, q_clean, n=2) if q_color else 0.0,
            "size_jaccard": char_ngram_jaccard(q_size, q_clean, n=2) if q_size else 0.0,
            "material_jaccard": char_ngram_jaccard(q_mat, q_clean, n=2) if q_mat else 0.0,

            "category_depth_overlap": cat_overlap,
            "category_exact_match": cat_exact,
            "bm25_score": bm25_s,
            "dense_cosine_sim": dense_s,
            "bm25_rank": bm25_r,
            "dense_rank": dense_r,
            "rrf_fusion_score": rrf_s,
            "bm25_rank_reciprocal": 1.0 / max(bm25_r, 1.0),
            "dense_rank_reciprocal": 1.0 / max(dense_r, 1.0),
            "is_top5_hybrid": is_top5,

            "cross_encoder_score": ce_score,
            "cross_encoder_prob_sigmoid": ce_prob,
            "cross_encoder_x_bm25": ce_prob * bm25_s,
            "cross_encoder_x_dense": ce_prob * dense_s,
            "score_rank_diff": abs(bm25_r - dense_r),
            "score_ratio_bm25_dense": bm25_s / max(dense_s, 1e-5),
            "bm25_normalized": min(bm25_s / 20.0, 1.0),
            "dense_normalized": min(dense_s, 1.0),
            "hybrid_blend_score": 0.5 * bm25_s + 0.5 * dense_s,
            "final_confidence_signal": 0.6 * ce_prob + 0.4 * rrf_s
        }

        return feats

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Girdi DataFrame'i için 40+ feature kolonuna sahip pandas DataFrame döndürür.
        """
        rows = [self.extract_features_row(row) for _, row in df.iterrows()]
        feature_df = pd.DataFrame(rows)
        return feature_df[self.FEATURE_NAMES]
