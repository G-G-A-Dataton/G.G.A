"""
src/segment_analysis.py
========================
G.G.A Takımı — Segment Analysis Module

Sorgu ve ürün gruplarını kategorilerine, sorgu tiplerine ve özniteliklerine
göre kırarak metrik (Recall@K, NDCG@K, MRR, ECE) raporlaması yapan modül.

Segmentler:
  1. Category L1 (Ayakkabı, Giyim, Elektronik vb.)
  2. Query Length (Head: 1 kelime, Medium: 2-3 kelime, Tail: 4+ kelime)
  3. Brand Query (Marka adı içeren sorgular)
  4. Model-Code Query (Model kodu/numarası içeren sorgular)
  5. Attribute Query (Renk, Beden, Materyal kelimesi içeren sorgular)
  6. Rare Query (Eğitim kümesinde düşük frekanslı sorgular)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.calibration import evaluate_calibration
from src.retrieval_metrics import evaluation_report

_COLOR_WORDS = {"siyah", "beyaz", "kırmızı", "mavi", "yeşil", "sarı", "gri", "pembe", "mor", "turuncu", "kahverengi", "lacivert"}
_SIZE_WORDS = {"beden", "numara", "size", "small", "medium", "large", "xl", "xxl", "36", "37", "38", "39", "40", "41", "42", "43", "44"}
_MATERIAL_WORDS = {"deri", "pamuk", "pamuklu", "kot", "süet", "keten", "ipek", "yün", "plastik", "ahşap", "metal"}
_MODEL_CODE_RE = re.compile(r"\b[a-z0-9]*\d+[a-z0-9]*\b", re.IGNORECASE)


def classify_query_segments(query_text: str) -> Dict[str, Any]:
    """Sorgu metnini analiz edip ait olduğu segment bayraklarını döndürür."""
    tokens = set(query_text.lower().strip().split())
    n_tokens = len(tokens)

    # Length segment
    if n_tokens == 1:
        length_seg = "Head (1 word)"
    elif 2 <= n_tokens <= 3:
        length_seg = "Medium (2-3 words)"
    else:
        length_seg = "Tail (4+ words)"

    has_color = bool(tokens & _COLOR_WORDS)
    has_size = bool(tokens & _SIZE_WORDS)
    has_material = bool(tokens & _MATERIAL_WORDS)
    has_model_code = bool(_MODEL_CODE_RE.search(query_text))

    return {
        "length_segment": length_seg,
        "n_tokens": n_tokens,
        "has_color": has_color,
        "has_size": has_size,
        "has_material": has_material,
        "has_attribute": (has_color or has_size or has_material),
        "has_model_code": has_model_code,
    }


def compute_segment_breakdown(
    df: pd.DataFrame,
    query_col: str = "term_id",
    query_text_col: str = "query_text",
    score_col: str = "score",
    label_col: str = "label",
    category_col: str = "item_category",
) -> Dict[str, pd.DataFrame]:
    """
    Sorgu ve sonuç grubunu segmentlerine ayırıp her segment için metrik tablosu üretir.
    """
    if df.empty:
        return {}

    out = df.copy()

    # Query text yoksa varsayılan atar
    if query_text_col not in out.columns:
        out[query_text_col] = out[query_col].astype(str)

    # Segment bayraklarını ekle
    seg_info = out[query_text_col].apply(classify_query_segments)
    seg_df = pd.DataFrame(list(seg_info))
    out = pd.concat([out.reset_index(drop=True), seg_df.reset_index(drop=True)], axis=1)

    # Kategori L1
    if category_col in out.columns:
        out["category_l1"] = out[category_col].astype(str).str.split("/").str[0].str.strip()
    else:
        out["category_l1"] = "General"

    breakdowns: Dict[str, pd.DataFrame] = {}

    # 1. Kategori Bazlı Rapor
    cat_rows = []
    for cat, group in out.groupby("category_l1"):
        if group[query_col].nunique() < 1:
            continue
        rep = evaluation_report(group, ks=[10, 50], score_col=score_col, label_col=label_col, query_col=query_col, verbose=False)
        cal = evaluate_calibration(group[label_col].to_numpy(), np.clip(group[score_col].to_numpy(), 0.0, 1.0))
        cat_rows.append({
            "segment_type": "Category L1",
            "segment_name": cat,
            "n_queries": int(group[query_col].nunique()),
            "n_pairs": len(group),
            "recall@10": rep.get("recall@10", 0.0),
            "ndcg@10": rep.get("ndcg@10", 0.0),
            "mrr": rep.get("mrr", 0.0),
            "ece": cal["ece"],
        })
    breakdowns["by_category"] = pd.DataFrame(cat_rows)

    # 2. Sorgu Uzunluğu Bazlı Rapor
    len_rows = []
    for seg_name, group in out.groupby("length_segment"):
        rep = evaluation_report(group, ks=[10, 50], score_col=score_col, label_col=label_col, query_col=query_col, verbose=False)
        cal = evaluate_calibration(group[label_col].to_numpy(), np.clip(group[score_col].to_numpy(), 0.0, 1.0))
        len_rows.append({
            "segment_type": "Query Length",
            "segment_name": seg_name,
            "n_queries": int(group[query_col].nunique()),
            "n_pairs": len(group),
            "recall@10": rep.get("recall@10", 0.0),
            "ndcg@10": rep.get("ndcg@10", 0.0),
            "mrr": rep.get("mrr", 0.0),
            "ece": cal["ece"],
        })
    breakdowns["by_query_length"] = pd.DataFrame(len_rows)

    # 3. Özellik (Attribute) Bazlı Rapor
    attr_rows = []
    for attr_name, mask_col in [("Has Color", "has_color"), ("Has Size", "has_size"), ("Has Model Code", "has_model_code")]:
        group = out[out[mask_col] == True]
        if not group.empty:
            rep = evaluation_report(group, ks=[10, 50], score_col=score_col, label_col=label_col, query_col=query_col, verbose=False)
            cal = evaluate_calibration(group[label_col].to_numpy(), np.clip(group[score_col].to_numpy(), 0.0, 1.0))
            attr_rows.append({
                "segment_type": "Attribute / Feature",
                "segment_name": attr_name,
                "n_queries": int(group[query_col].nunique()),
                "n_pairs": len(group),
                "recall@10": rep.get("recall@10", 0.0),
                "ndcg@10": rep.get("ndcg@10", 0.0),
                "mrr": rep.get("mrr", 0.0),
                "ece": cal["ece"],
            })
    breakdowns["by_attributes"] = pd.DataFrame(attr_rows)

    return breakdowns
