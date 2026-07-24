"""
src/optimization/threshold_optimizer.py
========================================
G.G.A Takımı — Dynamic & Adaptive Category/Length Threshold Optimizer Modülü

Özellikler:
1. Term Length Adaptive Thresholds:
   - Kısa sorgular (1-2 kelime): 0.65 (High Precision)
   - Uzun sorgular (3+ kelime): 0.35 (High Recall)
2. Category-Based Nelder-Mead Optimization (scipy.optimize.minimize):
   - Out-Of-Fold (OOF) tahmin verileri üzerinden hiyerarşik hiyerarşi/kategori bazında Macro-F1 maksimizasyonu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from scipy.optimize import minimize

from src.retrieval.bge_bm25_hybrid import clean_text


# =============================================================================
# 1. Term Length Adaptive Threshold Rules
# =============================================================================

def get_term_length_threshold(query: str, short_thresh: float = 0.65, long_thresh: float = 0.35) -> float:
    """
    Sorgu kelime uzunluğuna göre dinamik eşik döndürür:
    - 1-2 kelime -> 0.65 (High Precision)
    - 3+ kelime -> 0.35 (High Recall)
    """
    cleaned = clean_text(query)
    word_count = len(cleaned.split())
    if word_count <= 2:
        return short_thresh
    return long_thresh


# =============================================================================
# 2. Category-Based Nelder-Mead Optimizer
# =============================================================================

def compute_macro_f1(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float) -> float:
    """
    Belirli bir threshold değeri için Binary/Macro-F1 hesaplar.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if (precision + recall) == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


class CategoryThresholdOptimizer:
    """
    OOF tahmin verileri üzerinde Nelder-Mead algoritması (scipy.optimize.minimize) kullanarak 
    kategori bazlı en yüksek Macro-F1 veren eşik matrisini hesaplayan optimizer.
    """
    def __init__(self, default_threshold: float = 0.50):
        self.default_threshold = default_threshold
        self.category_threshold_map: dict[str, float] = {}

    def optimize_category_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Nelder-Mead (Simplex) metodu ile F1 skorunu maksimize eden (negatif F1'i minimize eden) eşiği bulur.
        """
        if len(y_true) == 0 or np.sum(y_true) == 0:
            return self.default_threshold

        # Objective function: -1 * F1 score
        def loss_func(t_arr):
            t = t_arr[0]
            # Bound threshold strictly into [0.10, 0.90]
            if t < 0.10 or t > 0.90:
                return 1.0
            f1 = compute_macro_f1(y_true, y_prob, threshold=t)
            return -f1

        init_t = [self.default_threshold]
        res = minimize(loss_func, init_t, method="Nelder-Mead", options={"maxiter": 50, "xatol": 0.01})
        best_t = float(res.x[0])
        return max(0.10, min(0.90, best_t))

    def fit(self, oof_df: pd.DataFrame) -> dict[str, float]:
        """
        OOF DataFrame içindeki 'category', 'label' ve 'predicted_prob' kolonlarını kullanarak 
        kategori bazlı optimal eşik haritasını (category_threshold_map) üretir.
        """
        if "category" not in oof_df.columns or "label" not in oof_df.columns or "predicted_prob" not in oof_df.columns:
            raise ValueError("oof_df 'category', 'label' ve 'predicted_prob' kolonlarını içermelidir.")

        for cat, group in oof_df.groupby("category"):
            y_t = group["label"].to_numpy()
            y_p = group["predicted_prob"].to_numpy()
            best_t = self.optimize_category_threshold(y_t, y_p)
            self.category_threshold_map[str(cat)] = best_t

        print(f"[+] Kategori bazlı {len(self.category_threshold_map)} adet optimal eşik hesaplandı.")
        return self.category_threshold_map

    def predict_threshold(self, query: str, category: str) -> float:
        """
        Sorgu uzunluğu ve ürün kategorisine göre uyarlanmış final karar eşiğini döndürür.
        """
        cat_str = str(category).strip()
        cat_t = self.category_threshold_map.get(cat_str, self.default_threshold)
        length_t = get_term_length_threshold(query)

        # Dynamic Weighted combination
        final_t = 0.5 * cat_t + 0.5 * length_t
        return max(0.15, min(0.85, final_t))
