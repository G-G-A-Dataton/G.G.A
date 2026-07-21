"""
src/reranker/listwise.py
========================
G.G.A Takımı — Pairwise and Listwise Ranking Loss Utilities

Aday listelerini gruplar bazında (query level) sıralamak için
Pairwise (RankNet/BPR) ve Listwise (ListNet/NDCG loss) yardımcıları.

Fonksiyonlar:
  - pairwise_margin_loss : (s_pos - s_neg - margin) duyarlı kayıp
  - listnet_loss          : Softmax + Cross-Entropy bazlı listwise loss
  - rerank_group_by_scores: Verilen skor vektörlerine göre query bazında reranking
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Pairwise & Listwise Loss Metrikleri (NumPy / SciPy)
# ---------------------------------------------------------------------------

def pairwise_margin_loss(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    margin: float = 0.2,
) -> float:
    """
    Pairwise Hinge Margin Loss.

    Loss = mean(max(0, margin - (pos_score - neg_score)))
    """
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    diff = pos_scores - neg_scores
    losses = np.maximum(0.0, margin - diff)
    return float(np.mean(losses))


def listnet_loss(
    y_true_relevance: np.ndarray,
    y_pred_scores: np.ndarray,
    temperature: float = 1.0,
) -> float:
    """
    ListNet Listwise Loss.

    P(y) = softmax(y_true / T)
    P_hat(s) = softmax(y_pred / T)
    Loss = - sum( P(y) * log( P_hat(s) ) )
    """
    y_true = np.asarray(y_true_relevance, dtype=np.float64) / temperature
    y_pred = np.asarray(y_pred_scores, dtype=np.float64) / temperature

    # Numeric stability for Softmax
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    p_true = _softmax(y_true)
    p_pred = _softmax(y_pred)

    # Cross entropy
    loss = - np.sum(p_true * np.log(p_pred + 1e-12))
    return float(loss)


# ---------------------------------------------------------------------------
# 2. Query-Group Reranker
# ---------------------------------------------------------------------------

def rerank_by_combination(
    df: pd.DataFrame,
    first_stage_score_col: str = "score",
    rerank_score_col: str = "rerank_score",
    weight_first: float = 0.3,
    weight_rerank: float = 0.7,
    query_col: str = "term_id",
) -> pd.DataFrame:
    """
    İlk aşama (Retrieval/GBDT) skoru ile 2. aşama (Reranker) skorunu harmanlar.

    combined_score = weight_first * norm(first_stage_score) + weight_rerank * norm(rerank_score)
    """
    if df.empty:
        return df

    out = df.copy()

    def _normalize(series):
        s_min = series.min()
        s_max = series.max()
        if s_max > s_min:
            return (series - s_min) / (s_max - s_min)
        return pd.Series(1.0, index=series.index)

    # Query grubu bazında min-max normalizasyonu
    out["_norm_s1"] = out.groupby(query_col)[first_stage_score_col].transform(_normalize)
    out["_norm_s2"] = out.groupby(query_col)[rerank_score_col].transform(_normalize)

    out["combined_rank_score"] = weight_first * out["_norm_s1"] + weight_rerank * out["_norm_s2"]

    # Her sorgu için sırala
    out = out.sort_values([query_col, "combined_rank_score"], ascending=[True, False])
    out = out.drop(columns=["_norm_s1", "_norm_s2"])

    return out.reset_index(drop=True)
