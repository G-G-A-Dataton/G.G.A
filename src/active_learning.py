"""
src/active_learning.py
======================
G.G.A Takımı — Active Learning & Uncertainty Sampling

Modelin tahminlerinde en belirsiz olduğu (borderline/uncertain) veya
yüksek özgüvenli hatalara yatkın adayları seçen modül.

Yöntemler:
  - Entropy (Bilgi Entropisi): -p log p - (1-p) log(1-p)
  - Margin Sampling: |p - 0.5| 'e en yakın örnekler
  - Least Confidence: 1 - max(p, 1-p)
  - High-Confidence Negative/Positive Filtering (hata arama)

Kullanım:
  >>> from src.active_learning import compute_uncertainty_scores, sample_uncertain_pairs
  >>> df_with_unc = compute_uncertainty_scores(candidate_df, proba_col="proba")
  >>> selected = sample_uncertain_pairs(df_with_unc, n_samples=1000, strategy="entropy")
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Belirsizlik Skorları Hesaplama
# ---------------------------------------------------------------------------

def compute_uncertainty_scores(
    df: pd.DataFrame,
    proba_col: str = "proba",
    copy: bool = True,
) -> pd.DataFrame:
    """
    DataFrame'e belirsizlik metrikleri ekler.

    Eklenecek kolonlar:
      - uncertainty_margin    : 1.0 - 2 * |p - 0.5|  (1 = maksimum belirsiz)
      - uncertainty_entropy   : -p log2(p) - (1-p) log2(1-p)
      - uncertainty_least_conf: 1 - max(p, 1-p)

    Parameters
    ----------
    df : pd.DataFrame
        Olasılık sütununu içeren DataFrame.
    proba_col : str
        Model tahmin olasılığı (0.0 - 1.0 arası).
    copy : bool
        Kopya DataFrame döndür.

    Returns
    -------
    pd.DataFrame
        Belirsizlik kolonları eklenmiş DataFrame.
    """
    if proba_col not in df.columns:
        raise ValueError(f"compute_uncertainty_scores: '{proba_col}' sütunu bulunamadı")

    out = df.copy() if copy else df

    probas = np.clip(out[proba_col].to_numpy(dtype=np.float64), 1e-7, 1.0 - 1e-7)

    # 1. Margin (0.5'e yakınlık)
    out["uncertainty_margin"] = 1.0 - 2.0 * np.abs(probas - 0.5)

    # 2. Shannon Entropisi (binary)
    entropy = - (probas * np.log2(probas) + (1.0 - probas) * np.log2(1.0 - probas))
    out["uncertainty_entropy"] = np.nan_to_num(entropy, nan=0.0)

    # 3. Least Confidence
    out["uncertainty_least_conf"] = 1.0 - np.maximum(probas, 1.0 - probas)

    return out


# ---------------------------------------------------------------------------
# 2. Örnekleme Stratejileri
# ---------------------------------------------------------------------------

def sample_uncertain_pairs(
    df: pd.DataFrame,
    n_samples: int = 1000,
    strategy: Literal["entropy", "margin", "least_conf", "hybrid"] = "entropy",
    proba_col: str = "proba",
    query_col: str = "term_id",
    max_per_query: Optional[int] = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Aktif öğrenme için en yüksek belirsizliğe sahip sorgu-ürün çiftlerini seçer.

    Parameters
    ----------
    df : pd.DataFrame
        Olasılık veya belirsizlik sütunlarını içeren DataFrame.
    n_samples : int
        Seçilecek toplam örnek sayısı.
    strategy : str
        'entropy', 'margin', 'least_conf' veya 'hybrid'.
    max_per_query : int, optional
        Tek bir sorgudan en fazla kaç örnek seçilebileceği (çeşitlilik garantisi).

    Returns
    -------
    pd.DataFrame
        Seçilen örneklerin DataFrame'i.
    """
    if df.empty:
        return df

    # Belirsizlik kolonları yoksa hesapla
    if f"uncertainty_{strategy if strategy != 'hybrid' else 'entropy'}" not in df.columns:
        df = compute_uncertainty_scores(df, proba_col=proba_col, copy=True)

    if strategy == "entropy":
        score_col = "uncertainty_entropy"
    elif strategy == "margin":
        score_col = "uncertainty_margin"
    elif strategy == "least_conf":
        score_col = "uncertainty_least_conf"
    elif strategy == "hybrid":
        # Entropi ve margin ortalaması
        df["_hybrid_score"] = (df["uncertainty_entropy"] + df["uncertainty_margin"]) / 2.0
        score_col = "_hybrid_score"
    else:
        raise ValueError(f"Bilinmeyen strateji: {strategy}")

    # En yüksek belirsizliğe göre sırala
    sorted_df = df.sort_values(score_col, ascending=False)

    if max_per_query is not None and query_col in sorted_df.columns:
        # Sorgu başına sınır koy
        selected_rows = []
        counts: dict[str, int] = {}
        for _, row in sorted_df.iterrows():
            qid = str(row[query_col])
            current = counts.get(qid, 0)
            if current < max_per_query:
                selected_rows.append(row)
                counts[qid] = current + 1
                if len(selected_rows) >= n_samples:
                    break
        result = pd.DataFrame(selected_rows)
    else:
        result = sorted_df.head(n_samples)

    if "_hybrid_score" in result.columns:
        result = result.drop(columns=["_hybrid_score"])

    return result.reset_index(drop=True)
