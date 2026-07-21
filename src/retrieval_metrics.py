"""
src/retrieval_metrics.py
========================
G.G.A Takımı — Retrieval Evaluation Metrics

Arama/retrieval kalitesini ölçen standart IR metrikleri.
Mevcut src/metrics.py (Macro-F1 / threshold) ile birlikte kullanılır;
bu modül retrieval-specific metrikleri kapsıyor.

Metrikler:
  - Recall@K      : Her query için top-K'da gerçek pozitif bulma oranı
  - Precision@K   : Her query için top-K'ın ne kadarı gerçek pozitif?
  - NDCG@K        : Normalized Discounted Cumulative Gain (sıralama kalitesi)
  - MRR           : Mean Reciprocal Rank (ilk pozitifin sırası)
  - evaluation_report : Tüm metrikleri tek çağrıda döndürür

Kullanım:
  >>> report = evaluation_report(df, ks=[1, 5, 10, 20, 50, 100])
  >>> print(report["recall@10"], report["ndcg@10"])
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. İç Yardımcılar
# ---------------------------------------------------------------------------

def _group_by_query(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    query_col: str = "term_id",
    item_col: str = "item_id",
) -> tuple[dict, dict]:
    """
    DataFrame'i query gruplarına böler.

    Returns
    -------
    positives_per_query : dict[str, set]
        {term_id: {item_id, ...}} — gerçek pozitifler
    ranked_per_query : dict[str, list[str]]
        {term_id: [item_id, ...]} — skora göre azalan sırada
    """
    required = {query_col, item_col, score_col, label_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"retrieval_metrics: eksik sütunlar: {sorted(missing)}")
    if df.empty:
        raise ValueError("retrieval_metrics: boş DataFrame")
    if not np.isin(df[label_col].to_numpy(), [0, 1]).all():
        raise ValueError(f"'{label_col}' sadece 0 ve 1 içermelidir")
    if not np.isfinite(df[score_col].to_numpy()).all():
        raise ValueError(f"'{score_col}' sonsuz veya NaN değer içeriyor")

    positives: dict[str, set] = defaultdict(set)
    ranked: dict[str, list] = {}

    for qid, group in df.groupby(query_col, sort=False):
        pos_mask = group[label_col].astype(int) == 1
        positives[qid] = set(group.loc[pos_mask, item_col].astype(str).tolist())
        ranked[qid] = (
            group.sort_values(score_col, ascending=False)[item_col]
            .astype(str)
            .tolist()
        )

    return dict(positives), ranked


# ---------------------------------------------------------------------------
# 2. Recall@K
# ---------------------------------------------------------------------------

def recall_at_k(
    df: pd.DataFrame,
    k: int,
    score_col: str = "score",
    label_col: str = "label",
    query_col: str = "term_id",
    item_col: str = "item_id",
) -> float:
    """
    Macro-averaged Recall@K.

    Her query için: top-K içinde bulunan pozitif sayısı / toplam pozitif sayısı.
    Sıfır pozitifli queryler atlanır.

    Parameters
    ----------
    df : pd.DataFrame
        term_id, item_id, score, label kolonları içermeli.
    k : int
        Kaç aday değerlendirilecek.

    Returns
    -------
    float
        Macro-averaged Recall@K (0.0 – 1.0).
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k pozitif tam sayı olmalı, alınan: {k!r}")

    positives, ranked = _group_by_query(df, score_col, label_col, query_col, item_col)
    scores = []
    for qid, pos_set in positives.items():
        if not pos_set:
            continue
        top_k_items = set(ranked[qid][:k])
        scores.append(len(pos_set & top_k_items) / len(pos_set))

    if not scores:
        return 0.0
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 3. Precision@K
# ---------------------------------------------------------------------------

def precision_at_k(
    df: pd.DataFrame,
    k: int,
    score_col: str = "score",
    label_col: str = "label",
    query_col: str = "term_id",
    item_col: str = "item_id",
) -> float:
    """
    Macro-averaged Precision@K.

    Her query için: top-K içinde bulunan pozitif sayısı / K.

    Returns
    -------
    float
        Macro-averaged Precision@K (0.0 – 1.0).
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k pozitif tam sayı olmalı, alınan: {k!r}")

    positives, ranked = _group_by_query(df, score_col, label_col, query_col, item_col)
    scores = []
    for qid, pos_set in positives.items():
        top_k_items = ranked[qid][:k]
        hits = sum(1 for item in top_k_items if item in pos_set)
        scores.append(hits / k)

    if not scores:
        return 0.0
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 4. NDCG@K
# ---------------------------------------------------------------------------

def ndcg_at_k(
    df: pd.DataFrame,
    k: int,
    score_col: str = "score",
    label_col: str = "label",
    query_col: str = "term_id",
    item_col: str = "item_id",
) -> float:
    """
    Macro-averaged NDCG@K (Normalized Discounted Cumulative Gain).

    Binary relevance (0/1). İdeal sıralama: tüm pozitifler en üstte.
    Sıfır pozitifli queryler atlanır.

    Returns
    -------
    float
        Macro-averaged NDCG@K (0.0 – 1.0).
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k pozitif tam sayı olmalı, alınan: {k!r}")

    positives, ranked = _group_by_query(df, score_col, label_col, query_col, item_col)
    scores = []
    for qid, pos_set in positives.items():
        if not pos_set:
            continue

        # Gerçek relevance vektörü (top-K uzunluğunda)
        top_k = ranked[qid][:k]
        gains = np.array([1.0 if item in pos_set else 0.0 for item in top_k])

        # DCG
        positions = np.arange(1, len(gains) + 1, dtype=np.float64)
        dcg = float(np.sum(gains / np.log2(positions + 1)))

        # İdeal DCG
        ideal_k = min(k, len(pos_set))
        ideal_gains = np.ones(ideal_k)
        ideal_positions = np.arange(1, ideal_k + 1, dtype=np.float64)
        idcg = float(np.sum(ideal_gains / np.log2(ideal_positions + 1)))

        scores.append(dcg / idcg if idcg > 0 else 0.0)

    if not scores:
        return 0.0
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 5. Mean Reciprocal Rank (MRR)
# ---------------------------------------------------------------------------

def mean_reciprocal_rank(
    df: pd.DataFrame,
    score_col: str = "score",
    label_col: str = "label",
    query_col: str = "term_id",
    item_col: str = "item_id",
) -> float:
    """
    Mean Reciprocal Rank (MRR).

    Her query için: ilk pozitif sonucun 1 / rank değeri.
    Pozitif hiç bulunamazsa 0.0.

    Returns
    -------
    float
        Macro-averaged MRR (0.0 – 1.0).
    """
    positives, ranked = _group_by_query(df, score_col, label_col, query_col, item_col)
    scores = []
    for qid, pos_set in positives.items():
        rr = 0.0
        for rank, item in enumerate(ranked[qid], start=1):
            if item in pos_set:
                rr = 1.0 / rank
                break
        scores.append(rr)

    if not scores:
        return 0.0
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 6. evaluation_report — Ana Giriş Noktası
# ---------------------------------------------------------------------------

def evaluation_report(
    df: pd.DataFrame,
    ks: Optional[List[int]] = None,
    score_col: str = "score",
    label_col: str = "label",
    query_col: str = "term_id",
    item_col: str = "item_id",
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Tüm retrieval metriklerini tek çağrıda hesaplar.

    Parameters
    ----------
    df : pd.DataFrame
        term_id, item_id, score, label kolonları içermeli.
    ks : list of int, optional
        Değerlendirilecek K değerleri. Varsayılan: [1, 5, 10, 20, 50, 100].
    verbose : bool
        Sonuçları stdout'a yazdır.

    Returns
    -------
    dict
        Anahtar formatı: "recall@K", "precision@K", "ndcg@K", "mrr"
        Tüm değerler 0.0 – 1.0 arasında float.

    Örnek
    -----
    >>> report = evaluation_report(df, ks=[10, 50, 100])
    >>> report["recall@10"]
    0.7823
    >>> report["ndcg@10"]
    0.6541
    """
    if ks is None:
        ks = [1, 5, 10, 20, 50, 100]

    # Tekrarlı gruplama işlemini tek seferlik yap
    positives, ranked = _group_by_query(df, score_col, label_col, query_col, item_col)
    n_queries = len(positives)

    def _recall(k: int) -> float:
        sc = []
        for qid, pos_set in positives.items():
            if not pos_set:
                continue
            sc.append(len(pos_set & set(ranked[qid][:k])) / len(pos_set))
        return float(np.mean(sc)) if sc else 0.0

    def _precision(k: int) -> float:
        sc = []
        for qid, pos_set in positives.items():
            hits = sum(1 for item in ranked[qid][:k] if item in pos_set)
            sc.append(hits / k)
        return float(np.mean(sc)) if sc else 0.0

    def _ndcg(k: int) -> float:
        sc = []
        for qid, pos_set in positives.items():
            if not pos_set:
                continue
            top_k = ranked[qid][:k]
            gains = np.array([1.0 if item in pos_set else 0.0 for item in top_k])
            positions = np.arange(1, len(gains) + 1, dtype=np.float64)
            dcg = float(np.sum(gains / np.log2(positions + 1)))
            ideal_k = min(k, len(pos_set))
            ideal_pos = np.arange(1, ideal_k + 1, dtype=np.float64)
            idcg = float(np.sum(np.ones(ideal_k) / np.log2(ideal_pos + 1)))
            sc.append(dcg / idcg if idcg > 0 else 0.0)
        return float(np.mean(sc)) if sc else 0.0

    def _mrr() -> float:
        sc = []
        for qid, pos_set in positives.items():
            rr = 0.0
            for rank, item in enumerate(ranked[qid], start=1):
                if item in pos_set:
                    rr = 1.0 / rank
                    break
            sc.append(rr)
        return float(np.mean(sc)) if sc else 0.0

    report: Dict[str, float] = {}
    for k in sorted(ks):
        report[f"recall@{k}"] = _recall(k)
        report[f"precision@{k}"] = _precision(k)
        report[f"ndcg@{k}"] = _ndcg(k)

    report["mrr"] = _mrr()
    report["n_queries"] = float(n_queries)

    if verbose:
        print("\n── Retrieval Evaluation Report ─────────────────────────")
        print(f"  Sorgu sayısı : {n_queries:,}")
        print(f"  {'K':>6}  {'Recall':>8}  {'Precision':>10}  {'NDCG':>8}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*8}")
        for k in sorted(ks):
            print(
                f"  {k:>6}  "
                f"{report[f'recall@{k}']:>8.4f}  "
                f"{report[f'precision@{k}']:>10.4f}  "
                f"{report[f'ndcg@{k}']:>8.4f}"
            )
        print(f"  {'MRR':>6}  {report['mrr']:>8.4f}")
        print("─────────────────────────────────────────────────────────\n")

    return report


# ---------------------------------------------------------------------------
# 7. DataFrame Builder (OOF sonuçlarından kolayca çağırılabilir)
# ---------------------------------------------------------------------------

def build_eval_dataframe(
    test_metadata: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    query_col: str = "term_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """
    Mevcut test_metadata.csv + tahmin skoru array'inden eval DataFrame'i üret.

    Parameters
    ----------
    test_metadata : pd.DataFrame
        En az term_id ve item_id kolonlarını içermeli.
    scores : np.ndarray of float
        Model tahmin skorları (olasılık veya raw score).
    labels : np.ndarray of int
        Gerçek etiketler (0 veya 1).

    Returns
    -------
    pd.DataFrame
        term_id, item_id, score, label kolonlarıyla.
    """
    if len(test_metadata) != len(scores) or len(test_metadata) != len(labels):
        raise ValueError(
            "test_metadata, scores ve labels aynı uzunlukta olmalı"
        )
    out = test_metadata[[query_col, item_col]].copy()
    out["score"] = np.asarray(scores, dtype=np.float64)
    out["label"] = np.asarray(labels, dtype=np.int8)
    return out.reset_index(drop=True)
