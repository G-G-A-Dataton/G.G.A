"""
src/retrieval/bm25.py
=====================
G.G.A Takımı — Unified BM25 Retrieval & Hard Negative Engine

Tekil BM25 Inverted Index ve Retriever implementasyonu.
Hem hard negative üretimi hem de hibrit arama (retrieval) için tek kaynak.

Kullanım:
  >>> from src.retrieval.bm25 import BM25Index, BM25Retriever
  >>> retriever = BM25Retriever(items_df)
  >>> results = retriever.retrieve("siyah ayakkabı", k=50)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.item_text import build_item_texts, clean_text


def _tokenize(text: str) -> List[str]:
    """Metni normalize et ve token'lara böl."""
    return clean_text(text).split()


class BM25Index:
    """
    BM25 Inverted Index — ~1M ürün kataloğu için performanslı Python içi indeks.
    """

    def __init__(
        self,
        corpus: List[str],
        doc_ids: List[str],
        k1: float = 1.5,
        b: float = 0.75,
        max_df_ratio: float = 0.15,
        epsilon: float = 0.25,
    ):
        self.doc_ids = list(doc_ids)
        self.k1 = k1
        self.b = b
        self.max_df_ratio = max_df_ratio
        self.epsilon = epsilon

        self.N = len(corpus)
        self.doc_lens = np.array([len(_tokenize(doc)) for doc in corpus], dtype=np.int32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.N > 0 else 0.0

        # Inverted index: term -> {doc_idx: term_frequency}
        self.inverted_index: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.idf: Dict[str, float] = {}

        self._build_index(corpus)

    def _build_index(self, corpus: List[str]) -> None:
        max_df = int(self.N * self.max_df_ratio)

        for doc_idx, doc in enumerate(corpus):
            tokens = _tokenize(doc)
            freqs: Dict[str, int] = defaultdict(int)
            for token in tokens:
                freqs[token] += 1
            for token, freq in freqs.items():
                self.inverted_index[token][doc_idx] = freq

        # Compute IDF
        total_idf = 0.0
        negative_idfs = []
        for token, doc_map in self.inverted_index.items():
            df = len(doc_map)
            if df > max_df:
                continue
            idf_val = math.log(self.N - df + 0.5) - math.log(df + 0.5)
            self.idf[token] = idf_val
            total_idf += idf_val
            if idf_val < 0:
                negative_idfs.append(token)

        avg_idf = total_idf / len(self.idf) if self.idf else 0.0
        eps_idf = self.epsilon * avg_idf
        for token in negative_idfs:
            self.idf[token] = eps_idf

    def top_n(self, query: str, n: int = 50) -> List[str]:
        """Sorgu için top-N ürün ID'lerini azalan skorda döndürür."""
        q_tokens = _tokenize(query)
        scores: Dict[int, float] = defaultdict(float)

        for token in q_tokens:
            if token not in self.idf:
                continue
            idf = self.idf[token]
            doc_map = self.inverted_index[token]

            for doc_idx, tf in doc_map.items():
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                num = tf * (self.k1 + 1.0)
                scores[doc_idx] += idf * (num / denom)

        if not scores:
            return []

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [self.doc_ids[doc_idx] for doc_idx, _ in sorted_docs]


class BM25Retriever:
    """
    BM25 Standard Retrieval Interface.
    """

    def __init__(self, items_df: pd.DataFrame, max_df_ratio: float = 0.3):
        item_texts = build_item_texts(items_df)
        item_ids = items_df["item_id"].astype(str).tolist()
        self.index = BM25Index(item_texts, item_ids, max_df_ratio=max_df_ratio)

    def retrieve(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        """Top-k sonuç ve min-max normalize edilmiş skorlar."""
        top_ids = self.index.top_n(query, n=k)
        if not top_ids:
            return []
        # Linear decay normalized scores for downstream rank fusion
        return [(item_id, (k - rank) / k) for rank, item_id in enumerate(top_ids)]

    def retrieve_batch(
        self, queries: List[Tuple[str, str]], k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Queries: [(term_id, query_text), ...]"""
        res = {}
        for term_id, query_text in queries:
            res[term_id] = self.retrieve(query_text, k=k)
        return res

    def save(self, path: str) -> None:
        """İndeksi pickle dosyası olarak kaydeder."""
        import pickle
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        """Kaydedilmiş pickle indeksini yükler."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
