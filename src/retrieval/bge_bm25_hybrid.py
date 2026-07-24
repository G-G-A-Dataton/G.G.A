"""
src/retrieval/bge_bm25_hybrid.py
=================================
G.G.A Takımı — Hybrid Retrieval (BM25 + BAAI/bge-m3) & Reciprocal Rank Fusion (RRF) Modülü

Türkçe E-Ticaret Arama Mimarisi:
1. Türkçe Duyarlı Metin Ön İşleme (tr_lower + clean_text)
2. Arama Metni Birleştirme ({title} {category} {brand} {attributes})
3. BM25 Sparse Retrieval (Native Okapi BM25 / rank_bm25, k1=1.2, b=0.75)
4. Dense Vector Retrieval (SentenceTransformer BAAI/bge-m3)
5. Reciprocal Rank Fusion (RRF, k=60, top_n=100)
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Union

# Optional rank_bm25 dependency
try:
    from rank_bm25 import BM25Okapi
    HAS_RANK_BM25 = True
except ImportError:
    BM25Okapi = None
    HAS_RANK_BM25 = False

# Sentence Transformers Dependency
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = None
    HAS_SENTENCE_TRANSFORMERS = False


# =============================================================================
# 1. Data Preprocessing (Türkçe Karakter Duyarlı Ön İşleme)
# =============================================================================

def tr_lower(text: str) -> str:
    """
    Türkçe karakter duyarlı küçük harfe çevirme.
    'İ' -> 'i', 'I' -> 'ı' dönüşüm kurallarını uygular.
    """
    if not text:
        return ""
    translation_table = str.maketrans({"İ": "i", "I": "ı"})
    return text.translate(translation_table).lower()


def clean_text(text: str) -> str:
    """
    Noktalama işaretlerini ve fazla boşlukları temizler.
    Metni Türkçe kurallara göre küçük harfe çevirir.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Türkçe küçük harfe çevir
    text = tr_lower(text)
    
    # Noktalama işaretlerini boşlukla değiştir
    punct_regex = re.compile(f"[{re.escape(string.punctuation)}]")
    text = punct_regex.sub(" ", text)
    
    # Birden fazla boşluğu tek boşluğa indirge ve kenar boşluklarını sil
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_search_text(row: pd.Series | dict) -> str:
    """
    Ürün objesini şu formatta birleştirilmiş bir search_text alanına dönüştürür:
    {title} {category} {brand} {attributes}
    """
    if isinstance(row, pd.Series):
        title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
        category = str(row.get("category", "")) if pd.notna(row.get("category")) else ""
        brand = str(row.get("brand", "")) if pd.notna(row.get("brand")) else ""
        attributes = str(row.get("attributes", "")) if pd.notna(row.get("attributes")) else ""
    elif isinstance(row, dict):
        title = str(row.get("title", ""))
        category = str(row.get("category", ""))
        brand = str(row.get("brand", ""))
        attributes = str(row.get("attributes", ""))
    else:
        return ""

    raw_combined = f"{title} {category} {brand} {attributes}"
    return clean_text(raw_combined)


# =============================================================================
# 2. Retrieval Engines (BM25 Engine + Dense BGE-M3 Engine)
# =============================================================================

class NativeOkapiBM25:
    """
    Sıfır bağımlılıklı Okapi BM25 Gerçekleştirmesi.
    k1 = 1.2, b = 0.75
    """
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.avgdl = sum(len(doc) for doc in corpus_tokens) / self.corpus_size if self.corpus_size > 0 else 1.0
        self.doc_freqs: list[Counter] = [Counter(doc) for doc in corpus_tokens]
        self.doc_lens: np.ndarray = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        
        # Calculate IDF
        df_counts: Counter = Counter()
        for doc in corpus_tokens:
            for word in set(doc):
                df_counts[word] += 1
                
        self.idf: dict[str, float] = {}
        for word, freq in df_counts.items():
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i, doc_freq in enumerate(self.doc_freqs):
                freq = doc_freq.get(token, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                scores[i] += idf_val * (freq * (self.k1 + 1.0)) / denom
        return scores


class BM25Engine:
    """
    rank_bm25 veya Native Okapi BM25 tabanlı Sparse Arama Motoru.
    k1 = 1.2, b = 0.75 olarak yapılandırılmıştır.
    """
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.item_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []

    def index(self, items_df: pd.DataFrame) -> None:
        """
        Ürün DataFrame'ini alarak BM25 indeksini oluşturur.
        """
        self.item_ids = items_df["item_id"].astype(str).tolist()
        self.corpus_tokens = []
        
        for _, row in items_df.iterrows():
            stext = build_search_text(row)
            self.corpus_tokens.append(stext.split())

        if HAS_RANK_BM25:
            self.bm25 = BM25Okapi(self.corpus_tokens, k1=self.k1, b=self.b)
        else:
            self.bm25 = NativeOkapiBM25(self.corpus_tokens, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        """
        Sorgu için en yüksek skorlu top_k ürün ID'sini ve skorunu döndürür.
        """
        if self.bm25 is None:
            raise ValueError("Arama yapmadan önce index() metodu çağrılmalıdır.")

        cleaned_query = clean_text(query)
        tokenized_query = cleaned_query.split()
        
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                results.append((self.item_ids[idx], score))
        return results


class DenseBGEEngine:
    """
    BAAI/bge-m3 (SentenceTransformer) tabanlı Dense Vector Arama Motoru.
    """
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str | None = None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.item_ids: list[str] = []
        self.embeddings: np.ndarray | None = None

    def initialize_model(self) -> None:
        if self.model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError("sentence_transformers yüklü değil: pip install sentence-transformers")
            print(f"[+] BAAI/bge-m3 modeli yükleniyor ({self.model_name})...")
            self.model = SentenceTransformer(self.model_name, device=self.device)

    def index(self, items_df: pd.DataFrame, batch_size: int = 64) -> None:
        """
        Ürün metinlerinin dense embedding'lerini üretir ve saklar.
        """
        self.initialize_model()
        self.item_ids = items_df["item_id"].astype(str).tolist()
        search_texts = [build_search_text(row) for _, row in items_df.iterrows()]

        print(f"[+] {len(search_texts):,} ürün için dense embedding hesaplanıyor...")
        embeddings = self.model.encode(
            search_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        self.embeddings = embeddings.astype(np.float32)

    def set_precomputed_embeddings(self, item_ids: list[str], embeddings: np.ndarray) -> None:
        """
        Önceden hesaplanmış embedding'leri doğrudan atar.
        """
        self.item_ids = [str(i) for i in item_ids]
        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = (embeddings / norms).astype(np.float32)

    def search(self, query: str, top_k: int = 200) -> list[tuple[str, float]]:
        """
        Sorgu embedding'ini üretip kosinüs benzerliğine göre top_k ürünü döndürür.
        """
        if self.embeddings is None:
            raise ValueError("Arama yapmadan önce index() veya set_precomputed_embeddings() çağrılmalıdır.")

        self.initialize_model()
        cleaned_query = clean_text(query)
        q_emb = self.model.encode(
            [cleaned_query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        # Dot product (L2 normalized vectors = Cosine Similarity)
        scores = np.dot(self.embeddings, q_emb.T).squeeze(axis=1)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(self.item_ids[idx], float(scores[idx])) for idx in top_indices]
        return results


# =============================================================================
# 3. Reciprocal Rank Fusion (RRF) Algorithm
# =============================================================================

def reciprocal_rank_fusion(
    bm25_results: list[Union[str, int]] | list[tuple[Union[str, int], float]],
    dense_results: list[Union[str, int]] | list[tuple[Union[str, int], float]],
    k: int = 60,
    top_n: int = 100
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) Algoritması.
    
    Formül:
        RRF(d) = sum( 1 / (k + r(d)) )
        
    Parametreler:
    ------------
    bm25_results : List[str/int] veya List[Tuple[str/int, score]]
        BM25 sıralı ürün ID listesi.
    dense_results : List[str/int] veya List[Tuple[str/int, score]]
        Dense Vector sıralı ürün ID listesi.
    k : int, default=60
        RRF katsayısı (varsayılan 60).
    top_n : int, default=100
        Döndürülecek en yüksek skorlu benzersiz ürün sayısı.
        
    Döndürür:
    --------
    List[Tuple[str, float]]
        Azalan sırada sıralanmış (item_id, rrf_score) çiftleri.
    """
    rrf_scores: dict[str, float] = {}

    # BM25 sıralarını işle
    for rank, item in enumerate(bm25_results, 1):
        item_id = str(item[0]) if isinstance(item, tuple) else str(item)
        if item_id not in rrf_scores:
            rrf_scores[item_id] = 0.0
        rrf_scores[item_id] += 1.0 / (k + rank)

    # Dense sıralarını işle
    for rank, item in enumerate(dense_results, 1):
        item_id = str(item[0]) if isinstance(item, tuple) else str(item)
        if item_id not in rrf_scores:
            rrf_scores[item_id] = 0.0
        rrf_scores[item_id] += 1.0 / (k + rank)

    # RRF skoruna göre azalan sırada sırala
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Top N benzersiz ürünü döndür
    return sorted_rrf[:top_n]


# =============================================================================
# 4. End-to-End Hybrid Search Pipeline
# =============================================================================

class HybridRetrievalPipeline:
    """
    BM25 ve Dense BGE-M3 arama motorlarını RRF ile birleştiren uçtan uca hibrit arama sınıfı.
    """
    def __init__(self, bm25_engine: BM25Engine, dense_engine: DenseBGEEngine):
        self.bm25_engine = bm25_engine
        self.dense_engine = dense_engine

    def search(
        self,
        query: str,
        k_rrf: int = 60,
        top_n: int = 100,
        candidate_k: int = 200
    ) -> list[tuple[str, float]]:
        """
        Sorgu için BM25 ve Dense aramaları gerçekleştirir, RRF ile birleştirip 
        100 adaylık filtrelenmiş ve sıralanmış (item_id, rrf_score) listesini döndürür.
        """
        bm25_candidates = self.bm25_engine.search(query, top_k=candidate_k)
        dense_candidates = self.dense_engine.search(query, top_k=candidate_k)

        final_results = reciprocal_rank_fusion(
            bm25_results=bm25_candidates,
            dense_results=dense_candidates,
            k=k_rrf,
            top_n=top_n
        )
        return final_results
