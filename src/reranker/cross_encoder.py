"""
src/reranker/cross_encoder.py
=============================
G.G.A Takımı — Cross-Encoder Reranker

İkinci aşama reranking için Cross-Encoder modeli sarmalayıcısı.
Sorgu metni ve Ürün başlığını/detayını birlikte encode ederek (joint attention)
daha hassas eşleşme skoru üretir.

Kullanım:
  >>> from src.reranker.cross_encoder import ProductReranker
  >>> reranker = ProductReranker("BAAI/bge-reranker-base")
  >>> scores = reranker.predict([("siyah ayakkabı", "Nike Siyah Kadın Spor Ayakkabı")])
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import pandas as pd

try:
    from sentence_transformers import CrossEncoder  # type: ignore
    HAS_CROSS_ENCODER = True
except ImportError:
    CrossEncoder = None
    HAS_CROSS_ENCODER = False


class ProductReranker:
    """
    Cross-Encoder Reranker sarmalayıcısı.

    Parameters
    ----------
    model_name : str
        HuggingFace model adı (örn. 'BAAI/bge-reranker-base' veya 'cross-encoder/ms-marco-MiniLM-L-6-v2').
    max_length : int
        Maksimum token uzunluğu.
    device : str, optional
        'cuda', 'cpu' veya None.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 256,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.model = None

    def _load_model(self):
        if self.model is None:
            if not HAS_CROSS_ENCODER:
                raise ImportError(
                    "sentence-transformers yüklü değil: pip install sentence-transformers"
                )
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )

    def predict(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        (query, item_text) çiftleri için reranking skorlarını hesapla.

        Parameters
        ----------
        pairs : list of (query_text, item_text)
        batch_size : int

        Returns
        -------
        np.ndarray of float32
            Reranking skorları (0.0 - 1.0 arası sigmoid uygulanmış veya raw).
        """
        if not pairs:
            return np.array([], dtype=np.float32)

        self._load_model()
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(scores, dtype=np.float32)

    def rerank_candidates(
        self,
        candidates_df: pd.DataFrame,
        query_col: str = "query_text",
        item_col: str = "item_title",
        top_k: int = 100,
        batch_size: int = 32,
    ) -> pd.DataFrame:
        """
        DataFrame'deki adayları Cross-Encoder ile yeniden sıralar.

        Parameters
        ----------
        candidates_df : pd.DataFrame
            query_text ve item_title sütunlarını içermeli.
        top_k : int
            Rerank edildikten sonra tutulacak en üst aday sayısı.

        Returns
        -------
        pd.DataFrame
            'rerank_score' ve 'rerank_rank' kolonları eklenmiş, azalan sırada.
        """
        if candidates_df.empty:
            return candidates_df

        pairs = list(zip(candidates_df[query_col].astype(str), candidates_df[item_col].astype(str)))
        scores = self.predict(pairs, batch_size=batch_size)

        out = candidates_df.copy()
        out["rerank_score"] = scores
        out = out.sort_values("rerank_score", ascending=False).head(top_k)
        out["rerank_rank"] = np.arange(1, len(out) + 1)
        return out.reset_index(drop=True)
