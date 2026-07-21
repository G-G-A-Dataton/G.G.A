"""
src/hard_negative_pool.py
=========================
G.G.A Takımı — Human-Verified Hard Negative Pool Manager

Zorlu negatifleri (BM25 hard negatives, high-confidence False Positives)
ve insan doğrulamalarını tutan, sürümleyen ve veri setine enjekte eden modül.

Durumlar (Status):
  - UNVERIFIED         : Otomatik üretilmiş hard negative aday (örn. BM25 rank top 10)
  - VERIFIED_NEGATIVE  : İnsan tarafından onaylanmış GERÇEK negatif (label = 0)
  - FALSE_NEGATIVE_POS : İnsan tarafından fark edilmiş aslında POZİTİF olan çift (label = 1)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


class HardNegativePool:
    """
    Hard negative havuzu yöneticisi.

    Attributes
    ----------
    pool_path : str
        Havuz veri seti yolu (.parquet veya .csv).
    """

    STATUS_UNVERIFIED = "UNVERIFIED"
    STATUS_VERIFIED_NEGATIVE = "VERIFIED_NEGATIVE"
    STATUS_FALSE_NEGATIVE_POS = "FALSE_NEGATIVE_POS"

    def __init__(self, pool_path: str = "datasets/hard_negative_pool.parquet"):
        self.pool_path = pool_path
        self._df: pd.DataFrame = self._load_or_create()

    def _load_or_create(self) -> pd.DataFrame:
        if os.path.exists(self.pool_path):
            try:
                if self.pool_path.endswith(".parquet"):
                    return pd.read_parquet(self.pool_path)
                return pd.read_csv(self.pool_path)
            except ImportError:
                csv_fallback = self.pool_path.replace(".parquet", ".csv")
                if os.path.exists(csv_fallback):
                    return pd.read_csv(csv_fallback)
                raise
        csv_fallback = self.pool_path.replace(".parquet", ".csv")
        if os.path.exists(csv_fallback):
            return pd.read_csv(csv_fallback)

        return pd.DataFrame(columns=[
            "term_id", "item_id", "source", "status",
            "added_at", "verified_at", "annotator_id", "notes"
        ])

    def add_candidates(
        self,
        candidates_df: pd.DataFrame,
        source: str = "bm25",
        status: str = STATUS_UNVERIFIED,
    ) -> int:
        """
        Yeni aday hard negatifleri havuza ekler (varsa günceller).

        Parameters
        ----------
        candidates_df : pd.DataFrame
            En az 'term_id' ve 'item_id' kolonları olmalı.
        source : str
            Aday kaynağı ("bm25", "faiss_dense", "model_fp").
        status : str
            Başlangıç durumu.

        Returns
        -------
        int
            Eklenen veya güncellenen kayıt sayısı.
        """
        if candidates_df.empty:
            return 0

        new_rows = candidates_df[["term_id", "item_id"]].copy()
        new_rows["term_id"] = new_rows["term_id"].astype(str)
        new_rows["item_id"] = new_rows["item_id"].astype(str)
        new_rows["source"] = source
        new_rows["status"] = status
        new_rows["added_at"] = datetime.now(tz=timezone.utc).isoformat()
        new_rows["verified_at"] = None
        new_rows["annotator_id"] = None
        new_rows["notes"] = None

        # Tekilleştirme (term_id, item_id bazlı)
        combined = pd.concat([self._df, new_rows], ignore_index=True)
        # Var olan onaylı kayıtların status'ünü koru
        combined = combined.drop_duplicates(subset=["term_id", "item_id"], keep="first")
        added_count = len(combined) - len(self._df)
        self._df = combined
        return added_count

    def update_verification(
        self,
        term_id: str,
        item_id: str,
        status: str,
        annotator_id: str = "human",
        notes: Optional[str] = None,
    ) -> bool:
        """Tekil bir çiftin doğrulama durumunu güncelle."""
        mask = (self._df["term_id"].astype(str) == str(term_id)) & (self._df["item_id"].astype(str) == str(item_id))
        if not mask.any():
            return False

        idx = self._df[mask].index[0]
        self._df.loc[idx, "status"] = status
        self._df.loc[idx, "verified_at"] = datetime.now(tz=timezone.utc).isoformat()
        self._df.loc[idx, "annotator_id"] = annotator_id
        if notes:
            self._df.loc[idx, "notes"] = notes
        return True

    def get_verified_negatives(self) -> pd.DataFrame:
        """Yalnızca doğrulanmış gerçek negatifleri döndür."""
        return self._df[self._df["status"] == self.STATUS_VERIFIED_NEGATIVE].copy()

    def get_false_negatives(self) -> pd.DataFrame:
        """İnsan tarafından pozitif olduğu tespit edilen yanlış negatifleri döndür."""
        return self._df[self._df["status"] == self.STATUS_FALSE_NEGATIVE_POS].copy()

    def save(self) -> None:
        """Havuzu diske kaydet."""
        os.makedirs(os.path.dirname(self.pool_path) or ".", exist_ok=True)
        if self.pool_path.endswith(".parquet"):
            try:
                self._df.to_parquet(self.pool_path, index=False)
            except ImportError:
                csv_fallback = self.pool_path.replace(".parquet", ".csv")
                self._df.to_csv(csv_fallback, index=False)
        else:
            self._df.to_csv(self.pool_path, index=False)

    def summary(self) -> Dict[str, Any]:
        """Havuz durum özeti."""
        return {
            "total_pairs": len(self._df),
            "by_status": self._df["status"].value_counts().to_dict(),
            "by_source": self._df["source"].value_counts().to_dict(),
        }
