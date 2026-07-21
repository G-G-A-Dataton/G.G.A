"""
src/monitoring/drift_detector.py
=================================
G.G.A Takımı — Feature & Prediction Drift Detection

Referans veri kümesi (eğitim/OOF) ile güncel çıkarım dağılımlarını karşılaştırarak
PSI (Population Stability Index) ve Kolmogorov-Smirnov (KS) testleri ile drift tespiti yapar.

Kullanım:
  >>> from src.monitoring.drift_detector import DriftDetector
  >>> detector = DriftDetector(ref_df)
  >>> report = detector.detect_drift(current_df)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Population Stability Index (PSI) hesaplar.

    PSI < 0.1  : Değişim Yok (No Drift)
    0.1 <= PSI < 0.25 : Orta Seviye Kayma (Moderate Drift)
    PSI >= 0.25 : Belirgin Kayma / Model Yeniden Eğitim Gerekli (Significant Drift)
    """
    ref_arr = np.asarray(reference, dtype=np.float64)
    curr_arr = np.asarray(current, dtype=np.float64)

    # Bin sınırlarını referans veriye göre ayarla
    quantiles = np.linspace(0, 100, n_bins + 1)
    bins = np.percentile(ref_arr, quantiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    ref_counts, _ = np.histogram(ref_arr, bins=bins)
    curr_counts, _ = np.histogram(curr_arr, bins=bins)

    ref_pct = (ref_counts + eps) / (len(ref_arr) + eps * len(ref_counts))
    curr_pct = (curr_counts + eps) / (len(curr_arr) + eps * len(curr_counts))

    psi_val = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return float(psi_val)


def compute_ks_test(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (ks_stat, p_value)."""
    res = stats.ks_2samp(reference, current)
    return float(res.statistic), float(res.pvalue)


class DriftDetector:
    """Feature and prediction distribution drift detector."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        psi_threshold: float = 0.1,
        ks_alpha: float = 0.05,
    ):
        self.reference_df = reference_df
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha

    def detect_drift(
        self,
        current_df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Belirtilen kolonlar için PSI ve KS-test drift analizi yapar.
        """
        target_cols = columns or [c for c in self.reference_df.columns if np.issubdtype(self.reference_df[c].dtype, np.number)]
        report: Dict[str, Any] = {"features": {}, "has_significant_drift": False}

        for col in target_cols:
            if col not in current_df.columns or col not in self.reference_df.columns:
                continue

            ref_vals = self.reference_df[col].dropna().to_numpy()
            curr_vals = current_df[col].dropna().to_numpy()

            if len(ref_vals) == 0 or len(curr_vals) == 0:
                continue

            psi_val = compute_psi(ref_vals, curr_vals)
            ks_stat, p_val = compute_ks_test(ref_vals, curr_vals)

            is_drift = psi_val >= self.psi_threshold or p_val < self.ks_alpha
            if psi_val >= 0.25:
                report["has_significant_drift"] = True

            report["features"][col] = {
                "psi": psi_val,
                "ks_statistic": ks_stat,
                "p_value": p_val,
                "is_drift": is_drift,
                "severity": "CRITICAL" if psi_val >= 0.25 else ("WARNING" if psi_val >= 0.1 else "OK"),
            }

        return report
