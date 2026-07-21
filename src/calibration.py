"""
src/calibration.py
==================
G.G.A Takımı — Probability Calibration & Reliability Metrics

Model tahmin olasılıklarının güvenilirliğini (probability calibration)
ölçen ve düzelten modül.

Teknikler:
  - ECE (Expected Calibration Error)
  - MCE (Maximum Calibration Error)
  - Platt Scaling (Logistic Regression on Logits)
  - Isotonic Regression
  - Temperature Scaling

Kullanım:
  >>> from src.calibration import PlattCalibrator, expected_calibration_error
  >>> ece = expected_calibration_error(y_true, y_prob)
  >>> calibrator = PlattCalibrator().fit(y_val_prob, y_val_true)
  >>> calibrated_prob = calibrator.predict_proba(y_test_prob)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# 1. Metrikler: ECE ve MCE
# ---------------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE) hesaplar.

    ECE = sum_{b=1}^B (|bin_b| / N) * |acc(bin_b) - conf(bin_b)|
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bins) - 1

    total_samples = len(y_true)
    ece = 0.0

    for i in range(n_bins):
        mask = bin_assignments == i
        if not np.any(mask):
            continue

        bin_acc = np.mean(y_true[mask])
        bin_conf = np.mean(y_prob[mask])
        weight = np.sum(mask) / total_samples

        ece += weight * np.abs(bin_acc - bin_conf)

    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Maximum Calibration Error (MCE) hesaplar.

    MCE = max_{b=1}^B |acc(bin_b) - conf(bin_b)|
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bins) - 1

    mce = 0.0

    for i in range(n_bins):
        mask = bin_assignments == i
        if not np.any(mask):
            continue

        bin_acc = np.mean(y_true[mask])
        bin_conf = np.mean(y_prob[mask])

        diff = np.abs(bin_acc - bin_conf)
        if diff > mce:
            mce = diff

    return float(mce)


# ---------------------------------------------------------------------------
# 2. Kalibratörler
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """
    Platt Scaling (Logit uzerine Lojistik Regresyon).
    """

    def __init__(self):
        self.model = LogisticRegression(C=1e3, solver="lbfgs")

    def _logit(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        return np.log(p / (1.0 - p)).reshape(-1, 1)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
        logits = self._logit(y_prob)
        self.model.fit(logits, y_true)
        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        logits = self._logit(y_prob)
        return self.model.predict_proba(logits)[:, 1]


class IsotonicCalibrator:
    """
    Isotonic Regression Kalibratoru.
    """

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        self.model.fit(y_prob, y_true)
        return self

    def predict_proba(self, y_prob: np.ndarray) -> np.ndarray:
        return self.model.predict(y_prob)


# ---------------------------------------------------------------------------
# 3. Kalibrasyon Raporu
# ---------------------------------------------------------------------------

def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Kalibrasyon istatistiklerini hesaplar.
    """
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    mce = maximum_calibration_error(y_true, y_prob, n_bins=n_bins)
    avg_conf = float(np.mean(y_prob))
    avg_acc = float(np.mean(y_true))

    return {
        "ece": ece,
        "mce": mce,
        "average_confidence": avg_conf,
        "average_accuracy": avg_acc,
        "overconfidence_bias": avg_conf - avg_acc,
    }
