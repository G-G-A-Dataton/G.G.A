"""
tests/test_calibration.py
==========================
G.G.A Takımı — Calibration Unit Tests
"""

import numpy as np
import pytest
from src.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    evaluate_calibration,
    expected_calibration_error,
    maximum_calibration_error,
)


def test_ece_and_mce():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.85, 0.15, 0.95, 0.05, 0.75, 0.25])

    ece = expected_calibration_error(y_true, y_prob, n_bins=5)
    mce = maximum_calibration_error(y_true, y_prob, n_bins=5)

    assert 0.0 <= ece <= 1.0
    assert 0.0 <= mce <= 1.0
    assert ece <= mce + 1e-6


def test_platt_calibrator():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=100)
    # Uncalibrated probabilities (overconfident)
    y_prob = np.where(y_true == 1, 0.99, 0.01)

    cal = PlattCalibrator().fit(y_prob, y_true)
    calibrated = cal.predict_proba(y_prob)

    assert calibrated.shape == y_prob.shape
    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()


def test_isotonic_calibrator():
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=100)
    y_prob = np.random.rand(100)

    cal = IsotonicCalibrator().fit(y_prob, y_true)
    calibrated = cal.predict_proba(y_prob)

    assert calibrated.shape == y_prob.shape
    assert ((calibrated >= 0.0) & (calibrated <= 1.0)).all()


def test_evaluate_calibration():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2])

    res = evaluate_calibration(y_true, y_prob)
    assert "ece" in res
    assert "mce" in res
    assert "overconfidence_bias" in res
