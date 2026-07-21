"""
tests/test_drift_detector.py
=============================
G.G.A Takımı — Drift Detector Unit Tests
"""

import numpy as np
import pandas as pd
import pytest
from src.monitoring.drift_detector import DriftDetector, compute_ks_test, compute_psi


def test_compute_psi_no_drift():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    curr = np.random.normal(0, 1, 1000)

    psi = compute_psi(ref, curr)
    assert psi < 0.1  # No drift


def test_compute_psi_significant_drift():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    curr = np.random.normal(5, 1, 1000)  # Large mean shift

    psi = compute_psi(ref, curr)
    assert psi >= 0.25  # Significant drift


def test_drift_detector_report():
    np.random.seed(42)
    ref_df = pd.DataFrame({"feat1": np.random.normal(0, 1, 500)})
    curr_df = pd.DataFrame({"feat1": np.random.normal(3, 1, 500)})

    detector = DriftDetector(ref_df, psi_threshold=0.1)
    report = detector.detect_drift(curr_df)

    assert report["has_significant_drift"] is True
    assert "feat1" in report["features"]
    assert report["features"]["feat1"]["is_drift"] is True
