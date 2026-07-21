"""
tests/test_experiment_tracker.py
=================================
G.G.A Takımı — Experiment Tracker Unit Tests
"""

import json
import pytest
from src.experiment_tracker import ExperimentTracker, log_run


def test_experiment_tracker_json_fallback(tmp_path):
    fallback_file = tmp_path / "experiment_log.json"
    tracker = ExperimentTracker(
        tracking_uri="invalid_uri_force_fallback",
        experiment_name="test_exp",
        fallback_path=str(fallback_file),
    )

    with tracker.start_run("test_run"):
        tracker.log_params({"lr": 0.05, "epochs": 10})
        tracker.log_metrics({"f1": 0.95})

    assert fallback_file.exists()
    data = json.loads(fallback_file.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["run_name"] == "test_run"
    assert data[0]["params"]["lr"] == 0.05
    assert data[0]["metrics"]["final"]["f1"] == 0.95


def test_log_run_convenience(tmp_path):
    # Verify log_run function doesn't crash
    log_run(
        params={"a": 1},
        metrics={"score": 0.8},
        run_name="quick_log",
        config_path="non_existent_config.yaml",
    )
