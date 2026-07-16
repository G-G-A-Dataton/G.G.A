import json
import os
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from scripts.delivery.run_offline_inference import _predict_batch
from scripts.delivery.write_extra_data_manifest import build_manifest
from scripts.training.run_model_shortlist import (
    _export_candidates,
    _resolve_code_revision,
)
from src.solution_contract import validate_deploy_decision


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeLightGBM:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def predict(self, features):
        return self.values[: len(features)]


class _FakeXGBoost:
    best_iteration = None

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def predict(self, matrix, iteration_range):
        return self.values[: matrix.num_row()]


class SolutionDeliveryTests(unittest.TestCase):
    def test_predict_batch_applies_all_fold_models_and_deploy_threshold(self):
        features = pd.DataFrame({"feature": [0.0, 1.0]})
        predictions = _predict_batch(
            features,
            [_FakeLightGBM([0.2, 0.8]), _FakeLightGBM([0.4, 0.6])],
            [_FakeXGBoost([0.6, 0.2]), _FakeXGBoost([0.4, 0.4])],
            {
                "lightgbm_weight": 0.5,
                "xgboost_weight": 0.5,
                "threshold": 0.5,
            },
        )
        np.testing.assert_array_equal(predictions, np.array([0, 1], dtype=np.int8))

    def test_explicit_source_revision_must_be_canonical(self):
        revision = "a" * 40
        self.assertEqual(_resolve_code_revision(revision), revision)
        with self.assertRaisesRegex(ValueError, "40-character"):
            _resolve_code_revision("ABC")

    def test_candidate_export_is_atomic_and_preserves_contract(self):
        candidates = pd.DataFrame(
            {
                "term_id": ["t1", "t1"],
                "item_id": ["i1", "i2"],
                "label": [1, 0],
                "neg_source": ["positive", "random"],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "generated.csv")
            self.assertEqual(_export_candidates(candidates, output), output)
            actual = pd.read_csv(output)
            self.assertEqual(
                actual.columns.tolist(),
                ["term_id", "item_id", "label", "neg_source"],
            )
            self.assertFalse(os.path.exists(output + ".tmp"))

    def test_extra_data_manifest_matches_training_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = os.path.join(directory, "generated.csv")
            model_path = os.path.join(directory, "models")
            os.makedirs(model_path)
            pd.DataFrame(
                {
                    "term_id": ["t1", "t1", "t2"],
                    "item_id": ["i1", "i2", "i3"],
                    "label": [1, 0, 0],
                    "neg_source": ["positive", "bm25", "random"],
                }
            ).to_csv(candidate_path, index=False)
            with open(
                os.path.join(model_path, "oof_manifest.json"),
                "w",
                encoding="utf-8",
            ) as manifest_file:
                json.dump(
                    {
                        "training": {
                            "rows": 3,
                            "positive_rows": 1,
                            "negative_rows": 2,
                        },
                        "candidate_sampling": {"random_state": 42},
                    },
                    manifest_file,
                )
            payload = build_manifest(candidate_path, model_path)
            self.assertEqual(payload["rows"], 3)
            self.assertEqual(payload["label_rows"], {"0": 2, "1": 1})
            self.assertEqual(payload["source_rows"]["bm25"], 1)
            self.assertFalse(payload["external_data_used"])

    def test_deploy_decision_is_bound_to_recomputed_selection(self):
        expected = {
            "validation": {},
            "deploy": {
                "selected_model": "weighted_blend",
                "lightgbm_weight": 0.65,
                "xgboost_weight": 0.35,
                "threshold": 0.37,
                "positive_rate": 0.19,
                "rows": 10,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            with open(
                os.path.join(directory, "ensemble_decision.json"),
                "w",
                encoding="utf-8",
            ) as decision_file:
                json.dump(expected, decision_file)
            with mock.patch(
                "src.solution_contract.build_deploy_decision",
                return_value=expected,
            ):
                actual = validate_deploy_decision(directory, directory)
            self.assertEqual(actual, expected)

    def test_all_official_entrypoints_expose_required_arguments(self):
        expected = {
            "step1.sh": "--env-path",
            "step2.sh": "--competition_data_path",
            "step3.sh": "--model_dump_path",
        }
        for script, required_argument in expected.items():
            result = subprocess.run(
                ["bash", os.path.join(PROJECT_ROOT, script), "--help"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn(required_argument, result.stdout)


if __name__ == "__main__":
    unittest.main()
