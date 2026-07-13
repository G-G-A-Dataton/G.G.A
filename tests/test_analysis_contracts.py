import unittest

import numpy as np
import pandas as pd

from scripts.analysis.run_ensemble_comparison import build_comparison
from scripts.analysis.run_feature_importance import aggregate_importance
from scripts.analysis.run_hata_taksonomisi import classify_error_signals
from scripts.analysis.run_threshold_analysis import analyze_thresholds
from src.error_analysis import split_errors
from src.modeling import select_cross_fitted_candidate


class AnalysisContractTests(unittest.TestCase):
    def test_feature_importance_aggregates_verified_fold_shapes(self):
        class FakeModel:
            def __init__(self, gain, split):
                self.gain = np.asarray(gain)
                self.split = np.asarray(split)

            def feature_name(self):
                return ["a", "b"]

            def feature_importance(self, importance_type):
                return self.gain if importance_type == "gain" else self.split

        result = aggregate_importance(
            [FakeModel([3, 1], [2, 1]), FakeModel([1, 1], [1, 1])],
            ["a", "b"],
        )
        self.assertEqual(result["feature"].tolist(), ["a", "b"])
        self.assertAlmostEqual(result["gain_ratio"].sum(), 1.0)

    def test_threshold_diagnostics_include_both_classes(self):
        result = analyze_thresholds(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.4, 0.6, 0.9]),
            [0.5],
        )
        self.assertEqual(result.loc[0, "macro_f1"], 1.0)
        self.assertEqual(result.loc[0, "tn"], 2)
        self.assertEqual(result.loc[0, "tp"], 2)

    def test_error_split_rejects_misaligned_probabilities(self):
        with self.assertRaisesRegex(ValueError, "aligned finite probabilities"):
            split_errors(pd.DataFrame({"label": [0, 1]}), [0.1])

    def test_error_taxonomy_uses_observed_conflicts(self):
        frame = pd.DataFrame(
            {
                "label": [0, 1, 0],
                "query_model_token_conflict": [1, 0, 0],
                "demographic_conflict": [0, 0, 0],
                "query_color_match": [0, -1, 0],
                "query_size_match": [0, 0, 0],
                "query_material_match": [0, 0, 0],
                "query_title_overlap": [0.5, 0.2, 0.0],
                "query_category_overlap": [0.0, 0.1, 0.0],
                "query_title_coverage": [0.8, 0.2, 0.0],
            }
        )
        self.assertEqual(
            classify_error_signals(frame).tolist(),
            ["MODEL_CODE_CONFLICT", "COLOR_CONFLICT", "NO_LEXICAL_EVIDENCE"],
        )

    def test_comparison_uses_common_selection_contract(self):
        y_true = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
        fold_ids = np.array([0, 0, 1, 1, 2, 2], dtype=np.int8)
        lightgbm = np.array([0.9, 0.1, 0.8, 0.2, 0.95, 0.05])
        xgboost = np.array([0.8, 0.2, 0.85, 0.15, 0.9, 0.1])
        selection = select_cross_fitted_candidate(
            y_true, lightgbm, xgboost, fold_ids, weights=[0.0, 0.5, 1.0]
        )
        comparison = build_comparison(selection)
        self.assertEqual(len(comparison), 3)
        self.assertEqual(int(comparison["selected"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
