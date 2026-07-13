import unittest

import numpy as np
import pandas as pd

from src.context_features import CONTEXT_FEATURE_COLS, add_context_features
from src.modeling import (
    build_group_fold_ids,
    cross_fitted_ensemble_evaluation,
    cross_fitted_threshold_evaluation,
    predictions_from_cross_fitted_selection,
    select_cross_fitted_candidate,
)


class ContextFeatureTests(unittest.TestCase):
    def test_relative_features_are_computed_per_complete_term(self):
        frame = pd.DataFrame(
            {
                "term_id": ["t1", "t1", "t1", "t2", "t2"],
                "query_title_overlap": [0.9, 0.5, 0.1, 0.2, 0.8],
                "query_title_coverage": [1.0, 0.5, 0.0, 0.2, 1.0],
                "query_category_overlap": [0.4, 0.2, 0.0, 0.1, 0.9],
                "tfidf_cosine": [0.8, 0.4, 0.1, 0.2, 0.7],
            }
        )
        result = add_context_features(frame)
        self.assertEqual(result[CONTEXT_FEATURE_COLS].columns.tolist(), CONTEXT_FEATURE_COLS)
        self.assertEqual(result.loc[0, "query_title_overlap_rank"], 1.0)
        self.assertEqual(result.loc[2, "query_title_overlap_rank"], 0.0)
        self.assertAlmostEqual(result.loc[1, "tfidf_cosine_gap"], 0.4)
        self.assertNotEqual(
            result.loc[0, "candidate_count_log1p"],
            result.loc[3, "candidate_count_log1p"],
        )


class ModelingEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array(([1, 0, 0, 0] * 6), dtype=np.int8)
        self.groups = np.repeat([f"t{i}" for i in range(6)], 4)
        self.first = np.tile([0.9, 0.4, 0.2, 0.1], 6)
        self.second = np.tile([0.8, 0.3, 0.4, 0.05], 6)

    def test_group_fold_ids_keep_terms_intact(self):
        fold_ids = build_group_fold_ids(self.y, self.groups, n_splits=3)
        for term_id in np.unique(self.groups):
            self.assertEqual(len(np.unique(fold_ids[self.groups == term_id])), 1)

    def test_threshold_report_distinguishes_validation_from_selection(self):
        fold_ids = build_group_fold_ids(self.y, self.groups, n_splits=3)
        report = cross_fitted_threshold_evaluation(self.y, self.first, fold_ids)
        self.assertEqual(report["cross_fitted_macro_f1"], 1.0)
        self.assertEqual(len(report["folds"]), 3)
        self.assertIn("all_oof_selection_macro_f1", report)

    def test_ensemble_tuning_is_cross_fitted(self):
        fold_ids = build_group_fold_ids(self.y, self.groups, n_splits=3)
        report = cross_fitted_ensemble_evaluation(
            self.y,
            self.first,
            self.second,
            fold_ids,
            weights=[0.0, 0.5, 1.0],
        )
        self.assertEqual(report["cross_fitted_macro_f1"], 1.0)
        self.assertAlmostEqual(
            report["deploy_first_model_weight"]
            + report["deploy_second_model_weight"],
            1.0,
        )

    def test_shortlist_selection_reuses_fold_specific_decisions(self):
        fold_ids = build_group_fold_ids(self.y, self.groups, n_splits=3)
        selection = select_cross_fitted_candidate(
            self.y,
            self.first,
            self.second,
            fold_ids,
            weights=[0.0, 0.5, 1.0],
        )
        predictions = predictions_from_cross_fitted_selection(
            self.first,
            self.second,
            fold_ids,
            selection,
        )
        self.assertTrue(np.array_equal(predictions, self.y))
        self.assertIn(
            selection["deploy"]["selected_model"],
            {"lightgbm", "xgboost", "weighted_blend"},
        )


if __name__ == "__main__":
    unittest.main()
