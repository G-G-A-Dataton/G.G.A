import unittest

import numpy as np

from scripts.submission.run_final_candidate_set import (
    candidate_predictions,
    rank_candidates,
)


class FinalCandidateSetTests(unittest.TestCase):
    def setUp(self):
        self.selection = {
            "candidates": {
                "lightgbm": {
                    "cross_fitted_macro_f1": 0.82,
                    "lightgbm_weight": 1.0,
                    "xgboost_weight": 0.0,
                    "threshold": 0.4,
                },
                "xgboost": {
                    "cross_fitted_macro_f1": 0.81,
                    "lightgbm_weight": 0.0,
                    "xgboost_weight": 1.0,
                    "threshold": 0.5,
                },
                "weighted_blend": {
                    "cross_fitted_macro_f1": 0.83,
                    "lightgbm_weight": 0.6,
                    "xgboost_weight": 0.4,
                    "threshold": 0.45,
                },
            },
            "deploy": {"selected_model": "weighted_blend"},
        }

    def test_rank_candidates_returns_two_best_cross_fitted_models(self):
        ranked = rank_candidates(self.selection)
        self.assertEqual(
            [candidate["candidate"] for candidate in ranked],
            ["weighted_blend", "lightgbm"],
        )

    def test_rank_candidates_prefers_selected_model_on_tie(self):
        self.selection["candidates"]["lightgbm"]["cross_fitted_macro_f1"] = 0.83
        ranked = rank_candidates(self.selection)
        self.assertEqual(ranked[0]["candidate"], "weighted_blend")

    def test_candidate_predictions_applies_weights_and_threshold(self):
        prediction = candidate_predictions(
            np.array([0.2, 0.8]),
            np.array([0.6, 0.2]),
            self.selection["candidates"]["weighted_blend"],
        )
        np.testing.assert_array_equal(prediction, np.array([0, 1], dtype=np.int8))

    def test_candidate_predictions_rejects_invalid_weights(self):
        candidate = dict(self.selection["candidates"]["weighted_blend"])
        candidate["xgboost_weight"] = 0.5
        with self.assertRaisesRegex(ValueError, "deploy parameters"):
            candidate_predictions(np.array([0.5]), np.array([0.5]), candidate)

    def test_rank_candidates_requires_complete_shortlist(self):
        del self.selection["candidates"]["xgboost"]
        with self.assertRaisesRegex(ValueError, "complete model shortlist"):
            rank_candidates(self.selection)


if __name__ == "__main__":
    unittest.main()
