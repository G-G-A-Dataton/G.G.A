import unittest

import numpy as np

from src.rescue import (
    band_override_predictions,
    prediction_summary,
    threshold_predictions,
)


class RescuePredictionTests(unittest.TestCase):
    def test_threshold_predictions_uses_inclusive_boundary(self):
        result = threshold_predictions(np.array([0.1, 0.5, 0.9]), 0.5)
        np.testing.assert_array_equal(result, [0, 1, 1])
        self.assertEqual(result.dtype, np.int8)

    def test_threshold_predictions_rejects_invalid_probabilities(self):
        for values in ([0.1, np.nan], [-0.1, 0.2], [0.2, 1.1]):
            with self.assertRaises(ValueError):
                threshold_predictions(values, 0.5)

    def test_band_override_changes_selected_rows_only(self):
        result = band_override_predictions(
            np.array([0.1, 0.6, 0.7, 0.2]),
            0.5,
            np.array([0.8, 0.1]),
            np.array([False, True, True, False]),
            0.4,
        )
        np.testing.assert_array_equal(result, [0, 1, 0, 0])

    def test_band_override_requires_exact_boolean_coverage(self):
        with self.assertRaises(TypeError):
            band_override_predictions([0.1, 0.2], 0.5, [0.8], [0, 1], 0.5)
        with self.assertRaises(ValueError):
            band_override_predictions(
                [0.1, 0.2], 0.5, [0.8, 0.9], np.array([False, True]), 0.5
            )

    def test_prediction_summary_reports_class_balance(self):
        self.assertEqual(
            prediction_summary(np.array([0, 1, 1], dtype=np.int8)),
            {
                "rows": 3,
                "positive_rows": 2,
                "negative_rows": 1,
                "positive_rate": 2 / 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
