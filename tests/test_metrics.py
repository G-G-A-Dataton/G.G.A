import unittest

import numpy as np

from src.metrics import find_best_threshold, macro_f1, macro_f1_from_proba


class MetricContractTests(unittest.TestCase):
    def test_macro_f1_always_scores_both_competition_classes(self):
        self.assertEqual(macro_f1([0, 0], [0, 0]), 0.5)

    def test_exact_threshold_search_keeps_unrounded_optimum(self):
        labels = np.array([1, 1, 0, 0], dtype=np.int8)
        probabilities = np.array([0.501, 0.5009, 0.5008, 0.1])
        threshold, score, results = find_best_threshold(labels, probabilities)

        self.assertEqual(threshold, 0.5009)
        self.assertEqual(score, 1.0)
        self.assertIn((0.5009, 1.0), results)

    def test_rejects_invalid_metric_inputs(self):
        with self.assertRaisesRegex(ValueError, "equal length"):
            macro_f1([0, 1], [1])
        with self.assertRaisesRegex(ValueError, "binary"):
            macro_f1([0, 2], [0, 1])
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            macro_f1_from_proba([0, 1], [0.2, 1.2])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            find_best_threshold([0, 1], [0.2, 0.8], thresholds=[])


if __name__ == "__main__":
    unittest.main()
