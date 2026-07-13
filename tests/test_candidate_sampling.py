import unittest

import pandas as pd

from src.candidate_sampling import (
    build_test_shaped_training_set,
    candidate_distribution,
    candidate_targets,
    sample_complete_terms,
)


def make_catalog(count=240):
    return pd.DataFrame(
        {
            "item_id": [f"i{index}" for index in range(count)],
            "category": [
                "electronics/phone" if index < count // 2 else "home/kitchen"
                for index in range(count)
            ],
        }
    )


class CandidateSamplingTests(unittest.TestCase):
    def setUp(self):
        self.items = make_catalog()
        self.positives = pd.DataFrame(
            {
                "term_id": ["t1"] * 4 + ["t2"] * 60,
                "item_id": [f"i{index}" for index in range(64)],
                "label": [1] * 64,
            }
        )

    def test_targets_match_submission_shape_rule(self):
        targets = candidate_targets(self.positives)
        self.assertEqual(targets.loc["t1"].to_dict(), {
            "positive_count": 4,
            "candidate_count": 100,
            "negative_count": 96,
        })
        self.assertEqual(targets.loc["t2", "candidate_count"], 120)

    def test_term_sampling_never_keeps_partial_positive_groups(self):
        sampled = sample_complete_terms(self.positives, n_terms=1, random_state=3)
        selected_term = sampled["term_id"].iloc[0]
        expected = self.positives[self.positives["term_id"] == selected_term]
        self.assertEqual(len(sampled), len(expected))
        self.assertEqual(set(sampled["item_id"]), set(expected["item_id"]))

    def test_candidate_set_is_exact_reproducible_and_leakage_free(self):
        first = build_test_shaped_training_set(
            self.positives,
            self.items,
            category_hard_fraction=0.5,
            random_state=9,
            verbose=False,
        )
        second = build_test_shaped_training_set(
            self.positives,
            self.items,
            category_hard_fraction=0.5,
            random_state=9,
            verbose=False,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first.groupby("term_id").size().to_dict(), {"t1": 100, "t2": 120})
        self.assertFalse(first.duplicated(["term_id", "item_id"]).any())
        known = set(zip(self.positives.term_id, self.positives.item_id))
        negatives = first[first.label == 0]
        self.assertFalse(known & set(zip(negatives.term_id, negatives.item_id)))
        self.assertIn("category", set(negatives.neg_source))
        self.assertEqual(candidate_distribution(first)["rows"], 220)

    def test_reference_must_cover_selected_positives(self):
        with self.assertRaisesRegex(ValueError, "contain every selected"):
            build_test_shaped_training_set(
                self.positives,
                self.items,
                positive_reference_df=self.positives.iloc[:-1],
                verbose=False,
            )

    def test_selected_terms_must_include_all_known_positives(self):
        partial = self.positives[self.positives["term_id"] == "t1"].iloc[:2]
        with self.assertRaisesRegex(ValueError, "complete positive groups"):
            build_test_shaped_training_set(
                partial,
                self.items,
                positive_reference_df=self.positives,
                verbose=False,
            )


if __name__ == "__main__":
    unittest.main()
