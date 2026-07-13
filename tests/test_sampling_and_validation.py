import unittest

import pandas as pd

from src.metrics import get_stratified_group_kfold
from src.negative_sampling import build_training_set, generate_random_negatives
from src.train_mix_v2 import build_mixed_training_set


def make_items(count=20):
    return pd.DataFrame(
        {
            "item_id": [f"i{index}" for index in range(1, count + 1)],
            "title": [f"product {index}" for index in range(1, count + 1)],
            "category": ["category"] * count,
            "brand": ["brand"] * count,
            "attributes": [""] * count,
            "description": [""] * count,
        }
    )


class NegativeSamplingTests(unittest.TestCase):
    def setUp(self):
        self.items = make_items()
        self.known = pd.DataFrame(
            {
                "term_id": ["t1"] * 4 + ["t2"] * 3,
                "item_id": ["i1", "i2", "i3", "i4", "i5", "i6", "i7"],
                "label": [1] * 7,
            }
        )
        self.sample = self.known.iloc[[0, 1, 4]].copy()

    def test_random_sampling_is_exact_unique_and_leakage_free(self):
        full = build_training_set(
            self.sample,
            self.items,
            ratio=3,
            random_state=7,
            verbose=False,
            positive_reference_df=self.known,
        )
        negatives = full[full["label"] == 0]
        self.assertEqual(
            negatives.groupby("term_id").size().to_dict(), {"t1": 6, "t2": 3}
        )
        self.assertFalse(negatives.duplicated(["term_id", "item_id"]).any())
        known_pairs = set(zip(self.known["term_id"], self.known["item_id"]))
        negative_pairs = set(zip(negatives["term_id"], negatives["item_id"]))
        self.assertFalse(known_pairs & negative_pairs)

    def test_random_fallback_honors_external_exclusions(self):
        quota = pd.DataFrame({"term_id": ["t1"] * 4, "item_id": [pd.NA] * 4})
        excluded = pd.DataFrame({"term_id": ["t1"], "item_id": ["i8"]})
        negatives = generate_random_negatives(
            quota,
            self.items,
            positive_reference_df=self.known,
            excluded_pairs_df=excluded,
            verbose=False,
        )
        self.assertNotIn(("t1", "i8"), set(zip(negatives.term_id, negatives.item_id)))

    def test_mixed_sampling_fills_every_per_term_quota(self):
        terms = pd.DataFrame({"term_id": ["t1", "t2"], "query": ["", ""]})
        full = build_mixed_training_set(
            self.sample,
            terms,
            self.items,
            ratio=3,
            bm25_top_n=5,
            bm25_max_df_ratio=1.0,
            random_state=7,
            verbose=False,
            positive_reference_df=self.known,
        )
        negatives = full[full["label"] == 0]
        self.assertEqual(
            negatives.groupby("term_id").size().to_dict(), {"t1": 6, "t2": 3}
        )
        self.assertEqual(set(negatives["neg_source"]), {"random"})

    def test_impossible_unique_quota_fails_fast(self):
        tiny_items = make_items(5)
        with self.assertRaisesRegex(ValueError, "Not enough unique catalog items"):
            generate_random_negatives(
                self.sample[self.sample["term_id"] == "t1"],
                tiny_items,
                ratio=2,
                positive_reference_df=self.known,
                verbose=False,
            )


class GroupValidationTests(unittest.TestCase):
    def test_query_groups_never_cross_fold_boundaries(self):
        frame = pd.DataFrame({"value": range(24)})
        labels = pd.Series(([1] + [0] * 3) * 6)
        groups = pd.Series(sum(([f"t{index}"] * 4 for index in range(6)), []))
        splitter = get_stratified_group_kfold(n_splits=3, random_state=42)
        for train_index, validation_index in splitter.split(
            frame, labels, groups=groups
        ):
            train_groups = set(groups.iloc[train_index])
            validation_groups = set(groups.iloc[validation_index])
            self.assertTrue(train_groups.isdisjoint(validation_groups))


if __name__ == "__main__":
    unittest.main()
