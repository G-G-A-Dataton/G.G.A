import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.tfidf_features import (
    build_tfidf_vectorizer,
    compute_tfidf_cosine_batch,
    load_vectorizer,
    save_vectorizer,
)


class TfidfFeatureTests(unittest.TestCase):
    def setUp(self):
        self.terms = pd.DataFrame(
            {"term_id": ["t1", "t2"], "query": ["siyah telefon", "deri çanta"]}
        )
        self.items = pd.DataFrame(
            {
                "item_id": ["i1", "i2"],
                "title": ["Telefon", "Çanta"],
                "category": ["Elektronik/Telefon", "Aksesuar/Çanta"],
                "brand": ["Acme", "Acme"],
                "attributes": ["renk: siyah", "materyal: deri"],
            }
        )

    def test_attributes_are_part_of_vectorizer_corpus(self):
        vectorizer = build_tfidf_vectorizer(
            self.terms, self.items, max_features=100, min_df=1
        )
        self.assertIn("siyah", vectorizer.vocabulary_)
        self.assertIn("materyal", vectorizer.vocabulary_)

    def test_cosine_requires_aligned_pairs(self):
        vectorizer = build_tfidf_vectorizer(
            self.terms, self.items, max_features=100, min_df=1
        )
        with self.assertRaisesRegex(ValueError, "equal length"):
            compute_tfidf_cosine_batch(
                ["telefon"], ["telefon", "çanta"], vectorizer, verbose=False
            )

    def test_vectorizer_save_is_reloadable_without_parent_component(self):
        vectorizer = build_tfidf_vectorizer(
            self.terms, self.items, max_features=100, min_df=1
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            previous = os.getcwd()
            try:
                os.chdir(temp_dir)
                save_vectorizer(vectorizer, "vectorizer.pkl")
                restored = load_vectorizer("vectorizer.pkl")
            finally:
                os.chdir(previous)
        np.testing.assert_array_equal(
            vectorizer.get_feature_names_out(), restored.get_feature_names_out()
        )


if __name__ == "__main__":
    unittest.main()
