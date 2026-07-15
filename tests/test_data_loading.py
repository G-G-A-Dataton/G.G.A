import os
import tempfile
import unittest

import pandas as pd

from src.data import load_terms, merge_pairs


class DataLoadingContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_csv(self, name, frame):
        path = os.path.join(self.temp_dir.name, name)
        frame.to_csv(path, index=False)
        return path

    def test_terms_require_exact_schema_and_unique_ids(self):
        wrong_schema = self.write_csv(
            "wrong_terms.csv",
            pd.DataFrame({"query": ["telefon"], "term_id": ["t1"]}),
        )
        with self.assertRaisesRegex(ValueError, "columns must be"):
            load_terms(wrong_schema)

        duplicates = self.write_csv(
            "duplicate_terms.csv",
            pd.DataFrame(
                {"term_id": ["t1", "t1"], "query": ["telefon", "telefon"]}
            ),
        )
        with self.assertRaisesRegex(ValueError, "duplicate term_id"):
            load_terms(duplicates)

    def test_merge_rejects_unresolved_references(self):
        pairs_path = self.write_csv(
            "training_pairs.csv",
            pd.DataFrame(
                {
                    "id": ["p1"],
                    "term_id": ["missing"],
                    "item_id": ["i1"],
                    "label": [1],
                }
            ),
        )
        terms = pd.DataFrame({"term_id": ["t1"], "query": ["telefon"]})
        items = pd.DataFrame(
            {
                "item_id": ["i1"],
                "title": ["Telefon"],
                "category": ["Elektronik"],
                "brand": [""],
                "gender": ["unknown"],
                "age_group": ["unknown"],
                "attributes": [""],
            }
        )
        with self.assertRaisesRegex(ValueError, "unresolved"):
            merge_pairs(pairs_path, terms, items)

    def test_loader_rejects_git_lfs_pointer_instead_of_parsing_it(self):
        path = os.path.join(self.temp_dir.name, "terms.csv")
        with open(path, "w", encoding="utf-8") as pointer_file:
            pointer_file.write(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:deadbeef\nsize 123\n"
            )
        with self.assertRaisesRegex(RuntimeError, "Git LFS pointer"):
            load_terms(path)


if __name__ == "__main__":
    unittest.main()
