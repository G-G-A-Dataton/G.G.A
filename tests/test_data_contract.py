import os
import tempfile
import unittest

import pandas as pd

from scripts.data.verify_pipeline import verify_submission_contract


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pairs_path = os.path.join(self.temp_dir.name, "pairs.csv")
        self.sample_path = os.path.join(self.temp_dir.name, "sample.csv")
        pd.DataFrame(
            {
                "id": ["a", "b"],
                "term_id": ["t1", "t2"],
                "item_id": ["i1", "i2"],
            }
        ).to_csv(self.pairs_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_sample(self, ids):
        pd.DataFrame({"id": ids, "prediction": [0] * len(ids)}).to_csv(
            self.sample_path, index=False
        )

    def test_accepts_exact_submission_input_contract(self):
        self.write_sample(["a", "b"])
        self.assertEqual(
            verify_submission_contract(
                self.pairs_path, self.sample_path, expected_rows=2
            ),
            2,
        )

    def test_rejects_mismatched_sample_id_order(self):
        self.write_sample(["b", "a"])
        with self.assertRaisesRegex(ValueError, "sample ID order"):
            verify_submission_contract(
                self.pairs_path, self.sample_path, expected_rows=2
            )


if __name__ == "__main__":
    unittest.main()
