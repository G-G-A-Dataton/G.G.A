import os
import tempfile
import unittest

import pandas as pd
import numpy as np

from src.features import FEATURE_COLS
from src.submission import generate_submission
from src.validate_submission import validate_submission


class SubmissionValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sample_path = os.path.join(self.temp_dir.name, "sample.csv")
        pd.DataFrame({"id": ["a", "b", "c"], "prediction": [0, 0, 0]}).to_csv(
            self.sample_path, index=False
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_submission(self, frame):
        path = os.path.join(self.temp_dir.name, "submission.csv")
        frame.to_csv(path, index=False)
        return path

    def validate(self, frame):
        return validate_submission(
            self.write_submission(frame),
            sample_submission_path=self.sample_path,
            verbose=False,
        )

    def test_accepts_exact_binary_contract(self):
        self.assertTrue(
            self.validate(pd.DataFrame({"id": ["a", "b", "c"], "prediction": [0, 1, 0]}))
        )

    def test_rejects_float_predictions(self):
        self.assertFalse(
            self.validate(
                pd.DataFrame({"id": ["a", "b", "c"], "prediction": [0.0, 1.0, 0.0]})
            )
        )

    def test_rejects_wrong_id_order(self):
        self.assertFalse(
            self.validate(pd.DataFrame({"id": ["b", "a", "c"], "prediction": [0, 1, 0]}))
        )

    def test_rejects_extra_or_reordered_columns(self):
        self.assertFalse(
            self.validate(
                pd.DataFrame(
                    {"prediction": [0, 1, 0], "id": ["a", "b", "c"], "extra": [1, 1, 1]}
                )
            )
        )

    def test_rejects_duplicate_ids(self):
        self.assertFalse(
            self.validate(pd.DataFrame({"id": ["a", "a", "c"], "prediction": [0, 1, 0]}))
        )

    def test_chunked_validation_checks_cross_chunk_duplicates(self):
        submission_path = self.write_submission(
            pd.DataFrame({"id": ["a", "b", "a"], "prediction": [0, 1, 0]})
        )
        self.assertFalse(
            validate_submission(
                submission_path,
                expected_rows=3,
                verbose=False,
                chunk_size=1,
            )
        )

    def test_sample_prefix_validation_supports_inference_smoke_runs(self):
        submission_path = self.write_submission(
            pd.DataFrame({"id": ["a", "b"], "prediction": [0, 1]})
        )
        self.assertTrue(
            validate_submission(
                submission_path,
                sample_submission_path=self.sample_path,
                expected_rows=2,
                verbose=False,
                chunk_size=1,
            )
        )

    def test_legacy_generation_api_streams_and_publishes_valid_output(self):
        class ZeroModel:
            def predict(self, features):
                return np.zeros(len(features))

        pairs_path = os.path.join(self.temp_dir.name, "pairs.csv")
        output_path = os.path.join(self.temp_dir.name, "generated.csv")
        pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "term_id": ["t1", "t1", "t1"],
                "item_id": ["i1", "i1", "i1"],
            }
        ).to_csv(pairs_path, index=False)
        terms = pd.DataFrame({"term_id": ["t1"], "query": ["gri çanta"]})
        items = pd.DataFrame(
            {
                "item_id": ["i1"],
                "title": ["Gri Çanta"],
                "category": ["aksesuar/çanta"],
                "brand": ["Acme"],
                "gender": ["unisex"],
                "age_group": ["yetişkin"],
                "attributes": ["renk: gri"],
            }
        )
        result = generate_submission(
            models=[ZeroModel()],
            terms_df=terms,
            items_df=items,
            feature_cols=FEATURE_COLS,
            submission_pairs_path=pairs_path,
            output_path=output_path,
            threshold=0.5,
            batch_size=2,
            verbose=False,
            sample_submission_path=self.sample_path,
        )
        self.assertEqual(result["prediction"].tolist(), [0, 0, 0])
        self.assertTrue(os.path.exists(output_path))
        self.assertFalse(os.path.exists(output_path + ".tmp"))


if __name__ == "__main__":
    unittest.main()
