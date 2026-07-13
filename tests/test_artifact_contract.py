import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import pandas as pd

from pipeline import inference
from scripts.training import run_train_full_v2
from scripts.training.run_train_full_v2 import write_artifact_manifest
from src.features import FEATURE_COLS, FEATURE_SCHEMA_VERSION


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = self.temp_dir.name
        self.model_paths = [
            os.path.join(root, f"lgbm_v2_fold_{index}.txt") for index in range(1, 6)
        ]
        self.vectorizer_path = os.path.join(root, "tfidf_vectorizer_v2.pkl")
        self.threshold_path = os.path.join(root, "best_threshold_v2.txt")
        self.manifest_path = os.path.join(root, "model_manifest_v2.json")
        for index, path in enumerate(self.model_paths, start=1):
            with open(path, "wb") as artifact_file:
                artifact_file.write(f"model-{index}".encode())
        with open(self.vectorizer_path, "wb") as artifact_file:
            artifact_file.write(b"vectorizer")
        with open(self.threshold_path, "w", encoding="utf-8") as artifact_file:
            artifact_file.write("0.45")
        self.write_manifest(training_mode="full")
        self.patch = mock.patch.multiple(
            inference,
            MODEL_PATHS=self.model_paths,
            VEC_PATH=self.vectorizer_path,
            THRESH_PATH=self.threshold_path,
            MANIFEST_PATH=self.manifest_path,
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp_dir.cleanup()

    def write_manifest(self, training_mode):
        paths = self.model_paths + [self.vectorizer_path, self.threshold_path]
        manifest = {
            "artifact_schema_version": 1,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "training_mode": training_mode,
            "feature_columns": FEATURE_COLS + ["tfidf_cosine"],
            "threshold": 0.45,
            "validation": {
                "splitter": "StratifiedGroupKFold",
                "group_column": "term_id",
                "n_splits": 5,
                "random_state": 42,
            },
            "negative_sampling": {
                "strategy": "bm25_random_fallback",
                "ratio": 3,
                "positive_reference_rows": inference.EXPECTED_POSITIVE_ROWS,
            },
            "training": {
                "positive_rows": inference.EXPECTED_POSITIVE_ROWS,
                "negative_rows": inference.EXPECTED_POSITIVE_ROWS * 3,
                "total_rows": inference.EXPECTED_POSITIVE_ROWS * 4,
            },
            "artifacts": {
                "models": [os.path.basename(path) for path in self.model_paths],
                "vectorizer": os.path.basename(self.vectorizer_path),
                "threshold": os.path.basename(self.threshold_path),
            },
            "sha256": {
                os.path.basename(path): inference.sha256_file(path) for path in paths
            },
        }
        with open(self.manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file)

    def test_accepts_complete_full_training_artifact_set(self):
        manifest = inference.check_dependencies()
        self.assertEqual(manifest["feature_schema_version"], FEATURE_SCHEMA_VERSION)

    def test_training_writer_emits_verifiable_manifest(self):
        write_artifact_manifest(
            artifact_dir=self.temp_dir.name,
            feature_cols=FEATURE_COLS + ["tfidf_cosine"],
            model_paths=self.model_paths,
            vectorizer_path=self.vectorizer_path,
            threshold_path=self.threshold_path,
            threshold=0.45,
            mean_f1=0.7,
            std_f1=0.01,
            best_f1=0.71,
            training_mode="full",
            positive_rows=inference.EXPECTED_POSITIVE_ROWS,
            negative_rows=inference.EXPECTED_POSITIVE_ROWS * 3,
            total_rows=inference.EXPECTED_POSITIVE_ROWS * 4,
            positive_reference_rows=inference.EXPECTED_POSITIVE_ROWS,
        )
        manifest = inference.check_dependencies()
        self.assertEqual(manifest["validation"]["group_column"], "term_id")

    def test_rejects_sample_training_artifacts(self):
        self.write_manifest(training_mode="sample")
        with self.assertRaisesRegex(ValueError, "training_mode"):
            inference.check_dependencies()

    def test_rejects_incomplete_full_training_counts(self):
        with open(self.manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        manifest["training"]["positive_rows"] -= 1
        with open(self.manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file)
        with self.assertRaisesRegex(ValueError, "row counts"):
            inference.check_dependencies()

    def test_rejects_tampered_artifact(self):
        with open(self.model_paths[0], "ab") as artifact_file:
            artifact_file.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            inference.check_dependencies()

    def test_failed_submission_validation_preserves_existing_output(self):
        output_path = os.path.join(self.temp_dir.name, "submission.csv")
        with open(output_path, "w", encoding="utf-8") as output_file:
            output_file.write("known-good\n")
        submission = pd.DataFrame({"id": ["a"], "prediction": [1]})

        with mock.patch.object(inference, "validate_submission", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "validation failed"):
                inference.publish_submission(submission, output_path)

        with open(output_path, encoding="utf-8") as output_file:
            self.assertEqual(output_file.read(), "known-good\n")
        self.assertFalse(os.path.exists(output_path + ".tmp"))

    def test_inference_streams_batches_to_a_valid_atomic_output(self):
        pairs_path = os.path.join(self.temp_dir.name, "submission_pairs.csv")
        sample_path = os.path.join(self.temp_dir.name, "sample_submission.csv")
        output_path = os.path.join(self.temp_dir.name, "generated.csv")
        pairs = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "term_id": ["t1", "t1", "t1"],
                "item_id": ["i1", "i1", "i1"],
            }
        )
        pairs.to_csv(pairs_path, index=False)
        pd.DataFrame({"id": ["a", "b", "c"], "prediction": [0, 0, 0]}).to_csv(
            sample_path, index=False
        )
        terms = pd.DataFrame({"term_id": ["t1"], "query": ["query"]})
        items = pd.DataFrame({"item_id": ["i1"], "title": ["title"]})

        class FakeModel:
            def feature_name(self):
                return FEATURE_COLS + ["tfidf_cosine"]

            def predict(self, features):
                return [0.0] * len(features)

        def fake_build_features(frame):
            for column in FEATURE_COLS:
                frame[column] = 0.0
            return frame

        def fake_add_tfidf(frame, vectorizer):
            frame["tfidf_cosine"] = 0.0
            return frame

        args = SimpleNamespace(
            batch_size=2,
            sample=3,
            threshold=None,
            output=output_path,
        )
        logger = mock.Mock()
        with mock.patch.multiple(
            inference,
            DATA_DIR=self.temp_dir.name,
            configure_logging=mock.Mock(return_value=logger),
            load_terms=mock.Mock(return_value=terms),
            load_items=mock.Mock(return_value=items),
            load_vectorizer=mock.Mock(return_value=object()),
            build_features=mock.Mock(side_effect=fake_build_features),
            add_tfidf_features=mock.Mock(side_effect=fake_add_tfidf),
        ), mock.patch.object(inference.lgb, "Booster", return_value=FakeModel()):
            result = inference.run_prediction_pipeline(args)

        self.assertEqual(result, output_path)
        self.assertEqual(pd.read_csv(output_path)["prediction"].tolist(), [0, 0, 0])
        self.assertFalse(os.path.exists(output_path + ".tmp"))

    def test_sample_inference_cannot_overwrite_production_output(self):
        args = SimpleNamespace(
            batch_size=2,
            sample=3,
            threshold=None,
            output=inference.DEFAULT_OUTPUT,
        )
        with self.assertRaisesRegex(ValueError, "cannot overwrite"):
            inference.run_prediction_pipeline(args)

    def test_sample_training_cannot_overwrite_production_artifacts(self):
        args = SimpleNamespace(
            sample=10,
            bm25_top_n=5,
            artifact_dir=run_train_full_v2.OUTPUT_DIR,
            no_error_analysis=True,
        )
        with mock.patch.object(run_train_full_v2, "parse_args", return_value=args):
            with self.assertRaisesRegex(ValueError, "cannot write"):
                run_train_full_v2.main()


if __name__ == "__main__":
    unittest.main()
