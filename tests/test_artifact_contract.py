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
from src.candidate_sampling import CANDIDATE_SAMPLING_SCHEMA_VERSION
from src.context_features import CONTEXT_FEATURE_SCHEMA_VERSION
from src.features import FEATURE_SCHEMA_VERSION
from src.modeling import MODEL_FEATURE_COLS


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = self.temp_dir.name
        self.model_paths = [
            os.path.join(root, f"lgbm_v2_fold_{index}.txt") for index in range(1, 6)
        ]
        self.vectorizer_path = os.path.join(root, "tfidf_vectorizer_v2.pkl")
        self.threshold_path = os.path.join(root, "best_threshold_v2.txt")
        self.oof_path = os.path.join(root, "oof_preds_v2.npy")
        self.threshold_report_path = os.path.join(root, "threshold_report_v2.json")
        self.manifest_path = os.path.join(root, "model_manifest_v2.json")
        for index, path in enumerate(self.model_paths, start=1):
            with open(path, "wb") as artifact_file:
                artifact_file.write(f"model-{index}".encode())
        with open(self.vectorizer_path, "wb") as artifact_file:
            artifact_file.write(b"vectorizer")
        with open(self.threshold_path, "w", encoding="utf-8") as artifact_file:
            artifact_file.write("0.45")
        with open(self.oof_path, "wb") as artifact_file:
            artifact_file.write(b"oof")
        with open(self.threshold_report_path, "wb") as artifact_file:
            artifact_file.write(b"threshold-report")
        self.write_manifest(training_mode="full")
        self.patch = mock.patch.multiple(
            inference,
            MODEL_PATHS=self.model_paths,
            VEC_PATH=self.vectorizer_path,
            THRESH_PATH=self.threshold_path,
            OOF_PATH=self.oof_path,
            THRESHOLD_REPORT_PATH=self.threshold_report_path,
            MANIFEST_PATH=self.manifest_path,
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp_dir.cleanup()

    def write_manifest(self, training_mode):
        paths = self.model_paths + [
            self.vectorizer_path,
            self.threshold_path,
            self.oof_path,
            self.threshold_report_path,
        ]
        manifest = {
            "artifact_schema_version": 2,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "context_feature_schema_version": CONTEXT_FEATURE_SCHEMA_VERSION,
            "candidate_sampling_schema_version": CANDIDATE_SAMPLING_SCHEMA_VERSION,
            "training_mode": training_mode,
            "code_revision": "a" * 40,
            "feature_columns": MODEL_FEATURE_COLS,
            "threshold": 0.45,
            "validation": {
                "splitter": "StratifiedGroupKFold",
                "group_column": "term_id",
                "n_splits": 5,
                "random_state": 42,
                "threshold_selection": "cross_fitted",
            },
            "candidate_sampling": {
                "strategy": "test_shaped_category_random",
                "min_candidates": 100,
                "dense_multiplier": 2.0,
                "category_hard_fraction": 0.5,
                "random_state": 42,
                "positive_reference_rows": inference.EXPECTED_POSITIVE_ROWS,
            },
            "training": {
                "terms": inference.EXPECTED_TRAINING_TERMS,
                "rows": inference.EXPECTED_TRAINING_ROWS,
                "positive_rows": inference.EXPECTED_POSITIVE_ROWS,
                "negative_rows": (
                    inference.EXPECTED_TRAINING_ROWS
                    - inference.EXPECTED_POSITIVE_ROWS
                ),
            },
            "metrics": {"cross_fitted_macro_f1": 0.7},
            "source_data_sha256": {
                "terms.csv": "a" * 64,
                "items.csv": "b" * 64,
                "training_pairs.csv": "c" * 64,
            },
            "artifacts": {
                "models": [os.path.basename(path) for path in self.model_paths],
                "vectorizer": os.path.basename(self.vectorizer_path),
                "threshold": os.path.basename(self.threshold_path),
                "oof_predictions": os.path.basename(self.oof_path),
                "threshold_report": os.path.basename(self.threshold_report_path),
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
            feature_cols=MODEL_FEATURE_COLS,
            model_paths=self.model_paths,
            vectorizer_path=self.vectorizer_path,
            threshold_path=self.threshold_path,
            oof_path=self.oof_path,
            threshold_report_path=self.threshold_report_path,
            threshold=0.45,
            training_mode="full",
            candidate_config={
                "min_candidates": 100,
                "dense_multiplier": 2.0,
                "category_hard_fraction": 0.5,
                "random_state": 42,
            },
            candidate_stats={
                "terms": inference.EXPECTED_TRAINING_TERMS,
                "rows": inference.EXPECTED_TRAINING_ROWS,
                "positive_rows": inference.EXPECTED_POSITIVE_ROWS,
                "negative_rows": inference.EXPECTED_TRAINING_ROWS
                - inference.EXPECTED_POSITIVE_ROWS,
            },
            positive_reference_rows=inference.EXPECTED_POSITIVE_ROWS,
            source_data_sha256={
                "terms.csv": "a" * 64,
                "items.csv": "b" * 64,
                "training_pairs.csv": "c" * 64,
            },
            validation_report={"cross_fitted_macro_f1": 0.7},
            revision="a" * 40,
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
                return MODEL_FEATURE_COLS

            def predict(self, features):
                return [0.0] * len(features)

        def fake_build_features(frame, **kwargs):
            for column in MODEL_FEATURE_COLS:
                frame[column] = 0.0
            return frame

        def fake_add_tfidf(frame, vectorizer, **kwargs):
            frame["tfidf_cosine"] = 0.0
            return frame

        def fake_add_context(frame, **kwargs):
            for column in MODEL_FEATURE_COLS:
                if column not in frame:
                    frame[column] = 0.0
            return frame

        def fake_load_feature_batch(base, context, start, end):
            return pd.DataFrame(
                0.0, index=range(end - start), columns=MODEL_FEATURE_COLS
            )

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
            verify_source_data_hashes=mock.Mock(),
            build_features=mock.Mock(side_effect=fake_build_features),
            add_tfidf_features=mock.Mock(side_effect=fake_add_tfidf),
            add_context_features=mock.Mock(side_effect=fake_add_context),
            build_base_feature_store=mock.Mock(
                return_value=("base_features.npy", "term_codes.npy")
            ),
            build_context_feature_store=mock.Mock(
                return_value="context_features.npy"
            ),
            load_feature_batch=mock.Mock(side_effect=fake_load_feature_batch),
            remove_feature_stores=mock.Mock(),
        ), mock.patch.object(
            inference.np, "load", side_effect=[object(), object()]
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

    def test_complete_term_iterator_rejects_reappearing_groups(self):
        chunks = [
            pd.DataFrame({"term_id": ["t1", "t1", "t2"]}),
            pd.DataFrame({"term_id": ["t2", "t1"]}),
        ]
        with self.assertRaisesRegex(ValueError, "reappear|not contiguous"):
            list(inference.iter_complete_term_batches(chunks))

    def test_complete_term_iterator_keeps_boundary_group_together(self):
        chunks = [
            pd.DataFrame({"term_id": ["t1", "t1", "t2"], "value": [1, 2, 3]}),
            pd.DataFrame({"term_id": ["t2", "t3"], "value": [4, 5]}),
        ]
        batches = list(inference.iter_complete_term_batches(chunks))
        combined = pd.concat(batches, ignore_index=True)
        self.assertEqual(combined["value"].tolist(), [1, 2, 3, 4, 5])
        t2_batches = [index for index, batch in enumerate(batches) if "t2" in set(batch.term_id)]
        self.assertEqual(len(t2_batches), 1)


if __name__ == "__main__":
    unittest.main()
