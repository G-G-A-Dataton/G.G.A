import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from scripts.training import run_model_shortlist
from src.modeling import MODEL_FEATURE_COLS
from src.oof_artifacts import (
    EXPECTED_POSITIVE_ROWS,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAINING_ROWS,
    EXPECTED_TRAINING_TERMS,
    OOF_FILENAMES,
    load_oof_artifacts,
    validate_oof_artifacts,
    write_oof_manifest,
)


class OofArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_files = [
            *[f"lgbm_fold_{index}.txt" for index in range(1, 6)],
            *[f"xgb_fold_{index}.json" for index in range(1, 6)],
        ]
        self.support_files = ["tfidf_vectorizer.pkl"]
        for index, filename in enumerate(
            [*OOF_FILENAMES, *self.model_files, *self.support_files]
        ):
            with open(os.path.join(self.temp_dir.name, filename), "wb") as artifact_file:
                artifact_file.write(f"artifact-{index}".encode())

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_manifest(self, training_mode="sample", test_mode="sample"):
        is_full = training_mode == "full" and test_mode == "full"
        return write_oof_manifest(
            output_dir=self.temp_dir.name,
            training_mode=training_mode,
            test_mode=test_mode,
            training_stats={
                "terms": EXPECTED_TRAINING_TERMS if is_full else 5,
                "rows": EXPECTED_TRAINING_ROWS if is_full else 500,
                "positive_rows": EXPECTED_POSITIVE_ROWS if is_full else 25,
                "negative_rows": (
                    EXPECTED_TRAINING_ROWS - EXPECTED_POSITIVE_ROWS
                    if is_full
                    else 475
                ),
            },
            test_rows=EXPECTED_TEST_ROWS if is_full else 20,
            candidate_config={
                "min_candidates": 100,
                "dense_multiplier": 2.0,
                "bm25_hard_fraction": 0.25,
                "category_hard_fraction": 0.5,
                "bm25_top_n": 200,
                "bm25_max_df_ratio": 0.15,
                "random_state": 42,
            },
            positive_reference_rows=EXPECTED_POSITIVE_ROWS,
            source_data_sha256={
                "terms.csv": "a" * 64,
                "items.csv": "b" * 64,
                "training_pairs.csv": "c" * 64,
                "submission_pairs.csv": "d" * 64,
            },
            model_files=self.model_files,
            support_files=self.support_files,
            code_revision="a" * 40,
            feature_columns=MODEL_FEATURE_COLS,
        )

    def test_accepts_hash_verified_grouped_oof_artifacts(self):
        self.write_manifest()
        manifest = validate_oof_artifacts(self.temp_dir.name)
        self.assertEqual(manifest["validation"]["group_column"], "term_id")

    def test_loader_checks_array_shapes_values_and_folds(self):
        arrays = {
            "oof_lgbm.npy": np.linspace(0, 1, 500, dtype=np.float32),
            "test_lgbm.npy": np.linspace(0, 1, 20, dtype=np.float32),
            "oof_xgb.npy": np.linspace(1, 0, 500, dtype=np.float32),
            "test_xgb.npy": np.linspace(1, 0, 20, dtype=np.float32),
            "y_true.npy": np.tile([0, 1], 250).astype(np.int8),
            "fold_ids.npy": np.tile(np.arange(5), 100).astype(np.int8),
        }
        for filename, values in arrays.items():
            with open(os.path.join(self.temp_dir.name, filename), "wb") as output_file:
                np.save(output_file, values, allow_pickle=False)
        self.write_manifest()
        manifest, loaded = load_oof_artifacts(self.temp_dir.name)
        self.assertEqual(manifest["training"]["rows"], 500)
        self.assertEqual(len(loaded["test_xgb.npy"]), 20)

    def test_consumer_rejects_manifest_for_different_source_data(self):
        self.write_manifest()
        source_dir = os.path.join(self.temp_dir.name, "source")
        os.makedirs(source_dir)
        for filename in (
            "terms.csv",
            "items.csv",
            "training_pairs.csv",
            "submission_pairs.csv",
        ):
            with open(os.path.join(source_dir, filename), "w", encoding="utf-8") as source:
                source.write("different data")
        with self.assertRaisesRegex(ValueError, "source data SHA-256 mismatch"):
            validate_oof_artifacts(
                self.temp_dir.name, source_data_dir=source_dir
            )

    def test_production_consumers_reject_sample_oof_artifacts(self):
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "full training"):
            validate_oof_artifacts(self.temp_dir.name, require_full=True)

    def test_production_consumers_reject_incomplete_full_oof_counts(self):
        self.write_manifest(training_mode="full", test_mode="full")
        manifest_path = os.path.join(self.temp_dir.name, "oof_manifest.json")
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        manifest["test_rows"] -= 1
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file)
        with self.assertRaisesRegex(ValueError, "full training"):
            validate_oof_artifacts(self.temp_dir.name, require_full=True)

    def test_rejects_tampered_oof_artifacts(self):
        self.write_manifest(training_mode="full", test_mode="full")
        with open(os.path.join(self.temp_dir.name, OOF_FILENAMES[0]), "ab") as file:
            file.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_oof_artifacts(self.temp_dir.name, require_full=True)

    def test_production_consumers_reject_stale_feature_columns(self):
        self.write_manifest()
        manifest_path = os.path.join(self.temp_dir.name, "oof_manifest.json")
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        manifest["feature_columns"] = ["stale"]
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file)
        with self.assertRaisesRegex(ValueError, "feature columns"):
            validate_oof_artifacts(self.temp_dir.name)

    def test_sample_shortlist_cannot_overwrite_production_oof_artifacts(self):
        args = SimpleNamespace(
            sample=10,
            sample_terms=None,
            test_sample=None,
            artifact_dir=os.path.join(
                run_model_shortlist.OUTPUT_DIR, "ensemble_artifacts"
            ),
        )
        with mock.patch.object(run_model_shortlist, "parse_args", return_value=args):
            with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                run_model_shortlist.main()


if __name__ == "__main__":
    unittest.main()
