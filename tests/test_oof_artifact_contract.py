import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from scripts.training import run_model_shortlist
from src.features import FEATURE_COLS
from src.oof_artifacts import (
    EXPECTED_POSITIVE_ROWS,
    EXPECTED_TEST_ROWS,
    OOF_FILENAMES,
    validate_oof_artifacts,
    write_oof_manifest,
)


class OofArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        for index, filename in enumerate(OOF_FILENAMES):
            with open(os.path.join(self.temp_dir.name, filename), "wb") as artifact_file:
                artifact_file.write(f"artifact-{index}".encode())

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_manifest(
        self, training_mode="sample", test_mode="sample", feature_columns=None
    ):
        is_full = training_mode == "full" and test_mode == "full"
        return write_oof_manifest(
            output_dir=self.temp_dir.name,
            feature_columns=feature_columns or ["feature"],
            training_mode=training_mode,
            test_mode=test_mode,
            training_rows=EXPECTED_POSITIVE_ROWS * 4 if is_full else 10,
            test_rows=EXPECTED_TEST_ROWS if is_full else 20,
            negative_ratio=3,
            positive_rows=5 if training_mode == "sample" else EXPECTED_POSITIVE_ROWS,
            positive_reference_rows=EXPECTED_POSITIVE_ROWS,
        )

    def test_accepts_hash_verified_grouped_oof_artifacts(self):
        self.write_manifest()
        manifest = validate_oof_artifacts(self.temp_dir.name)
        self.assertEqual(manifest["validation"]["group_column"], "term_id")

    def test_production_consumers_reject_sample_oof_artifacts(self):
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "full training"):
            validate_oof_artifacts(self.temp_dir.name, require_full=True)

    def test_production_consumers_reject_incomplete_full_oof_counts(self):
        self.write_manifest(
            training_mode="full",
            test_mode="full",
            feature_columns=FEATURE_COLS + ["tfidf_cosine"],
        )
        manifest_path = os.path.join(self.temp_dir.name, "oof_manifest.json")
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        manifest["test_rows"] -= 1
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file)

        with self.assertRaisesRegex(ValueError, "full training"):
            validate_oof_artifacts(self.temp_dir.name, require_full=True)

    def test_rejects_tampered_oof_artifacts(self):
        self.write_manifest(
            training_mode="full",
            test_mode="full",
            feature_columns=FEATURE_COLS + ["tfidf_cosine"],
        )
        with open(os.path.join(self.temp_dir.name, OOF_FILENAMES[0]), "ab") as file:
            file.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_oof_artifacts(self.temp_dir.name, require_full=True)

    def test_production_consumers_reject_stale_feature_columns(self):
        self.write_manifest(training_mode="full", test_mode="full")
        with self.assertRaisesRegex(ValueError, "feature columns"):
            validate_oof_artifacts(self.temp_dir.name, require_full=True)

    def test_sample_shortlist_cannot_overwrite_production_oof_artifacts(self):
        args = SimpleNamespace(
            sample=10,
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
