import json
import os
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.delivery_artifacts import (
    validate_delivery_manifest,
    write_delivery_manifest,
)


class DeliveryArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = self.temp_dir.name
        self.submission = os.path.join(self.root, "submission.csv")
        self.decision = os.path.join(self.root, "decision.json")
        self.oof_manifest = os.path.join(self.root, "oof_manifest.json")
        self.delivery = os.path.join(self.root, "delivery.json")
        pd.DataFrame(
            {"id": ["a", "b", "c"], "prediction": [1, 0, 1]}
        ).to_csv(self.submission, index=False)
        with open(self.decision, "w", encoding="utf-8") as decision_file:
            json.dump(
                {
                    "deploy": {
                        "selected_model": "lightgbm",
                        "rows": 3,
                        "positive_rate": 2 / 3,
                    }
                },
                decision_file,
            )
        with open(self.oof_manifest, "w", encoding="utf-8") as oof_file:
            json.dump(
                {
                    "code_revision": "a" * 40,
                    "source_data_sha256": {"terms.csv": "b" * 64},
                },
                oof_file,
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write(self):
        return write_delivery_manifest(
            self.delivery,
            project_root=self.root,
            submission_path=self.submission,
            decision_path=self.decision,
            oof_manifest_path=self.oof_manifest,
            code_revision="c" * 40,
            submission_rows=3,
            positive_rows=2,
        )

    def test_delivery_manifest_binds_candidate_and_decision(self):
        self._write()
        with mock.patch(
            "src.delivery_artifacts.validate_oof_artifacts",
            return_value={
                "code_revision": "a" * 40,
                "source_data_sha256": {"terms.csv": "b" * 64},
            },
        ):
            manifest = validate_delivery_manifest(
                self.delivery, project_root=self.root
            )
        self.assertEqual(manifest["submission"]["positive_rows"], 2)

    def test_delivery_manifest_rejects_tampered_submission(self):
        self._write()
        with open(self.submission, "a", encoding="utf-8") as submission_file:
            submission_file.write("d,1\n")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_delivery_manifest(self.delivery, project_root=self.root)

    def test_delivery_manifest_rejects_paths_outside_project(self):
        outside = os.path.join(os.path.dirname(self.root), "outside.csv")
        with open(outside, "w", encoding="utf-8") as outside_file:
            outside_file.write("id,prediction\na,1\n")
        try:
            with self.assertRaisesRegex(ValueError, "inside project_root"):
                write_delivery_manifest(
                    self.delivery,
                    project_root=self.root,
                    submission_path=outside,
                    decision_path=self.decision,
                    oof_manifest_path=self.oof_manifest,
                    code_revision="c" * 40,
                    submission_rows=1,
                    positive_rows=1,
                )
        finally:
            os.remove(outside)


if __name__ == "__main__":
    unittest.main()
