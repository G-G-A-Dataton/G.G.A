import hashlib
import json
import os
import tempfile
import unittest

import pandas as pd

from src.data_freeze import verify_data_freeze


class DataFreezeTests(unittest.TestCase):
    def test_verifies_and_rejects_changed_source_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = os.path.join(temp_dir, "tiny.csv")
            config_path = os.path.join(temp_dir, "config.json")
            pd.DataFrame({"id": ["a", "b"], "value": [1, 2]}).to_csv(
                data_path, index=False
            )
            with open(data_path, "rb") as data_file:
                digest = hashlib.sha256(data_file.read()).hexdigest()
            config = {
                "config_schema_version": 1,
                "data_freeze": {
                    "version": "test-v1",
                    "files": {
                        "tiny.csv": {
                            "bytes": os.path.getsize(data_path),
                            "rows": 2,
                            "columns": ["id", "value"],
                            "sha256": digest,
                        }
                    },
                },
            }
            with open(config_path, "w", encoding="utf-8") as config_file:
                json.dump(config, config_file)
            self.assertEqual(
                verify_data_freeze(config_path, temp_dir)["version"], "test-v1"
            )
            with open(data_path, "a", encoding="utf-8") as data_file:
                data_file.write("c,3\n")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                verify_data_freeze(config_path, temp_dir)


if __name__ == "__main__":
    unittest.main()
