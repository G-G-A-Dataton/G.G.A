import tempfile
import unittest
from pathlib import Path

from scripts.run_reproducibility_dry_run import (
    hardlink_or_copy,
    parse_test_count,
    sha256_file,
    write_reports,
)
from scripts.verify_environment import (
    expected_lock_requirements,
    normalize_package_name,
    resolved_requirements,
)


class ReproducibilityTests(unittest.TestCase):
    def test_lock_parser_reads_only_top_level_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "requirements.lock"
            lock_path.write_text(
                "alpha==1.2.3 \\\n    --hash=sha256:abc\n"
                "    # via dependency\n"
                "beta-package==4.5.6 \\\n    --hash=sha256:def\n",
                encoding="utf-8",
            )
            self.assertEqual(
                expected_lock_requirements(lock_path),
                {"alpha": "1.2.3", "beta-package": "4.5.6"},
            )

    def test_distribution_names_use_pep_503_comparison(self):
        self.assertEqual(normalize_package_name("rank_bm25"), "rank-bm25")
        self.assertEqual(normalize_package_name("Rank.BM25"), "rank-bm25")

    def test_lock_must_preserve_direct_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "requirements.txt"
            lock = root / "requirements.lock"
            direct.write_text("alpha==1.0\n", encoding="utf-8")
            lock.write_text("alpha==2.0 --hash=sha256:abc\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not preserve"):
                resolved_requirements(direct, lock)

    def test_hardlink_or_copy_preserves_content(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            destination = Path(directory) / "nested" / "destination.bin"
            source.write_bytes(b"accepted-artifact")
            method = hardlink_or_copy(source, destination)
            self.assertIn(method, {"hardlink", "copy"})
            self.assertEqual(sha256_file(source), sha256_file(destination))

    def test_test_count_requires_unittest_summary(self):
        self.assertEqual(parse_test_count("Ran 100 tests in 1.2s\nOK"), 100)
        with self.assertRaisesRegex(ValueError, "unittest count"):
            parse_test_count("OK")

    def test_report_records_byte_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "status": "PASS",
                "source_revision": "a" * 40,
                "artifact_revision": "b" * 40,
                "python_version": "3.13.5",
                "locked_packages": 158,
                "tests_passed": 100,
                "network_access": "disabled",
                "asset_materialization": ["hardlink"],
                "accepted_submission_sha256": "c" * 64,
                "reproduced_submission_sha256": "c" * 64,
                "byte_identical": True,
                "submission_rows": 3_359_679,
                "positive_rows": 645_783,
                "steps": [{"name": "tests", "duration_seconds": 1.0}],
            }
            markdown = root / "report.md"
            json_report = root / "report.json"
            write_reports(payload, markdown, json_report)
            self.assertIn("Byte-identical: **true**", markdown.read_text())
            self.assertTrue(json_report.is_file())


if __name__ == "__main__":
    unittest.main()
