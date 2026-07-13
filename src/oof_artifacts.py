"""Integrity contract for grouped out-of-fold prediction artifacts."""

import hashlib
import json
import os

from src.features import FEATURE_COLS, FEATURE_SCHEMA_VERSION


OOF_FILENAMES = [
    "oof_lgbm.npy",
    "test_lgbm.npy",
    "oof_xgb.npy",
    "test_xgb.npy",
    "y_true.npy",
    "test_metadata.csv",
]
OOF_MANIFEST = "oof_manifest.json"
EXPECTED_POSITIVE_ROWS = 250_000
EXPECTED_TEST_ROWS = 3_359_679


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_oof_manifest(
    output_dir,
    feature_columns,
    training_mode,
    test_mode,
    training_rows,
    test_rows,
    negative_ratio,
    positive_rows,
    positive_reference_rows,
):
    paths = [os.path.join(output_dir, filename) for filename in OOF_FILENAMES]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Cannot manifest missing OOF artifacts: {missing}")
    manifest = {
        "artifact_schema_version": 1,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "validation": {
            "splitter": "StratifiedGroupKFold",
            "group_column": "term_id",
            "n_splits": 5,
            "random_state": 42,
        },
        "training_mode": training_mode,
        "test_mode": test_mode,
        "feature_columns": feature_columns,
        "training_rows": int(training_rows),
        "test_rows": int(test_rows),
        "negative_sampling": {
            "strategy": "bm25_random_fallback",
            "ratio": int(negative_ratio),
            "positive_rows": int(positive_rows),
            "positive_reference_rows": int(positive_reference_rows),
        },
        "files": OOF_FILENAMES,
        "sha256": {
            os.path.basename(path): sha256_file(path) for path in paths
        },
    }
    manifest_path = os.path.join(output_dir, OOF_MANIFEST)
    temporary_path = manifest_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")
    os.replace(temporary_path, manifest_path)
    return manifest_path


def validate_oof_artifacts(output_dir, require_full=False):
    manifest_path = os.path.join(output_dir, OOF_MANIFEST)
    paths = [os.path.join(output_dir, filename) for filename in OOF_FILENAMES]
    missing = [path for path in [manifest_path, *paths] if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing OOF artifacts: {missing}")
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    errors = []
    if manifest.get("artifact_schema_version") != 1:
        errors.append("unsupported artifact schema")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        errors.append("feature schema mismatch")
    if require_full and manifest.get("feature_columns") != FEATURE_COLS + [
        "tfidf_cosine"
    ]:
        errors.append("production feature columns do not match the current contract")
    expected_validation = {
        "splitter": "StratifiedGroupKFold",
        "group_column": "term_id",
        "n_splits": 5,
        "random_state": 42,
    }
    if manifest.get("validation") != expected_validation:
        errors.append("OOF predictions are not term_id-grouped")
    if manifest.get("files") != OOF_FILENAMES:
        errors.append("OOF file contract mismatch")
    sampling = manifest.get("negative_sampling", {})
    if (
        sampling.get("strategy") != "bm25_random_fallback"
        or sampling.get("ratio") != 3
        or sampling.get("positive_reference_rows", 0) <= 0
    ):
        errors.append("OOF negative sampling contract mismatch")
    for field in ("training_rows", "test_rows"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{field} must be a positive integer")
    if require_full and (
        manifest.get("training_mode") != "full"
        or manifest.get("test_mode") != "full"
        or sampling.get("positive_rows") != EXPECTED_POSITIVE_ROWS
        or sampling.get("positive_reference_rows") != EXPECTED_POSITIVE_ROWS
        or manifest.get("training_rows") != EXPECTED_POSITIVE_ROWS * 4
        or manifest.get("test_rows") != EXPECTED_TEST_ROWS
    ):
        errors.append("full training and test predictions are required")
    for path in paths:
        filename = os.path.basename(path)
        expected_hash = manifest.get("sha256", {}).get(filename)
        if not expected_hash or sha256_file(path) != expected_hash:
            errors.append(f"SHA-256 mismatch: {filename}")
    if errors:
        raise ValueError("Invalid OOF artifact manifest: " + "; ".join(errors))
    return manifest
