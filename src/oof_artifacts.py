"""Integrity contract for grouped shortlist OOF and test predictions."""

import hashlib
import json
import os
import re

import numpy as np

from src.candidate_sampling import CANDIDATE_SAMPLING_SCHEMA_VERSION
from src.context_features import CONTEXT_FEATURE_SCHEMA_VERSION
from src.features import FEATURE_SCHEMA_VERSION
from src.modeling import MODEL_FEATURE_COLS


OOF_FILENAMES = [
    "oof_lgbm.npy",
    "test_lgbm.npy",
    "oof_xgb.npy",
    "test_xgb.npy",
    "y_true.npy",
    "fold_ids.npy",
]
OOF_MANIFEST = "oof_manifest.json"
EXPECTED_POSITIVE_ROWS = 250_000
EXPECTED_TRAINING_ROWS = 1_877_700
EXPECTED_TRAINING_TERMS = 17_968
EXPECTED_TEST_ROWS = 3_359_679
PRODUCTION_CANDIDATE_SAMPLING = {
    "strategy": "test_shaped_bm25_category_random",
    "min_candidates": 100,
    "dense_multiplier": 2.0,
    "bm25_hard_fraction": 0.20,
    "category_hard_fraction": 0.5,
    "bm25_top_n": 200,
    "bm25_max_df_ratio": 0.15,
    "random_state": 42,
    "positive_reference_rows": EXPECTED_POSITIVE_ROWS,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_oof_manifest(
    *,
    output_dir,
    training_mode,
    test_mode,
    training_stats,
    test_rows,
    candidate_config,
    positive_reference_rows,
    source_data_sha256,
    model_files,
    support_files,
    code_revision,
    feature_columns=None,
):
    feature_columns = MODEL_FEATURE_COLS if feature_columns is None else feature_columns
    artifact_files = [*OOF_FILENAMES, *model_files, *support_files]
    paths = [os.path.join(output_dir, filename) for filename in artifact_files]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Cannot manifest missing OOF artifacts: {missing}")
    manifest = {
        "artifact_schema_version": 2,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "context_feature_schema_version": CONTEXT_FEATURE_SCHEMA_VERSION,
        "candidate_sampling_schema_version": CANDIDATE_SAMPLING_SCHEMA_VERSION,
        "code_revision": code_revision,
        "validation": {
            "splitter": "StratifiedGroupKFold",
            "group_column": "term_id",
            "n_splits": 5,
            "random_state": 42,
            "selection": "cross_fitted",
        },
        "training_mode": training_mode,
        "test_mode": test_mode,
        "feature_columns": feature_columns,
        "training": training_stats,
        "test_rows": int(test_rows),
        "candidate_sampling": {
            "strategy": "test_shaped_bm25_category_random",
            **candidate_config,
            "positive_reference_rows": int(positive_reference_rows),
        },
        "source_data_sha256": source_data_sha256,
        "prediction_files": OOF_FILENAMES,
        "model_files": model_files,
        "support_files": support_files,
        "sha256": {
            filename: sha256_file(path)
            for filename, path in zip(artifact_files, paths)
        },
    }
    manifest_path = os.path.join(output_dir, OOF_MANIFEST)
    temporary_path = manifest_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")
    os.replace(temporary_path, manifest_path)
    return manifest_path


def validate_oof_artifacts(output_dir, require_full=False, source_data_dir=None):
    manifest_path = os.path.join(output_dir, OOF_MANIFEST)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing OOF manifest: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    prediction_files = manifest.get("prediction_files", [])
    model_files = manifest.get("model_files", [])
    support_files = manifest.get("support_files", [])
    paths = [
        os.path.join(output_dir, filename)
        for filename in [*prediction_files, *model_files, *support_files]
    ]
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing OOF artifacts: {missing}")

    errors = []
    if manifest.get("artifact_schema_version") != 2:
        errors.append("unsupported artifact schema")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        errors.append("feature schema mismatch")
    if (
        manifest.get("context_feature_schema_version")
        != CONTEXT_FEATURE_SCHEMA_VERSION
    ):
        errors.append("context feature schema mismatch")
    if (
        manifest.get("candidate_sampling_schema_version")
        != CANDIDATE_SAMPLING_SCHEMA_VERSION
    ):
        errors.append("candidate sampling schema mismatch")
    if manifest.get("feature_columns") != MODEL_FEATURE_COLS:
        errors.append("production feature columns do not match the current contract")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("code_revision", ""))):
        errors.append("invalid code revision")
    expected_validation = {
        "splitter": "StratifiedGroupKFold",
        "group_column": "term_id",
        "n_splits": 5,
        "random_state": 42,
        "selection": "cross_fitted",
    }
    if manifest.get("validation") != expected_validation:
        errors.append("OOF predictions are not cross-fitted by term_id")
    if prediction_files != OOF_FILENAMES:
        errors.append("OOF prediction file contract mismatch")
    if (
        not isinstance(model_files, list)
        or len(model_files) != 10
        or len(set(model_files)) != len(model_files)
    ):
        errors.append("five LightGBM and five XGBoost model files are required")
    if support_files != ["tfidf_vectorizer.pkl"]:
        errors.append("TF-IDF support artifact contract mismatch")

    sampling = manifest.get("candidate_sampling", {})
    if not _valid_candidate_sampling(sampling):
        errors.append("OOF candidate sampling contract mismatch")
    training = manifest.get("training", {})
    for field in ("terms", "rows", "positive_rows", "negative_rows"):
        value = training.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"training.{field} must be a positive integer")
    test_rows = manifest.get("test_rows")
    if isinstance(test_rows, bool) or not isinstance(test_rows, int) or test_rows <= 0:
        errors.append("test_rows must be a positive integer")

    source_hashes = manifest.get("source_data_sha256", {})
    valid_source_hashes = isinstance(source_hashes, dict) and set(source_hashes) == {
        "terms.csv",
        "items.csv",
        "training_pairs.csv",
        "submission_pairs.csv",
    } and not any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value))
        for value in source_hashes.values()
    )
    if not valid_source_hashes:
        errors.append("source data SHA-256 contract mismatch")
    if source_data_dir is not None and valid_source_hashes:
        for filename, expected_hash in source_hashes.items():
            source_path = os.path.join(source_data_dir, filename)
            if not os.path.isfile(source_path):
                errors.append(f"missing source data: {filename}")
            elif sha256_file(source_path) != expected_hash:
                errors.append(f"source data SHA-256 mismatch: {filename}")

    if require_full and sampling != PRODUCTION_CANDIDATE_SAMPLING:
        errors.append("full artifacts require the production candidate sampling config")
    if require_full and (
        manifest.get("training_mode") != "full"
        or manifest.get("test_mode") != "full"
        or training.get("terms") != EXPECTED_TRAINING_TERMS
        or training.get("rows") != EXPECTED_TRAINING_ROWS
        or training.get("positive_rows") != EXPECTED_POSITIVE_ROWS
        or training.get("negative_rows")
        != EXPECTED_TRAINING_ROWS - EXPECTED_POSITIVE_ROWS
        or test_rows != EXPECTED_TEST_ROWS
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


def _valid_candidate_sampling(sampling):
    if not isinstance(sampling, dict) or set(sampling) != set(
        PRODUCTION_CANDIDATE_SAMPLING
    ):
        return False
    fractions = (
        sampling.get("bm25_hard_fraction"),
        sampling.get("category_hard_fraction"),
    )
    return (
        sampling.get("strategy") == "test_shaped_bm25_category_random"
        and isinstance(sampling.get("min_candidates"), int)
        and not isinstance(sampling.get("min_candidates"), bool)
        and sampling["min_candidates"] > 0
        and isinstance(sampling.get("dense_multiplier"), (int, float))
        and not isinstance(sampling.get("dense_multiplier"), bool)
        and np.isfinite(sampling["dense_multiplier"])
        and sampling["dense_multiplier"] >= 1.0
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(value)
            and 0.0 <= value <= 1.0
            for value in fractions
        )
        and sum(fractions) <= 1.0
        and isinstance(sampling.get("bm25_top_n"), int)
        and not isinstance(sampling.get("bm25_top_n"), bool)
        and sampling["bm25_top_n"] > 0
        and isinstance(sampling.get("bm25_max_df_ratio"), (int, float))
        and not isinstance(sampling.get("bm25_max_df_ratio"), bool)
        and np.isfinite(sampling["bm25_max_df_ratio"])
        and 0.0 < sampling["bm25_max_df_ratio"] <= 1.0
        and isinstance(sampling.get("random_state"), int)
        and not isinstance(sampling.get("random_state"), bool)
        and sampling.get("positive_reference_rows") == EXPECTED_POSITIVE_ROWS
    )


def load_oof_artifacts(output_dir, require_full=False, source_data_dir=None):
    """Validate a shortlist artifact directory and load aligned mmap arrays."""
    manifest = validate_oof_artifacts(
        output_dir,
        require_full=require_full,
        source_data_dir=source_data_dir,
    )
    arrays = {
        filename: np.load(os.path.join(output_dir, filename), mmap_mode="r")
        for filename in OOF_FILENAMES
    }
    for filename, values in arrays.items():
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"{filename} must be a finite one-dimensional array")

    training_rows = manifest["training"]["rows"]
    test_rows = manifest["test_rows"]
    training_files = ("oof_lgbm.npy", "oof_xgb.npy", "y_true.npy", "fold_ids.npy")
    test_files = ("test_lgbm.npy", "test_xgb.npy")
    if any(len(arrays[name]) != training_rows for name in training_files):
        raise ValueError("Training array lengths do not match the OOF manifest")
    if any(len(arrays[name]) != test_rows for name in test_files):
        raise ValueError("Test array lengths do not match the OOF manifest")
    for name in ("oof_lgbm.npy", "oof_xgb.npy", "test_lgbm.npy", "test_xgb.npy"):
        if ((arrays[name] < 0) | (arrays[name] > 1)).any():
            raise ValueError(f"{name} contains values outside [0, 1]")
    if not np.isin(arrays["y_true.npy"], [0, 1]).all():
        raise ValueError("y_true.npy must contain binary labels")
    expected_folds = np.arange(manifest["validation"]["n_splits"])
    if not np.array_equal(np.unique(arrays["fold_ids.npy"]), expected_folds):
        raise ValueError("fold_ids.npy does not contain the expected folds")
    return manifest, arrays
