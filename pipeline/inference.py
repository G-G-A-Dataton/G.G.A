"""Manifest-verified, group-safe batch inference for the G.G.A submission."""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.candidate_sampling import CANDIDATE_SAMPLING_SCHEMA_VERSION
from src.context_features import CONTEXT_FEATURE_SCHEMA_VERSION, add_context_features
from src.data import load_items, load_terms
from src.features import FEATURE_SCHEMA_VERSION, build_features
from src.modeling import MODEL_FEATURE_COLS
from src.out_of_core_features import (
    build_base_feature_store,
    build_context_feature_store,
    load_feature_batch,
    remove_feature_stores,
)
from src.tfidf_features import add_tfidf_features, load_vectorizer
from src.validate_submission import EXPECTED_ROWS, validate_submission


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATHS = [os.path.join(OUTPUT_DIR, f"lgbm_v2_fold_{i}.txt") for i in range(1, 6)]
VEC_PATH = os.path.join(OUTPUT_DIR, "tfidf_vectorizer_v2.pkl")
THRESH_PATH = os.path.join(OUTPUT_DIR, "best_threshold_v2.txt")
OOF_PATH = os.path.join(OUTPUT_DIR, "oof_preds_v2.npy")
THRESHOLD_REPORT_PATH = os.path.join(OUTPUT_DIR, "threshold_report_v2.json")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "model_manifest_v2.json")
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, "submission_v2.csv")
EXPECTED_POSITIVE_ROWS = 250_000
EXPECTED_TRAINING_ROWS = 1_877_700
EXPECTED_TRAINING_TERMS = 17_968


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="G.G.A end-to-end prediction pipeline")
    parser.add_argument("--mode", choices=["predict"], default="predict")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Write the first N rows after scoring every intersected term completely",
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def configure_logging():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(OUTPUT_DIR, "pipeline.log"), encoding="utf-8"
            ),
        ],
        force=True,
    )
    return logging.getLogger("G.G.A.Pipeline")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_dependencies():
    required = [
        *MODEL_PATHS,
        VEC_PATH,
        THRESH_PATH,
        OOF_PATH,
        THRESHOLD_REPORT_PATH,
        MANIFEST_PATH,
    ]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing inference artifacts:\n"
            f"{formatted}\n"
            "Run: python scripts/training/run_train_full_v2.py"
        )

    with open(MANIFEST_PATH, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    errors = []
    if manifest.get("artifact_schema_version") != 2:
        errors.append("unsupported artifact_schema_version")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        errors.append("feature_schema_version does not match the current code")
    if (
        manifest.get("context_feature_schema_version")
        != CONTEXT_FEATURE_SCHEMA_VERSION
    ):
        errors.append("context feature schema does not match the current code")
    if (
        manifest.get("candidate_sampling_schema_version")
        != CANDIDATE_SAMPLING_SCHEMA_VERSION
    ):
        errors.append("candidate sampling schema does not match the current code")
    if manifest.get("training_mode") != "full":
        errors.append("training_mode must be 'full' for production inference")
    if manifest.get("feature_columns") != MODEL_FEATURE_COLS:
        errors.append("feature_columns do not match the current feature contract")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("code_revision", ""))):
        errors.append("code_revision is not a Git commit SHA")

    expected_validation = {
        "splitter": "StratifiedGroupKFold",
        "group_column": "term_id",
        "n_splits": 5,
        "random_state": 42,
        "threshold_selection": "cross_fitted",
    }
    if manifest.get("validation") != expected_validation:
        errors.append("validation contract is not grouped and cross-fitted")

    sampling = manifest.get("candidate_sampling", {})
    expected_sampling = {
        "strategy": "test_shaped_category_random",
        "min_candidates": 100,
        "dense_multiplier": 2.0,
        "category_hard_fraction": 0.5,
        "random_state": 42,
        "positive_reference_rows": EXPECTED_POSITIVE_ROWS,
    }
    if sampling != expected_sampling:
        errors.append("candidate sampling contract is invalid")

    training = manifest.get("training", {})
    expected_counts = {
        "terms": EXPECTED_TRAINING_TERMS,
        "rows": EXPECTED_TRAINING_ROWS,
        "positive_rows": EXPECTED_POSITIVE_ROWS,
        "negative_rows": EXPECTED_TRAINING_ROWS - EXPECTED_POSITIVE_ROWS,
    }
    if any(training.get(key) != value for key, value in expected_counts.items()):
        errors.append("full training row counts do not match the production contract")

    manifest_threshold = manifest.get("threshold")
    if (
        isinstance(manifest_threshold, bool)
        or not isinstance(manifest_threshold, (int, float))
        or not 0.0 <= manifest_threshold <= 1.0
    ):
        errors.append("manifest threshold is outside [0, 1]")
    metrics = manifest.get("metrics", {})
    if not isinstance(metrics.get("cross_fitted_macro_f1"), (int, float)):
        errors.append("cross-fitted validation metric is missing")

    expected_artifacts = {
        "models": [os.path.basename(path) for path in MODEL_PATHS],
        "vectorizer": os.path.basename(VEC_PATH),
        "threshold": os.path.basename(THRESH_PATH),
        "oof_predictions": os.path.basename(OOF_PATH),
        "threshold_report": os.path.basename(THRESHOLD_REPORT_PATH),
    }
    if manifest.get("artifacts") != expected_artifacts:
        errors.append("artifact filenames do not match the production contract")

    expected_hashes = manifest.get("sha256", {})
    for path in required[:-1]:
        filename = os.path.basename(path)
        expected_hash = expected_hashes.get(filename)
        if not expected_hash or sha256_file(path) != expected_hash:
            errors.append(f"SHA-256 mismatch: {filename}")

    source_hashes = manifest.get("source_data_sha256", {})
    expected_source_names = {"terms.csv", "items.csv", "training_pairs.csv"}
    if set(source_hashes) != expected_source_names or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value))
        for value in source_hashes.values()
    ):
        errors.append("source data SHA-256 contract is invalid")

    if errors:
        raise ValueError("Invalid inference artifact manifest: " + "; ".join(errors))
    return manifest


def verify_source_data_hashes(manifest):
    """Reject inference against source data different from the training freeze."""
    for filename, expected_hash in manifest["source_data_sha256"].items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path) or sha256_file(path) != expected_hash:
            raise ValueError(f"Source data SHA-256 mismatch: {filename}")


def load_threshold(override=None):
    if override is not None:
        threshold = override
    else:
        with open(THRESH_PATH, encoding="utf-8") as threshold_file:
            threshold = float(threshold_file.read().strip())
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")
    return threshold


def resolve_output_path(args):
    if args.output:
        return os.path.abspath(args.output)
    if args.sample:
        return os.path.join(OUTPUT_DIR, f"submission_v2_sample_{args.sample}.csv")
    return DEFAULT_OUTPUT


def publish_submission(submission, output_path, sample_path=None):
    """Validate a temporary CSV before atomically replacing the public output."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temporary_output = output_path + ".tmp"
    try:
        submission.to_csv(temporary_output, index=False)
        if not validate_submission(
            temporary_output,
            sample_submission_path=sample_path,
            expected_rows=len(submission),
            verbose=True,
        ):
            raise RuntimeError(f"Submission validation failed: {output_path}")
        os.replace(temporary_output, output_path)
    except Exception:
        if os.path.exists(temporary_output):
            os.remove(temporary_output)
        raise


def iter_complete_term_batches(chunks):
    """Yield batches without splitting a contiguous term_id candidate group."""
    carry = None
    completed_terms = set()
    for chunk in chunks:
        if chunk.empty:
            continue
        data = (
            chunk.reset_index(drop=True)
            if carry is None
            else pd.concat([carry, chunk], ignore_index=True)
        )
        if data["term_id"].isna().any():
            raise ValueError("submission_pairs.csv contains null term_id values")
        run_starts = data["term_id"].ne(data["term_id"].shift())
        run_terms = data.loc[run_starts, "term_id"]
        if run_terms.duplicated().any():
            raise ValueError("submission_pairs.csv term_id groups are not contiguous")

        last_term = data["term_id"].iloc[-1]
        boundary = data["term_id"].ne(last_term)
        ready = data.loc[boundary].copy()
        carry = data.loc[~boundary].copy()
        ready_terms = set(ready["term_id"].drop_duplicates().tolist())
        if ready_terms & completed_terms:
            raise ValueError("submission_pairs.csv term_id groups reappear later")
        if not ready.empty:
            completed_terms.update(ready_terms)
            yield ready.reset_index(drop=True)

    if carry is not None and not carry.empty:
        last_terms = set(carry["term_id"].drop_duplicates().tolist())
        if last_terms & completed_terms:
            raise ValueError("submission_pairs.csv term_id groups reappear later")
        yield carry.reset_index(drop=True)


def run_prediction_pipeline(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.sample is not None and args.sample <= 0:
        raise ValueError("--sample must be positive")
    if args.sample is not None and args.sample > EXPECTED_ROWS:
        raise ValueError(f"--sample cannot exceed {EXPECTED_ROWS}")
    if (
        args.sample is not None
        and args.output
        and os.path.realpath(args.output) == os.path.realpath(DEFAULT_OUTPUT)
    ):
        raise ValueError("Sample inference cannot overwrite the production submission")

    logger = configure_logging()
    started_at = time.time()
    logger.info("G.G.A prediction pipeline started.")
    manifest = check_dependencies()
    verify_source_data_hashes(manifest)

    logger.info("Step 1/5: Loading terms and items.")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df = load_items(os.path.join(DATA_DIR, "items.csv"))
    pairs_path = os.path.join(DATA_DIR, "submission_pairs.csv")

    logger.info("Step 2/5: Loading models, vectorizer, and threshold.")
    models = [lgb.Booster(model_file=path) for path in MODEL_PATHS]
    vectorizer = load_vectorizer(VEC_PATH)
    threshold = load_threshold(args.threshold)
    if args.threshold is None and threshold != manifest["threshold"]:
        raise ValueError("Threshold file does not match model manifest")
    for model_path, model in zip(MODEL_PATHS, models):
        model_features = model.feature_name()
        if model_features and model_features != MODEL_FEATURE_COLS:
            raise ValueError(
                f"Feature contract mismatch in {model_path}: "
                f"expected {MODEL_FEATURE_COLS}, got {model_features}"
            )

    logger.info("Step 3/5: Building disk-backed global context features.")
    output_path = resolve_output_path(args)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temporary_output = output_path + ".tmp"
    expected_rows = args.sample or EXPECTED_ROWS
    store_prefix = os.path.join(
        OUTPUT_DIR, f".inference_features_{os.getpid()}"
    )
    store_paths = []
    id_reader = None
    try:
        base_path, codes_path = build_base_feature_store(
            pairs_path,
            terms_df,
            items_df,
            vectorizer,
            row_count=expected_rows,
            batch_size=args.batch_size,
            output_prefix=store_prefix,
        )
        store_paths.extend([base_path, codes_path])
        context_path = build_context_feature_store(
            base_path, codes_path, store_prefix
        )
        store_paths.append(context_path)
        base_store = np.load(base_path, mmap_mode="r")
        context_store = np.load(context_path, mmap_mode="r")
        id_reader = pd.read_csv(
            pairs_path,
            usecols=["id"],
            dtype={"id": "string"},
            nrows=expected_rows,
            chunksize=args.batch_size,
        )
        processed_rows = 0
        for id_batch in id_reader:
            batch_started_at = time.time()
            end = processed_rows + len(id_batch)
            model_batch = load_feature_batch(
                base_store, context_store, processed_rows, end
            )
            probabilities = np.mean(
                [model.predict(model_batch) for model in models], axis=0
            )
            predictions = (probabilities >= threshold).astype(np.int8)
            pd.DataFrame(
                {
                    "id": id_batch["id"].to_numpy(),
                    "prediction": predictions,
                }
            ).to_csv(
                temporary_output,
                mode="w" if processed_rows == 0 else "a",
                header=processed_rows == 0,
                index=False,
            )
            processed_rows = end
            duration = max(time.time() - batch_started_at, 1e-9)
            logger.info(
                "Processed: %s / %s (%.1f%%) | %.0f rows/s",
                f"{processed_rows:,}",
                f"{expected_rows:,}",
                min(processed_rows / expected_rows * 100, 100.0),
                len(id_batch) / duration,
            )
        id_reader.close()

        if processed_rows != expected_rows:
            raise ValueError(
                f"Submission row count mismatch: {processed_rows:,} != {expected_rows:,}"
            )

        logger.info("Step 4/5: Validating candidate submission.")
        if not validate_submission(
            temporary_output,
            sample_submission_path=os.path.join(DATA_DIR, "sample_submission.csv"),
            expected_rows=expected_rows,
            verbose=True,
        ):
            raise RuntimeError(f"Submission validation failed: {output_path}")
        os.replace(temporary_output, output_path)
    except Exception:
        if os.path.exists(temporary_output):
            os.remove(temporary_output)
        raise
    finally:
        if id_reader is not None:
            id_reader.close()
        remove_feature_stores(*store_paths)

    logger.info(
        "Step 5/5: Pipeline completed in %.1f seconds. Output: %s",
        time.time() - started_at,
        output_path,
    )
    return output_path


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "predict":
        return run_prediction_pipeline(args)
    return None


if __name__ == "__main__":
    main()
