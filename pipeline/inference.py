"""End-to-end batch inference for the G.G.A submission pipeline."""

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_items, load_terms
from src.features import FEATURE_COLS, FEATURE_SCHEMA_VERSION, build_features
from src.tfidf_features import add_tfidf_features, load_vectorizer
from src.validate_submission import EXPECTED_ROWS, validate_submission


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATHS = [os.path.join(OUTPUT_DIR, f"lgbm_v2_fold_{i}.txt") for i in range(1, 6)]
VEC_PATH = os.path.join(OUTPUT_DIR, "tfidf_vectorizer_v2.pkl")
THRESH_PATH = os.path.join(OUTPUT_DIR, "best_threshold_v2.txt")
MANIFEST_PATH = os.path.join(OUTPUT_DIR, "model_manifest_v2.json")
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, "submission_v2.csv")
EXPECTED_POSITIVE_ROWS = 250_000


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="G.G.A end-to-end prediction pipeline")
    parser.add_argument("--mode", choices=["predict"], default="predict")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process the first N submission rows without overwriting the full submission",
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
    required = MODEL_PATHS + [VEC_PATH, THRESH_PATH, MANIFEST_PATH]
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
    expected_features = FEATURE_COLS + ["tfidf_cosine"]
    errors = []
    if manifest.get("artifact_schema_version") != 1:
        errors.append("unsupported artifact_schema_version")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        errors.append("feature_schema_version does not match the current code")
    if manifest.get("training_mode") != "full":
        errors.append("training_mode must be 'full' for production inference")
    if manifest.get("feature_columns") != expected_features:
        errors.append("feature_columns do not match the current feature contract")
    expected_validation = {
        "splitter": "StratifiedGroupKFold",
        "group_column": "term_id",
        "n_splits": 5,
        "random_state": 42,
    }
    if manifest.get("validation") != expected_validation:
        errors.append("validation contract is not grouped by term_id")

    negative_sampling = manifest.get("negative_sampling", {})
    if (
        negative_sampling.get("strategy") != "bm25_random_fallback"
        or negative_sampling.get("ratio") != 3
        or negative_sampling.get("positive_reference_rows") != EXPECTED_POSITIVE_ROWS
    ):
        errors.append("negative sampling contract is invalid")
    expected_training = {
        "positive_rows": EXPECTED_POSITIVE_ROWS,
        "negative_rows": EXPECTED_POSITIVE_ROWS * 3,
        "total_rows": EXPECTED_POSITIVE_ROWS * 4,
    }
    if manifest.get("training") != expected_training:
        errors.append("full training row counts do not match the production contract")

    manifest_threshold = manifest.get("threshold")
    if (
        isinstance(manifest_threshold, bool)
        or not isinstance(manifest_threshold, (int, float))
        or not 0.0 <= manifest_threshold <= 1.0
    ):
        errors.append("manifest threshold is outside [0, 1]")

    expected_artifacts = {
        "models": [os.path.basename(path) for path in MODEL_PATHS],
        "vectorizer": os.path.basename(VEC_PATH),
        "threshold": os.path.basename(THRESH_PATH),
    }
    if manifest.get("artifacts") != expected_artifacts:
        errors.append("artifact filenames do not match the production contract")

    expected_hashes = manifest.get("sha256", {})
    for path in MODEL_PATHS + [VEC_PATH, THRESH_PATH]:
        filename = os.path.basename(path)
        expected_hash = expected_hashes.get(filename)
        if not expected_hash or sha256_file(path) != expected_hash:
            errors.append(f"SHA-256 mismatch: {filename}")

    if errors:
        raise ValueError("Invalid inference artifact manifest: " + "; ".join(errors))
    return manifest


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

    logger.info("Step 1/5: Loading terms and items.")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df = load_items(os.path.join(DATA_DIR, "items.csv"))

    pairs_path = os.path.join(DATA_DIR, "submission_pairs.csv")
    logger.info("Step 2/5: Resolving submission pairs: %s", pairs_path)

    logger.info("Step 3/5: Loading models, vectorizer, and threshold.")
    models = [lgb.Booster(model_file=path) for path in MODEL_PATHS]
    vectorizer = load_vectorizer(VEC_PATH)
    threshold = load_threshold(args.threshold)
    feature_cols = FEATURE_COLS + ["tfidf_cosine"]
    if args.threshold is None and threshold != manifest["threshold"]:
        raise ValueError("Threshold file does not match model manifest")
    for model_path, model in zip(MODEL_PATHS, models):
        model_features = model.feature_name()
        if model_features and model_features != feature_cols:
            raise ValueError(
                f"Feature contract mismatch in {model_path}: "
                f"expected {feature_cols}, got {model_features}"
            )

    logger.info("Step 4/5: Generating features and predictions in batches.")
    pairs_reader = pd.read_csv(
        pairs_path,
        nrows=args.sample,
        chunksize=args.batch_size,
        dtype={"id": "string", "term_id": "string", "item_id": "string"},
    )
    output_path = resolve_output_path(args)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temporary_output = output_path + ".tmp"
    processed_rows = 0
    expected_rows = args.sample or EXPECTED_ROWS
    try:
        for source_batch in pairs_reader:
            batch_started_at = time.time()
            start = processed_rows
            end = start + len(source_batch)
            batch = source_batch.merge(
                terms_df, on="term_id", how="left", validate="many_to_one"
            ).merge(items_df, on="item_id", how="left", validate="many_to_one")
            if batch[["query", "title"]].isna().any().any():
                raise ValueError(
                    f"Unresolved term_id or item_id in rows {start}:{end}"
                )

            batch = build_features(batch)
            batch = add_tfidf_features(batch, vectorizer)
            X_batch = batch[feature_cols]
            probabilities = np.vstack(
                [model.predict(X_batch) for model in models]
            ).mean(axis=0)
            predictions = (probabilities >= threshold).astype(np.int8)
            pd.DataFrame(
                {"id": source_batch["id"].to_numpy(), "prediction": predictions}
            ).to_csv(
                temporary_output,
                mode="w" if start == 0 else "a",
                header=start == 0,
                index=False,
            )
            processed_rows = end

            duration = max(time.time() - batch_started_at, 1e-9)
            logger.info(
                "Processed: %s / %s (%.1f%%) | %.0f rows/s",
                f"{end:,}",
                f"{expected_rows:,}",
                min(end / expected_rows * 100, 100.0),
                len(batch) / duration,
            )

        if processed_rows == 0:
            raise ValueError("Submission pairs are empty")

        logger.info(
            "Step 5/5: Validating and publishing submission (threshold=%s).",
            threshold,
        )
        sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
        if not validate_submission(
            temporary_output,
            sample_submission_path=sample_path,
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
        pairs_reader.close()

    logger.info(
        "Pipeline completed in %.1f seconds. Output: %s",
        time.time() - started_at,
        output_path,
    )
    return output_path


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "predict":
        run_prediction_pipeline(args)


if __name__ == "__main__":
    main()
