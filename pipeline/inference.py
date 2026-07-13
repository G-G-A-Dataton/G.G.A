"""End-to-end batch inference for the G.G.A submission pipeline."""

import argparse
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
from src.features import FEATURE_COLS, build_features
from src.tfidf_features import add_tfidf_features, load_vectorizer
from src.validate_submission import validate_submission


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATHS = [os.path.join(OUTPUT_DIR, f"lgbm_v2_fold_{i}.txt") for i in range(1, 6)]
VEC_PATH = os.path.join(OUTPUT_DIR, "tfidf_vectorizer_v2.pkl")
THRESH_PATH = os.path.join(OUTPUT_DIR, "best_threshold_v2.txt")
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, "submission_v2.csv")


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


def check_dependencies():
    required = MODEL_PATHS + [VEC_PATH, THRESH_PATH]
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Missing inference artifacts:\n"
            f"{formatted}\n"
            "Run: python scripts/training/run_train_full_v2.py"
        )


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


def run_prediction_pipeline(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.sample is not None and args.sample <= 0:
        raise ValueError("--sample must be positive")

    logger = configure_logging()
    started_at = time.time()
    logger.info("G.G.A prediction pipeline started.")
    check_dependencies()

    logger.info("Step 1/5: Loading terms and items.")
    terms_df = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items_df = load_items(os.path.join(DATA_DIR, "items.csv"))

    pairs_path = os.path.join(DATA_DIR, "submission_pairs.csv")
    logger.info("Step 2/5: Loading submission pairs: %s", pairs_path)
    sub_df = pd.read_csv(
        pairs_path,
        nrows=args.sample,
        dtype={"id": "string", "term_id": "string", "item_id": "string"},
    )
    if sub_df.empty:
        raise ValueError("Submission pairs are empty")

    logger.info("Step 3/5: Loading models, vectorizer, and threshold.")
    models = [lgb.Booster(model_file=path) for path in MODEL_PATHS]
    vectorizer = load_vectorizer(VEC_PATH)
    threshold = load_threshold(args.threshold)
    feature_cols = FEATURE_COLS + ["tfidf_cosine"]
    for model_path, model in zip(MODEL_PATHS, models):
        model_features = model.feature_name()
        if model_features and model_features != feature_cols:
            raise ValueError(
                f"Feature contract mismatch in {model_path}: "
                f"expected {feature_cols}, got {model_features}"
            )

    logger.info("Step 4/5: Generating features and predictions in batches.")
    all_predictions = []
    for start in range(0, len(sub_df), args.batch_size):
        batch_started_at = time.time()
        end = min(start + args.batch_size, len(sub_df))
        batch = sub_df.iloc[start:end].copy()
        batch = batch.merge(
            terms_df, on="term_id", how="left", validate="many_to_one"
        )
        batch = batch.merge(
            items_df, on="item_id", how="left", validate="many_to_one"
        )
        if batch[["query", "title"]].isna().any().any():
            raise ValueError(f"Unresolved term_id or item_id in rows {start}:{end}")

        batch = build_features(batch)
        batch = add_tfidf_features(batch, vectorizer)
        X_batch = batch[feature_cols]
        fold_predictions = np.vstack([model.predict(X_batch) for model in models])
        all_predictions.append(fold_predictions.mean(axis=0))

        duration = max(time.time() - batch_started_at, 1e-9)
        logger.info(
            "Processed: %s / %s (%.1f%%) | %.0f rows/s",
            f"{end:,}",
            f"{len(sub_df):,}",
            end / len(sub_df) * 100,
            len(batch) / duration,
        )

    probabilities = np.concatenate(all_predictions)
    predictions = (probabilities >= threshold).astype(np.int8)
    output_path = resolve_output_path(args)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    logger.info("Step 5/5: Writing and validating submission (threshold=%s).", threshold)
    submission = pd.DataFrame(
        {"id": sub_df["id"].to_numpy(), "prediction": predictions}
    )
    submission.to_csv(output_path, index=False)

    sample_path = None
    if args.sample is None:
        sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
    is_valid = validate_submission(
        output_path,
        sample_submission_path=sample_path,
        expected_rows=len(sub_df),
        verbose=True,
    )
    if not is_valid:
        raise RuntimeError(f"Submission validation failed: {output_path}")

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
