"""Canonical grouped LightGBM training pipeline for production artifacts."""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.candidate_sampling import (
    CANDIDATE_SAMPLING_SCHEMA_VERSION,
    build_test_shaped_training_set,
    candidate_distribution,
    sample_complete_terms,
)
from src.context_features import CONTEXT_FEATURE_SCHEMA_VERSION, add_context_features
from src.data import load_items, load_terms
from src.error_analysis import generate_error_report
from src.features import FEATURE_SCHEMA_VERSION, build_features
from src.metrics import macro_f1_from_proba
from src.modeling import (
    MODEL_FEATURE_COLS,
    build_group_fold_ids,
    cross_fitted_threshold_evaluation,
)
from src.tfidf_features import (
    add_tfidf_features,
    build_tfidf_vectorizer,
    save_vectorizer,
)


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
EXPECTED_POSITIVE_ROWS = 250_000
EXPECTED_TRAINING_ROWS = 1_877_700
RANDOM_SEED = 42
N_SPLITS = 5
MIN_CANDIDATES = 100
DENSE_MULTIPLIER = 2.0
CATEGORY_HARD_FRACTION = 0.5
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM = (1, 1)

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.04,
    "num_leaves": 63,
    "min_child_samples": 100,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "reg_alpha": 0.2,
    "reg_lambda": 1.5,
    "max_bin": 255,
    "seed": RANDOM_SEED,
    "feature_fraction_seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,
    "data_random_seed": RANDOM_SEED,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": -1,
    "verbosity": -1,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train canonical G.G.A artifacts")
    parser.add_argument(
        "--sample-terms",
        "--sample",
        dest="sample_terms",
        type=int,
        default=None,
        help="Train on N complete term_id groups; omitted means all groups",
    )
    parser.add_argument("--min-candidates", type=int, default=MIN_CANDIDATES)
    parser.add_argument("--dense-multiplier", type=float, default=DENSE_MULTIPLIER)
    parser.add_argument(
        "--category-hard-fraction", type=float, default=CATEGORY_HARD_FRACTION
    )
    parser.add_argument("--num-boost-round", type=int, default=1_500)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--no-error-analysis", action="store_true")
    parser.add_argument("--artifact-dir", default=None)
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def atomic_save_npy(path, values):
    temporary_path = path + ".tmp"
    with open(temporary_path, "wb") as output_file:
        np.save(output_file, values, allow_pickle=False)
    os.replace(temporary_path, path)


def atomic_write_text(path, value):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        output_file.write(value)
    os.replace(temporary_path, path)


def write_artifact_manifest(
    *,
    artifact_dir,
    feature_cols,
    model_paths,
    vectorizer_path,
    threshold_path,
    oof_path,
    threshold_report_path,
    threshold,
    training_mode,
    candidate_config,
    candidate_stats,
    positive_reference_rows,
    source_data_sha256,
    validation_report,
    revision,
):
    artifact_paths = [
        *model_paths,
        vectorizer_path,
        threshold_path,
        oof_path,
        threshold_report_path,
    ]
    missing = [path for path in artifact_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Cannot manifest missing artifacts: {missing}")
    manifest = {
        "artifact_schema_version": 2,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "context_feature_schema_version": CONTEXT_FEATURE_SCHEMA_VERSION,
        "candidate_sampling_schema_version": CANDIDATE_SAMPLING_SCHEMA_VERSION,
        "training_mode": training_mode,
        "code_revision": revision,
        "feature_columns": feature_cols,
        "validation": {
            "splitter": "StratifiedGroupKFold",
            "group_column": "term_id",
            "n_splits": N_SPLITS,
            "random_state": RANDOM_SEED,
            "threshold_selection": "cross_fitted",
        },
        "candidate_sampling": {
            "strategy": "test_shaped_category_random",
            **candidate_config,
            "positive_reference_rows": int(positive_reference_rows),
        },
        "training": candidate_stats,
        "metrics": validation_report,
        "threshold": float(threshold),
        "source_data_sha256": source_data_sha256,
        "artifacts": {
            "models": [os.path.basename(path) for path in model_paths],
            "vectorizer": os.path.basename(vectorizer_path),
            "threshold": os.path.basename(threshold_path),
            "oof_predictions": os.path.basename(oof_path),
            "threshold_report": os.path.basename(threshold_report_path),
        },
        "sha256": {
            os.path.basename(path): sha256_file(path) for path in artifact_paths
        },
    }
    manifest_path = os.path.join(artifact_dir, "model_manifest_v2.json")
    temporary_path = manifest_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")
    os.replace(temporary_path, manifest_path)
    return manifest_path


def _sample_terms_from_args(args):
    return getattr(args, "sample_terms", getattr(args, "sample", None))


def _validate_args(args):
    sample_terms = _sample_terms_from_args(args)
    if sample_terms is not None and sample_terms <= 0:
        raise ValueError("--sample-terms must be positive")
    for name in ("num_boost_round", "early_stopping_rounds"):
        value = getattr(args, name, 1)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    return sample_terms


def main(argv=None):
    args = parse_args(argv) if argv is not None else parse_args()
    sample_terms = _validate_args(args)
    artifact_dir = os.path.abspath(
        getattr(args, "artifact_dir", None)
        or (
            os.path.join(OUTPUT_DIR, "sample_artifacts_v2")
            if sample_terms
            else OUTPUT_DIR
        )
    )
    if sample_terms and os.path.realpath(artifact_dir) == os.path.realpath(OUTPUT_DIR):
        raise ValueError("Sample training cannot write to the production artifact directory")
    os.makedirs(artifact_dir, exist_ok=True)

    print("=" * 72)
    print("G.G.A canonical training: test-shaped candidates + grouped OOF")
    print("=" * 72)
    paths = {
        "terms.csv": os.path.join(DATA_DIR, "terms.csv"),
        "items.csv": os.path.join(DATA_DIR, "items.csv"),
        "training_pairs.csv": os.path.join(DATA_DIR, "training_pairs.csv"),
    }

    print("[1/7] Loading and validating source data")
    terms_df = load_terms(paths["terms.csv"])
    items_df = load_items(paths["items.csv"])
    positives = pd.read_csv(
        paths["training_pairs.csv"],
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"},
    )
    if (
        positives.columns.tolist() != ["id", "term_id", "item_id", "label"]
        or len(positives) != EXPECTED_POSITIVE_ROWS
        or not (positives["label"] == 1).all()
        or positives.duplicated(["term_id", "item_id"]).any()
    ):
        raise ValueError("training_pairs.csv does not satisfy the positive-pair contract")
    selected_positives = (
        sample_complete_terms(positives, sample_terms, RANDOM_SEED)
        if sample_terms
        else positives.copy()
    )

    print("[2/7] Building leakage-free test-shaped candidate sets")
    candidate_config = {
        "min_candidates": int(getattr(args, "min_candidates", MIN_CANDIDATES)),
        "dense_multiplier": float(
            getattr(args, "dense_multiplier", DENSE_MULTIPLIER)
        ),
        "category_hard_fraction": float(
            getattr(args, "category_hard_fraction", CATEGORY_HARD_FRACTION)
        ),
        "random_state": RANDOM_SEED,
    }
    candidates = build_test_shaped_training_set(
        selected_positives,
        items_df,
        positive_reference_df=positives,
        verbose=True,
        **{key: value for key, value in candidate_config.items() if key != "random_state"},
        random_state=RANDOM_SEED,
    )
    stats = candidate_distribution(candidates)
    if not sample_terms and stats["rows"] != EXPECTED_TRAINING_ROWS:
        raise RuntimeError(
            f"Full candidate row mismatch: {stats['rows']:,} != {EXPECTED_TRAINING_ROWS:,}"
        )

    print("[3/7] Merging references and computing lexical features")
    merged = candidates.merge(
        terms_df, on="term_id", how="left", validate="many_to_one"
    ).merge(items_df, on="item_id", how="left", validate="many_to_one")
    if merged[["query", "title"]].isna().any().any():
        raise ValueError("Candidate set contains unresolved term_id or item_id values")
    merged = build_features(merged, copy=False)

    print("[4/7] Fitting TF-IDF and computing candidate-relative features")
    vectorizer_path = os.path.join(artifact_dir, "tfidf_vectorizer_v2.pkl")
    vectorizer = build_tfidf_vectorizer(
        terms_df,
        items_df,
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM,
        min_df=2,
    )
    save_vectorizer(vectorizer, vectorizer_path)
    merged = add_tfidf_features(merged, vectorizer, copy=False)
    merged = add_context_features(merged, copy=False)

    X = merged[MODEL_FEATURE_COLS].astype("float32")
    y = merged["label"].to_numpy(dtype=np.int8)
    groups = merged["term_id"].to_numpy()
    fold_ids = build_group_fold_ids(y, groups, n_splits=N_SPLITS, random_state=RANDOM_SEED)

    print("[5/7] Training five grouped LightGBM folds")
    oof_predictions = np.zeros(len(y), dtype=np.float32)
    models = []
    model_paths = []
    fold_default_scores = []
    for fold in range(N_SPLITS):
        training_index = np.flatnonzero(fold_ids != fold)
        validation_index = np.flatnonzero(fold_ids == fold)
        train_set = lgb.Dataset(X.iloc[training_index], label=y[training_index])
        validation_set = lgb.Dataset(
            X.iloc[validation_index], label=y[validation_index], reference=train_set
        )
        model = lgb.train(
            LGBM_PARAMS,
            train_set,
            num_boost_round=getattr(args, "num_boost_round", 1_500),
            valid_sets=[validation_set],
            callbacks=[
                lgb.early_stopping(
                    getattr(args, "early_stopping_rounds", 100), verbose=False
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        probabilities = model.predict(
            X.iloc[validation_index], num_iteration=model.best_iteration
        )
        oof_predictions[validation_index] = probabilities
        score = macro_f1_from_proba(y[validation_index], probabilities, 0.5)
        fold_default_scores.append(float(score))
        model_path = os.path.join(artifact_dir, f"lgbm_v2_fold_{fold + 1}.txt")
        temporary_model_path = model_path + ".tmp"
        model.save_model(temporary_model_path, num_iteration=model.best_iteration)
        os.replace(temporary_model_path, model_path)
        models.append(model)
        model_paths.append(model_path)
        print(
            f"  fold={fold + 1} rows={len(validation_index):,} "
            f"f1@0.5={score:.6f} best_iteration={model.best_iteration}"
        )

    print("[6/7] Cross-fitting threshold and writing verifiable artifacts")
    threshold_report = cross_fitted_threshold_evaluation(y, oof_predictions, fold_ids)
    threshold_report["fold_default_macro_f1"] = fold_default_scores
    threshold = threshold_report["deploy_threshold"]
    oof_path = os.path.join(artifact_dir, "oof_preds_v2.npy")
    threshold_path = os.path.join(artifact_dir, "best_threshold_v2.txt")
    threshold_report_path = os.path.join(artifact_dir, "threshold_report_v2.json")
    atomic_save_npy(oof_path, oof_predictions)
    atomic_write_text(threshold_path, f"{threshold:.17g}\n")
    atomic_write_text(
        threshold_report_path,
        json.dumps(threshold_report, ensure_ascii=False, indent=2) + "\n",
    )

    importance = np.mean(
        [model.feature_importance(importance_type="gain") for model in models], axis=0
    )
    pd.DataFrame(
        {"feature": MODEL_FEATURE_COLS, "importance_gain": importance}
    ).sort_values("importance_gain", ascending=False).to_csv(
        os.path.join(artifact_dir, "feature_importance_v2.csv"), index=False
    )
    if not getattr(args, "no_error_analysis", False):
        generate_error_report(
            merged,
            oof_predictions,
            threshold=threshold,
            output_path=os.path.join(artifact_dir, "error_report_v2.md"),
        )

    source_hashes = {name: sha256_file(path) for name, path in paths.items()}
    manifest_path = write_artifact_manifest(
        artifact_dir=artifact_dir,
        feature_cols=MODEL_FEATURE_COLS,
        model_paths=model_paths,
        vectorizer_path=vectorizer_path,
        threshold_path=threshold_path,
        oof_path=oof_path,
        threshold_report_path=threshold_report_path,
        threshold=threshold,
        training_mode="sample" if sample_terms else "full",
        candidate_config=candidate_config,
        candidate_stats=stats,
        positive_reference_rows=len(positives),
        source_data_sha256=source_hashes,
        validation_report=threshold_report,
        revision=git_revision(),
    )

    print("[7/7] Training complete")
    print(f"  candidates: {len(y):,}")
    print(f"  cross-fitted Macro-F1: {threshold_report['cross_fitted_macro_f1']:.6f}")
    print(f"  deploy threshold: {threshold:.8f}")
    print(f"  manifest: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    main()
