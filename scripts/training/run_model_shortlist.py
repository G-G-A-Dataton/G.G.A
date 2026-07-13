"""Train LightGBM/XGBoost grouped OOF models and stream test predictions."""

import argparse
import gc
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.training.run_train_full_v2 import git_revision, sha256_file
from src.candidate_sampling import (
    build_test_shaped_training_set,
    candidate_distribution,
    sample_complete_terms,
)
from src.context_features import add_context_features
from src.data import load_items, load_terms
from src.features import build_features
from src.metrics import macro_f1_from_proba
from src.modeling import (
    MODEL_FEATURE_COLS,
    build_group_fold_ids,
    cross_fitted_ensemble_evaluation,
    cross_fitted_threshold_evaluation,
)
from src.oof_artifacts import EXPECTED_TEST_ROWS, write_oof_manifest
from src.out_of_core_features import (
    build_base_feature_store,
    build_context_feature_store,
    load_feature_batch,
    remove_feature_stores,
)
from src.tfidf_features import (
    add_tfidf_features,
    build_tfidf_vectorizer,
    save_vectorizer,
)


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
PRODUCTION_ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "ensemble_artifacts")
RANDOM_SEED = 42
N_SPLITS = 5
BM25_HARD_FRACTION = 0.25
CATEGORY_HARD_FRACTION = 0.50
BM25_TOP_N = 200
BM25_MAX_DF_RATIO = 0.15

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
    "deterministic": True,
    "force_col_wise": True,
    "seed": RANDOM_SEED,
    "num_threads": -1,
    "verbosity": -1,
}
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "learning_rate": 0.04,
    "max_depth": 7,
    "min_child_weight": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.2,
    "reg_lambda": 1.5,
    "seed": RANDOM_SEED,
    "nthread": -1,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Grouped shortlist and OOF export")
    parser.add_argument(
        "--sample-terms", "--sample", dest="sample_terms", type=int, default=None
    )
    parser.add_argument("--test-sample", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--num-boost-round", type=int, default=1_200)
    parser.add_argument("--early-stopping-rounds", type=int, default=80)
    parser.add_argument(
        "--bm25-hard-fraction", type=float, default=BM25_HARD_FRACTION
    )
    parser.add_argument(
        "--category-hard-fraction", type=float, default=CATEGORY_HARD_FRACTION
    )
    parser.add_argument("--bm25-top-n", type=int, default=BM25_TOP_N)
    parser.add_argument(
        "--bm25-max-df-ratio", type=float, default=BM25_MAX_DF_RATIO
    )
    parser.add_argument("--artifact-dir", default=None)
    return parser.parse_args(argv)


def _sample_terms_from_args(args):
    value = getattr(args, "sample_terms", None)
    return value if value is not None else getattr(args, "sample", None)


def _atomic_save(path, values):
    temporary_path = path + ".tmp"
    with open(temporary_path, "wb") as output_file:
        np.save(output_file, values, allow_pickle=False)
    os.replace(temporary_path, path)


def _xgb_predict(model, matrix):
    iteration_range = (
        (0, model.best_iteration + 1)
        if getattr(model, "best_iteration", None) is not None
        else (0, 0)
    )
    return model.predict(matrix, iteration_range=iteration_range)


def prepare_training_data(args, artifact_dir, sample_terms):
    terms = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items = load_items(os.path.join(DATA_DIR, "items.csv"))
    positives = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"},
    )
    if len(positives) != 250_000 or not (positives["label"] == 1).all():
        raise ValueError("training_pairs.csv does not satisfy the positive contract")
    selected = (
        sample_complete_terms(positives, sample_terms, RANDOM_SEED)
        if sample_terms
        else positives
    )
    candidates = build_test_shaped_training_set(
        selected,
        items,
        terms_df=terms,
        positive_reference_df=positives,
        min_candidates=100,
        dense_multiplier=2.0,
        bm25_hard_fraction=getattr(
            args, "bm25_hard_fraction", BM25_HARD_FRACTION
        ),
        category_hard_fraction=getattr(
            args, "category_hard_fraction", CATEGORY_HARD_FRACTION
        ),
        bm25_top_n=getattr(args, "bm25_top_n", BM25_TOP_N),
        bm25_max_df_ratio=getattr(
            args, "bm25_max_df_ratio", BM25_MAX_DF_RATIO
        ),
        random_state=RANDOM_SEED,
    )
    merged = candidates.merge(
        terms, on="term_id", how="left", validate="many_to_one"
    ).merge(items, on="item_id", how="left", validate="many_to_one")
    merged = build_features(merged, copy=False)
    vectorizer = build_tfidf_vectorizer(
        terms, items, max_features=10_000, ngram_range=(1, 1), min_df=2
    )
    vectorizer_path = os.path.join(artifact_dir, "tfidf_vectorizer.pkl")
    save_vectorizer(vectorizer, vectorizer_path)
    merged = add_tfidf_features(merged, vectorizer, copy=False)
    merged = add_context_features(merged, copy=False)
    X = merged[MODEL_FEATURE_COLS].astype("float32")
    y = merged["label"].to_numpy(dtype=np.int8)
    groups = merged["term_id"].to_numpy()
    fold_ids = build_group_fold_ids(y, groups, n_splits=N_SPLITS, random_state=42)
    return {
        "terms": terms,
        "items": items,
        "positives": positives,
        "candidates": candidates,
        "vectorizer": vectorizer,
        "vectorizer_path": vectorizer_path,
        "X": X,
        "y": y,
        "fold_ids": fold_ids,
    }


def _train_oof(data, args, artifact_dir):
    X = data["X"]
    y = data["y"]
    fold_ids = data["fold_ids"]
    oof_lgbm = np.zeros(len(y), dtype=np.float32)
    oof_xgb = np.zeros(len(y), dtype=np.float32)
    lgbm_models = []
    xgb_models = []
    model_files = []

    for fold in range(N_SPLITS):
        train_index = np.flatnonzero(fold_ids != fold)
        validation_index = np.flatnonzero(fold_ids == fold)
        lgb_train = lgb.Dataset(X.iloc[train_index], label=y[train_index])
        lgb_validation = lgb.Dataset(
            X.iloc[validation_index], label=y[validation_index], reference=lgb_train
        )
        lgb_model = lgb.train(
            LGBM_PARAMS,
            lgb_train,
            num_boost_round=getattr(args, "num_boost_round", 1_200),
            valid_sets=[lgb_validation],
            callbacks=[
                lgb.early_stopping(
                    getattr(args, "early_stopping_rounds", 80), verbose=False
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        oof_lgbm[validation_index] = lgb_model.predict(
            X.iloc[validation_index], num_iteration=lgb_model.best_iteration
        )
        lgb_filename = f"lgbm_fold_{fold + 1}.txt"
        lgb_path = os.path.join(artifact_dir, lgb_filename)
        lgb_temp = lgb_path + ".tmp"
        lgb_model.save_model(lgb_temp, num_iteration=lgb_model.best_iteration)
        os.replace(lgb_temp, lgb_path)
        lgbm_models.append(lgb_model)
        model_files.append(lgb_filename)

        xgb_train = xgb.DMatrix(X.iloc[train_index], label=y[train_index])
        xgb_validation = xgb.DMatrix(
            X.iloc[validation_index], label=y[validation_index]
        )
        xgb_model = xgb.train(
            XGB_PARAMS,
            xgb_train,
            num_boost_round=getattr(args, "num_boost_round", 1_200),
            evals=[(xgb_validation, "validation")],
            early_stopping_rounds=getattr(args, "early_stopping_rounds", 80),
            verbose_eval=False,
        )
        oof_xgb[validation_index] = _xgb_predict(xgb_model, xgb_validation)
        xgb_filename = f"xgb_fold_{fold + 1}.json"
        xgb_path = os.path.join(artifact_dir, xgb_filename)
        xgb_temp = xgb_path + ".tmp.json"
        xgb_model.save_model(xgb_temp)
        os.replace(xgb_temp, xgb_path)
        xgb_models.append(xgb_model)
        model_files.append(xgb_filename)
        del xgb_train, xgb_validation
        gc.collect()

        print(
            f"  fold={fold + 1} "
            f"lgbm_f1={macro_f1_from_proba(y[validation_index], oof_lgbm[validation_index]):.6f} "
            f"xgb_f1={macro_f1_from_proba(y[validation_index], oof_xgb[validation_index]):.6f}"
        )

    return oof_lgbm, oof_xgb, lgbm_models, xgb_models, model_files


def _stream_test_predictions(data, models, args, artifact_dir, test_rows):
    lgbm_models, xgb_models = models
    temp_lgbm = os.path.join(artifact_dir, "test_lgbm.npy.tmp")
    temp_xgb = os.path.join(artifact_dir, "test_xgb.npy.tmp")
    test_lgbm = np.lib.format.open_memmap(
        temp_lgbm, mode="w+", dtype="float32", shape=(test_rows,)
    )
    test_xgb = np.lib.format.open_memmap(
        temp_xgb, mode="w+", dtype="float32", shape=(test_rows,)
    )
    store_prefix = os.path.join(artifact_dir, f".shortlist_features_{os.getpid()}")
    store_paths = []
    offset = 0
    try:
        base_path, codes_path = build_base_feature_store(
            os.path.join(DATA_DIR, "submission_pairs.csv"),
            data["terms"],
            data["items"],
            data["vectorizer"],
            row_count=test_rows,
            batch_size=getattr(args, "batch_size", 100_000),
            output_prefix=store_prefix,
        )
        store_paths.extend([base_path, codes_path])
        context_path = build_context_feature_store(
            base_path, codes_path, store_prefix
        )
        store_paths.append(context_path)
        base_store = np.load(base_path, mmap_mode="r")
        context_store = np.load(context_path, mmap_mode="r")
        for start in range(0, test_rows, getattr(args, "batch_size", 100_000)):
            end = min(start + getattr(args, "batch_size", 100_000), test_rows)
            X_batch = load_feature_batch(base_store, context_store, start, end)
            lgb_values = np.mean(
                [model.predict(X_batch) for model in lgbm_models], axis=0
            )
            xgb_matrix = xgb.DMatrix(X_batch)
            xgb_values = np.mean(
                [_xgb_predict(model, xgb_matrix) for model in xgb_models], axis=0
            )
            test_lgbm[start:end] = lgb_values
            test_xgb[start:end] = xgb_values
            offset = end
            print(f"  test predictions: {offset:,}/{test_rows:,}")
    finally:
        remove_feature_stores(*store_paths)
    if offset != test_rows:
        raise RuntimeError(f"Test prediction rows mismatch: {offset:,} != {test_rows:,}")
    test_lgbm.flush()
    test_xgb.flush()
    del test_lgbm, test_xgb
    os.replace(temp_lgbm, os.path.join(artifact_dir, "test_lgbm.npy"))
    os.replace(temp_xgb, os.path.join(artifact_dir, "test_xgb.npy"))


def main(argv=None):
    args = parse_args(argv) if argv is not None else parse_args()
    sample_terms = _sample_terms_from_args(args)
    test_sample = getattr(args, "test_sample", None)
    if sample_terms is not None and sample_terms <= 0:
        raise ValueError("--sample-terms must be positive")
    if test_sample is not None and test_sample <= 0:
        raise ValueError("--test-sample must be positive")
    if getattr(args, "batch_size", 1) <= 0:
        raise ValueError("--batch-size must be positive")
    artifact_dir = os.path.abspath(
        getattr(args, "artifact_dir", None)
        or (
            os.path.join(OUTPUT_DIR, "ensemble_sample_artifacts")
            if sample_terms or test_sample
            else PRODUCTION_ARTIFACT_DIR
        )
    )
    if (sample_terms or test_sample) and os.path.realpath(
        artifact_dir
    ) == os.path.realpath(PRODUCTION_ARTIFACT_DIR):
        raise ValueError("Sample OOF runs cannot overwrite production ensemble artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    test_rows = test_sample or (50_000 if sample_terms else EXPECTED_TEST_ROWS)
    if test_rows > EXPECTED_TEST_ROWS:
        raise ValueError(f"--test-sample cannot exceed {EXPECTED_TEST_ROWS}")

    print("[1/5] Preparing shared test-shaped training matrix")
    data = prepare_training_data(args, artifact_dir, sample_terms)
    print("[2/5] Training grouped LightGBM and XGBoost OOF models")
    oof_lgbm, oof_xgb, lgb_models, xgb_models, model_files = _train_oof(
        data, args, artifact_dir
    )
    print("[3/5] Evaluating thresholds and blend without fold leakage")
    lgb_report = cross_fitted_threshold_evaluation(
        data["y"], oof_lgbm, data["fold_ids"]
    )
    xgb_report = cross_fitted_threshold_evaluation(
        data["y"], oof_xgb, data["fold_ids"]
    )
    blend_report = cross_fitted_ensemble_evaluation(
        data["y"], oof_lgbm, oof_xgb, data["fold_ids"]
    )
    print(
        f"  cross-fitted F1: LGBM={lgb_report['cross_fitted_macro_f1']:.6f} "
        f"XGB={xgb_report['cross_fitted_macro_f1']:.6f} "
        f"blend={blend_report['cross_fitted_macro_f1']:.6f}"
    )

    print("[4/5] Streaming test predictions to memory-mapped arrays")
    _stream_test_predictions(
        data, (lgb_models, xgb_models), args, artifact_dir, test_rows
    )
    _atomic_save(os.path.join(artifact_dir, "oof_lgbm.npy"), oof_lgbm)
    _atomic_save(os.path.join(artifact_dir, "oof_xgb.npy"), oof_xgb)
    _atomic_save(os.path.join(artifact_dir, "y_true.npy"), data["y"])
    _atomic_save(os.path.join(artifact_dir, "fold_ids.npy"), data["fold_ids"])

    print("[5/5] Writing hash-verified shortlist manifest")
    source_names = [
        "terms.csv",
        "items.csv",
        "training_pairs.csv",
        "submission_pairs.csv",
    ]
    manifest_path = write_oof_manifest(
        output_dir=artifact_dir,
        training_mode="sample" if sample_terms else "full",
        test_mode="sample" if test_rows != EXPECTED_TEST_ROWS else "full",
        training_stats=candidate_distribution(data["candidates"]),
        test_rows=test_rows,
        candidate_config={
            "min_candidates": 100,
            "dense_multiplier": 2.0,
            "bm25_hard_fraction": getattr(
                args, "bm25_hard_fraction", BM25_HARD_FRACTION
            ),
            "category_hard_fraction": getattr(
                args, "category_hard_fraction", CATEGORY_HARD_FRACTION
            ),
            "bm25_top_n": getattr(args, "bm25_top_n", BM25_TOP_N),
            "bm25_max_df_ratio": getattr(
                args, "bm25_max_df_ratio", BM25_MAX_DF_RATIO
            ),
            "random_state": 42,
        },
        positive_reference_rows=len(data["positives"]),
        source_data_sha256={
            name: sha256_file(os.path.join(DATA_DIR, name)) for name in source_names
        },
        model_files=model_files,
        support_files=[os.path.basename(data["vectorizer_path"])],
        code_revision=git_revision(),
        feature_columns=MODEL_FEATURE_COLS,
    )
    print(f"  manifest: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    main()
