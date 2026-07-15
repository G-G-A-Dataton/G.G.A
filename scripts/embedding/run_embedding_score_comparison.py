"""Grouped ablation of manifest-verified embedding cosine features."""

import argparse
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.candidate_sampling import build_test_shaped_training_set, sample_complete_terms
from src.context_features import add_context_features
from src.data import load_items, load_terms
from src.embedding_cosine import add_embedding_cosine_feature, load_embedding_indexes
from src.features import build_features
from src.modeling import (
    MODEL_FEATURE_COLS,
    build_group_fold_ids,
    cross_fitted_threshold_evaluation,
)
from src.tfidf_features import add_tfidf_features, build_tfidf_vectorizer


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "embedding_comparison.csv")
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, "docs", "embedding_comparison.md")
RANDOM_SEED = 42
EMBEDDING_FEATURE = "embedding_cosine"
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
    "num_threads": 8,
    "verbosity": -1,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare canonical features with verified embedding cosine"
    )
    parser.add_argument("--sample-terms", type=int, default=300)
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def run_cv(frame, feature_columns, num_boost_round, early_stopping_rounds, label):
    """Evaluate one feature contract with shared grouped folds and cross-fitting."""
    missing = sorted(set(feature_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"embedding experiment is missing features: {missing}")
    X = frame[feature_columns].astype("float32")
    y = frame["label"].to_numpy(dtype=np.int8)
    fold_ids = build_group_fold_ids(
        y, frame["term_id"].to_numpy(), n_splits=5, random_state=RANDOM_SEED
    )
    oof = np.zeros(len(frame), dtype=np.float32)
    iterations = []
    started = time.monotonic()
    for fold in np.unique(fold_ids):
        train_index = np.flatnonzero(fold_ids != fold)
        validation_index = np.flatnonzero(fold_ids == fold)
        train_data = lgb.Dataset(X.iloc[train_index], label=y[train_index])
        validation_data = lgb.Dataset(
            X.iloc[validation_index],
            label=y[validation_index],
            reference=train_data,
        )
        model = lgb.train(
            LGBM_PARAMS,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[validation_data],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof[validation_index] = model.predict(
            X.iloc[validation_index], num_iteration=model.best_iteration
        )
        iterations.append(model.best_iteration)
    report = cross_fitted_threshold_evaluation(y, oof, fold_ids)
    return {
        "candidate": label,
        "feature_count": len(feature_columns),
        "cross_fitted_macro_f1": report["cross_fitted_macro_f1"],
        "fold_macro_f1_mean": report["fold_macro_f1_mean"],
        "fold_macro_f1_std": report["fold_macro_f1_std"],
        "deploy_threshold": report["deploy_threshold"],
        "all_oof_selection_macro_f1": report["all_oof_selection_macro_f1"],
        "mean_best_iteration": float(np.mean(iterations)),
        "training_seconds": time.monotonic() - started,
    }


def _atomic_write_frame(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _write_report(path, results, separation, sample_terms):
    base = results.loc[results["candidate"] == "base"].iloc[0]
    embedding = results.loc[results["candidate"] == "base_plus_embedding"].iloc[0]
    delta = embedding["cross_fitted_macro_f1"] - base["cross_fitted_macro_f1"]
    lines = [
        "# Embedding Cosine Ablation",
        "",
        "Only manifest-verified, fully covered embedding matrices are accepted. The comparison uses complete query groups and fold-external threshold selection.",
        "",
        f"- Complete sampled terms: `{sample_terms:,}`",
        f"- Cosine separation: `{separation:.6f}`",
        f"- Cross-fitted Macro-F1 delta: `{delta:+.6f}`",
        "- Production promotion requires a positive full-data ablation; this sample result alone is not sufficient.",
        "",
        "| Candidate | Features | Cross-fitted Macro-F1 | Fold std | Deploy threshold | Seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.feature_count} | "
            f"{row.cross_fitted_macro_f1:.6f} | {row.fold_macro_f1_std:.6f} | "
            f"{row.deploy_threshold:.8f} | {row.training_seconds:.1f} |"
        )
    lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
    os.replace(temporary_path, path)


def main(argv=None):
    args = parse_args(argv)
    if args.sample_terms < 5:
        raise ValueError("--sample-terms must be at least 5")
    if args.num_boost_round <= 0 or args.early_stopping_rounds <= 0:
        raise ValueError("boosting and early-stopping rounds must be positive")
    terms = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items = load_items(os.path.join(DATA_DIR, "items.csv"))
    positives = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={
            "id": "string",
            "term_id": "string",
            "item_id": "string",
            "label": "int8",
        },
    )
    selected = sample_complete_terms(
        positives, args.sample_terms, random_state=RANDOM_SEED
    )
    candidates = build_test_shaped_training_set(
        selected,
        items,
        positive_reference_df=positives,
        random_state=RANDOM_SEED,
    )
    frame = candidates.merge(
        terms, on="term_id", how="left", validate="many_to_one"
    ).merge(items, on="item_id", how="left", validate="many_to_one")
    frame = build_features(frame, copy=False)
    vectorizer = build_tfidf_vectorizer(terms, items)
    frame = add_tfidf_features(frame, vectorizer, copy=False)
    frame = add_context_features(frame, copy=False)
    term_index, item_index = load_embedding_indexes(PROJECT_ROOT)
    frame = add_embedding_cosine_feature(frame, term_index, item_index)
    separation = float(
        frame.loc[frame["label"] == 1, EMBEDDING_FEATURE].mean()
        - frame.loc[frame["label"] == 0, EMBEDDING_FEATURE].mean()
    )
    results = pd.DataFrame(
        [
            run_cv(
                frame,
                MODEL_FEATURE_COLS,
                args.num_boost_round,
                args.early_stopping_rounds,
                "base",
            ),
            run_cv(
                frame,
                [*MODEL_FEATURE_COLS, EMBEDDING_FEATURE],
                args.num_boost_round,
                args.early_stopping_rounds,
                "base_plus_embedding",
            ),
        ]
    )
    _atomic_write_frame(results, args.output)
    _write_report(args.report, results, separation, args.sample_terms)
    print(results.to_string(index=False))
    print(f"comparison={args.output}\nreport={args.report}")
    return args.output


if __name__ == "__main__":
    main()
