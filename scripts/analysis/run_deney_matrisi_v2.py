"""Run grouped negative-ratio and feature-set ablation experiments."""

import argparse
import itertools
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.candidate_sampling import sample_complete_terms
from src.data import load_items, load_terms
from src.features import FEATURE_COLS, build_features
from src.modeling import build_group_fold_ids, cross_fitted_threshold_evaluation
from src.negative_sampling import build_training_set
from src.tfidf_features import add_tfidf_features, build_tfidf_vectorizer


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "experiment_matrix_v2.csv")
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, "docs", "experiment_matrix_v2.md")
RANDOM_SEED = 42

LEXICAL_CORE = [
    "query_title_overlap",
    "query_title_coverage",
    "query_title_precision",
    "query_title_phrase",
    "query_category_overlap",
    "query_category_coverage",
    "query_brand_match",
]
FEATURE_SETS = {
    "lexical_core": LEXICAL_CORE,
    "lexical_tfidf": [
        *LEXICAL_CORE,
        "query_model_token_match",
        "query_model_token_conflict",
        "tfidf_cosine",
    ],
    "structured": [
        *LEXICAL_CORE,
        "query_cat_l1_overlap",
        "query_cat_l2_overlap",
        "query_cat_l3_overlap",
        "cat_depth",
        "gender_match",
        "age_group_match",
        "demographic_conflict",
        "query_color_match",
        "query_size_match",
        "query_material_match",
    ],
    "full": [*FEATURE_COLS, "tfidf_cosine"],
}
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Grouped feature and negative-ratio ablation matrix"
    )
    parser.add_argument("--sample-terms", type=int, default=100)
    parser.add_argument("--ratios", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def run_single(frame, feature_names, num_boost_round, early_stopping_rounds):
    """Train grouped OOF folds and report cross-fitted threshold performance."""
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError(f"experiment feature set is missing columns: {missing}")
    X = frame[feature_names].astype("float32")
    y = frame["label"].to_numpy(dtype=np.int8)
    fold_ids = build_group_fold_ids(y, frame["term_id"].to_numpy())
    oof = np.zeros(len(frame), dtype=np.float32)
    best_iterations = []
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
        best_iterations.append(model.best_iteration)
    evaluation = cross_fitted_threshold_evaluation(y, oof, fold_ids)
    return {
        "cross_fitted_macro_f1": evaluation["cross_fitted_macro_f1"],
        "fold_macro_f1_mean": evaluation["fold_macro_f1_mean"],
        "fold_macro_f1_std": evaluation["fold_macro_f1_std"],
        "deploy_threshold": evaluation["deploy_threshold"],
        "all_oof_selection_macro_f1": evaluation["all_oof_selection_macro_f1"],
        "mean_best_iteration": float(np.mean(best_iterations)),
        "feature_count": len(feature_names),
        "training_seconds": time.monotonic() - started,
    }


def _prepare_ratio_dataset(selected, positives, terms, items, vectorizer, ratio):
    pairs = build_training_set(
        selected,
        items,
        ratio=ratio,
        random_state=RANDOM_SEED,
        verbose=False,
        positive_reference_df=positives,
    )
    frame = pairs.merge(
        terms, on="term_id", how="left", validate="many_to_one"
    ).merge(items, on="item_id", how="left", validate="many_to_one")
    frame = build_features(frame, copy=False)
    return add_tfidf_features(frame, vectorizer, copy=False, verbose=False)


def _atomic_write_frame(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _write_report(results, path, sample_terms):
    best = results.loc[results["cross_fitted_macro_f1"].idxmax()]
    lines = [
        "# Grouped Experiment Matrix v2",
        "",
        "This is an ablation study. Production training uses the test-shaped candidate distribution, not a fixed negative ratio.",
        "",
        f"- Complete sampled terms: `{sample_terms:,}`",
        "- Validation: `StratifiedGroupKFold(group=term_id, n_splits=5, seed=42)`",
        "- Threshold selection: cross-fitted outside each evaluated fold",
        f"- Best diagnostic configuration: `{best['feature_set']}` at `{int(best['negative_ratio'])}:1`",
        f"- Best cross-fitted Macro-F1: `{best['cross_fitted_macro_f1']:.6f}`",
        "",
        "| Negative ratio | Feature set | Features | Cross-fitted Macro-F1 | Fold std | Deploy threshold | Seconds |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples(index=False):
        lines.append(
            f"| {row.negative_ratio}:1 | {row.feature_set} | {row.feature_count} | "
            f"{row.cross_fitted_macro_f1:.6f} | {row.fold_macro_f1_std:.6f} | "
            f"{row.deploy_threshold:.8f} | {row.training_seconds:.1f} |"
        )
    lines.extend(
        [
            "",
            "All-OOF selection scores in the CSV are reproducibility diagnostics, not unbiased validation estimates.",
            "",
        ]
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
    os.replace(temporary_path, path)


def main(argv=None):
    args = parse_args(argv)
    if args.sample_terms < 5:
        raise ValueError("--sample-terms must be at least 5")
    if not args.ratios or any(ratio <= 0 for ratio in args.ratios):
        raise ValueError("--ratios must contain positive integers")
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
    vectorizer = build_tfidf_vectorizer(
        terms, items, max_features=10_000, ngram_range=(1, 1), min_df=2
    )
    datasets = {
        ratio: _prepare_ratio_dataset(
            selected, positives, terms, items, vectorizer, ratio
        )
        for ratio in sorted(set(args.ratios))
    }

    rows = []
    experiments = list(itertools.product(datasets, FEATURE_SETS))
    for index, (ratio, feature_set) in enumerate(experiments, start=1):
        print(
            f"[{index}/{len(experiments)}] ratio={ratio}:1 "
            f"features={feature_set}"
        )
        result = run_single(
            datasets[ratio],
            FEATURE_SETS[feature_set],
            args.num_boost_round,
            args.early_stopping_rounds,
        )
        rows.append(
            {"negative_ratio": ratio, "feature_set": feature_set, **result}
        )
    results = pd.DataFrame(rows).sort_values(
        ["negative_ratio", "feature_set"]
    )
    _atomic_write_frame(results, args.output)
    _write_report(results, args.report, args.sample_terms)
    print(f"matrix={args.output}\nreport={args.report}")
    return args.output


if __name__ == "__main__":
    main()
