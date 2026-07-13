"""Compare random and BM25 negatives with complete-term grouped validation."""

import argparse
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
from src.negative_sampling import build_training_set, verify_no_leakage


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "hard_neg_comparison.csv")
RANDOM_SEED = 42
NEGATIVE_RATIO = 3
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
        description="Leakage-free BM25 versus random-negative comparison"
    )
    parser.add_argument("--bm25", required=True, help="BM25 negative-pair CSV")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample-terms", type=int, default=200)
    mode.add_argument("--all-terms", action="store_true")
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def train_and_evaluate(
    positives,
    negatives,
    terms,
    items,
    strategy,
    num_boost_round,
    early_stopping_rounds,
):
    """Train one strategy and evaluate fold-external threshold selection."""
    positive_pairs = positives[["term_id", "item_id"]].copy()
    positive_pairs["label"] = 1
    negative_pairs = negatives[["term_id", "item_id"]].copy()
    negative_pairs["label"] = 0
    pairs = pd.concat([positive_pairs, negative_pairs], ignore_index=True)
    if pairs.duplicated(["term_id", "item_id"]).any():
        raise ValueError(f"{strategy} contains duplicate term-item pairs")
    pairs = pairs.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    frame = pairs.merge(
        terms, on="term_id", how="left", validate="many_to_one"
    ).merge(items, on="item_id", how="left", validate="many_to_one")
    frame = build_features(frame, copy=False)
    X = frame[FEATURE_COLS].astype("float32")
    y = frame["label"].to_numpy(dtype=np.int8)
    fold_ids = build_group_fold_ids(y, frame["term_id"].to_numpy())
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
    evaluation = cross_fitted_threshold_evaluation(y, oof, fold_ids)
    return {
        "strategy": strategy,
        "terms": int(frame["term_id"].nunique()),
        "positive_rows": int((y == 1).sum()),
        "negative_rows": int((y == 0).sum()),
        "cross_fitted_macro_f1": evaluation["cross_fitted_macro_f1"],
        "fold_macro_f1_mean": evaluation["fold_macro_f1_mean"],
        "fold_macro_f1_std": evaluation["fold_macro_f1_std"],
        "deploy_threshold": evaluation["deploy_threshold"],
        "all_oof_selection_macro_f1": evaluation["all_oof_selection_macro_f1"],
        "mean_best_iteration": float(np.mean(iterations)),
        "training_seconds": time.monotonic() - started,
    }


def _load_bm25_negatives(path, selected, positive_reference):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"BM25 negative file not found: {path}")
    negatives = pd.read_csv(
        path, usecols=["term_id", "item_id"], dtype="string"
    ).drop_duplicates(["term_id", "item_id"])
    negatives = negatives[negatives["term_id"].isin(selected["term_id"].unique())]
    if not verify_no_leakage(negatives, positive_reference):
        raise ValueError("BM25 negatives overlap known positive pairs")

    quota = selected.groupby("term_id", observed=True).size() * NEGATIVE_RATIO
    groups = []
    missing = {}
    for term_id, required in quota.items():
        candidates = negatives[negatives["term_id"] == term_id]
        if len(candidates) < required:
            missing[str(term_id)] = int(required - len(candidates))
            continue
        groups.append(
            candidates.sample(n=int(required), random_state=RANDOM_SEED)
        )
    if missing:
        raise ValueError(
            "BM25 input does not satisfy per-term quotas; first deficits: "
            f"{dict(list(missing.items())[:10])}"
        )
    return pd.concat(groups, ignore_index=True)


def _atomic_write_frame(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def main(argv=None):
    args = parse_args(argv)
    if args.sample_terms is not None and args.sample_terms < 5:
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
    selected = (
        positives
        if args.all_terms
        else sample_complete_terms(
            positives, args.sample_terms, random_state=RANDOM_SEED
        )
    )
    random_pairs = build_training_set(
        selected,
        items,
        ratio=NEGATIVE_RATIO,
        random_state=RANDOM_SEED,
        verbose=False,
        positive_reference_df=positives,
    )
    random_negatives = random_pairs[random_pairs["label"] == 0]
    bm25_negatives = _load_bm25_negatives(args.bm25, selected, positives)
    results = pd.DataFrame(
        [
            train_and_evaluate(
                selected,
                random_negatives,
                terms,
                items,
                "random",
                args.num_boost_round,
                args.early_stopping_rounds,
            ),
            train_and_evaluate(
                selected,
                bm25_negatives,
                terms,
                items,
                "bm25",
                args.num_boost_round,
                args.early_stopping_rounds,
            ),
        ]
    )
    _atomic_write_frame(results, args.output)
    print(results.to_string(index=False))
    delta = (
        results.loc[results["strategy"] == "bm25", "cross_fitted_macro_f1"].iloc[0]
        - results.loc[
            results["strategy"] == "random", "cross_fitted_macro_f1"
        ].iloc[0]
    )
    print(f"cross_fitted_bm25_delta={delta:+.6f}\ncomparison={args.output}")
    return args.output


if __name__ == "__main__":
    main()
