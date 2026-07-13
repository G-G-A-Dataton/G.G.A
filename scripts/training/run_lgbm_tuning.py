"""Tune LightGBM on the canonical test-shaped matrix with grouped OOF folds."""

import argparse
import itertools
import os
import sys
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.training.run_model_shortlist import (
    LGBM_PARAMS,
    prepare_training_data,
)
from src.modeling import cross_fitted_threshold_evaluation


DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "lgbm_tuning_grouped.csv")
DEFAULT_WORKSPACE = os.path.join(PROJECT_ROOT, "outputs", "tuning_workspace")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Grouped LightGBM tuning on test-shaped candidates"
    )
    parser.add_argument("--sample-terms", type=int, default=300)
    parser.add_argument("--num-leaves", nargs="+", type=int, default=[31, 63, 127])
    parser.add_argument(
        "--learning-rates", nargs="+", type=float, default=[0.02, 0.04, 0.08]
    )
    parser.add_argument(
        "--min-child-samples", nargs="+", type=int, default=[50, 100, 200]
    )
    parser.add_argument("--num-boost-round", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def evaluate_parameters(data, params, num_boost_round, early_stopping_rounds):
    """Train deterministic folds and return leakage-free threshold metrics."""
    X = data["X"]
    y = data["y"]
    fold_ids = data["fold_ids"]
    oof = np.zeros(len(y), dtype=np.float32)
    iterations = []
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
            params,
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
        "cross_fitted_macro_f1": report["cross_fitted_macro_f1"],
        "fold_macro_f1_mean": report["fold_macro_f1_mean"],
        "fold_macro_f1_std": report["fold_macro_f1_std"],
        "deploy_threshold": report["deploy_threshold"],
        "all_oof_selection_macro_f1": report["all_oof_selection_macro_f1"],
        "mean_best_iteration": float(np.mean(iterations)),
    }


def _atomic_write_frame(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def main(argv=None):
    args = parse_args(argv)
    if args.sample_terms < 5:
        raise ValueError("--sample-terms must be at least 5")
    if (
        any(value <= 1 for value in args.num_leaves)
        or any(value <= 0 for value in args.learning_rates)
        or any(value <= 0 for value in args.min_child_samples)
        or args.num_boost_round <= 0
        or args.early_stopping_rounds <= 0
    ):
        raise ValueError("all tuning parameters must be positive")
    os.makedirs(args.workspace, exist_ok=True)
    training_args = SimpleNamespace(
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    data = prepare_training_data(training_args, args.workspace, args.sample_terms)
    rows = []
    grid = list(
        itertools.product(
            args.num_leaves,
            args.learning_rates,
            args.min_child_samples,
        )
    )
    for index, (num_leaves, learning_rate, min_child_samples) in enumerate(
        grid, start=1
    ):
        print(
            f"[{index}/{len(grid)}] leaves={num_leaves} lr={learning_rate} "
            f"min_child={min_child_samples}"
        )
        params = {
            **LGBM_PARAMS,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
        }
        result = evaluate_parameters(
            data, params, args.num_boost_round, args.early_stopping_rounds
        )
        rows.append(
            {
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "min_child_samples": min_child_samples,
                **result,
            }
        )
    results = pd.DataFrame(rows).sort_values(
        ["cross_fitted_macro_f1", "fold_macro_f1_std"],
        ascending=[False, True],
    )
    _atomic_write_frame(results, args.output)
    print(results.to_string(index=False))
    print(f"tuning={args.output}")
    return args.output


if __name__ == "__main__":
    main()
