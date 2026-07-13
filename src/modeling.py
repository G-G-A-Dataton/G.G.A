"""Shared grouped validation, threshold selection, and ensemble evaluation."""

import numpy as np
import pandas as pd

from src.context_features import CONTEXT_FEATURE_COLS
from src.features import FEATURE_COLS
from src.metrics import find_best_threshold, get_stratified_group_kfold, macro_f1


MODEL_FEATURE_COLS = FEATURE_COLS + ["tfidf_cosine", *CONTEXT_FEATURE_COLS]


def build_group_fold_ids(y, groups, n_splits=5, random_state=42):
    """Assign every row to one deterministic, leakage-free validation fold."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if y.ndim != 1 or groups.ndim != 1 or len(y) == 0 or len(y) != len(groups):
        raise ValueError("y and groups must be non-empty aligned vectors")
    if pd.isna(groups).any() or len(pd.unique(groups)) < n_splits:
        raise ValueError("groups must contain at least n_splits non-null values")
    splitter = get_stratified_group_kfold(
        n_splits=n_splits, random_state=random_state
    )
    fold_ids = np.full(len(y), -1, dtype=np.int8)
    placeholder = np.zeros((len(y), 1), dtype=np.int8)
    for fold, (_, validation_index) in enumerate(
        splitter.split(placeholder, y, groups=groups)
    ):
        fold_ids[validation_index] = fold
    if (fold_ids < 0).any():
        raise RuntimeError("Fold assignment did not cover every row")
    return fold_ids


def cross_fitted_threshold_evaluation(y_true, probabilities, fold_ids):
    """Select a threshold without using the fold on which it is evaluated."""
    y_true, probabilities, fold_ids = _validate_oof_inputs(
        y_true, probabilities, fold_ids
    )
    cross_fitted_predictions = np.zeros(len(y_true), dtype=np.int8)
    fold_results = []
    for fold in np.unique(fold_ids):
        validation = fold_ids == fold
        selection = ~validation
        threshold, selection_score, _ = find_best_threshold(
            y_true[selection], probabilities[selection]
        )
        fold_predictions = (probabilities[validation] >= threshold).astype(np.int8)
        cross_fitted_predictions[validation] = fold_predictions
        fold_results.append(
            {
                "fold": int(fold),
                "threshold": float(threshold),
                "selection_macro_f1": float(selection_score),
                "validation_macro_f1": macro_f1(
                    y_true[validation], fold_predictions
                ),
                "validation_rows": int(validation.sum()),
            }
        )
    deploy_threshold, all_oof_selection_score, _ = find_best_threshold(
        y_true, probabilities
    )
    validation_scores = [row["validation_macro_f1"] for row in fold_results]
    return {
        "cross_fitted_macro_f1": macro_f1(y_true, cross_fitted_predictions),
        "fold_macro_f1_mean": float(np.mean(validation_scores)),
        "fold_macro_f1_std": float(np.std(validation_scores)),
        "deploy_threshold": float(deploy_threshold),
        "all_oof_selection_macro_f1": float(all_oof_selection_score),
        "folds": fold_results,
    }


def cross_fitted_ensemble_evaluation(
    y_true,
    first_probabilities,
    second_probabilities,
    fold_ids,
    weights=None,
):
    """Tune two-model blend weight and threshold outside each evaluated fold."""
    y_true, first_probabilities, fold_ids = _validate_oof_inputs(
        y_true, first_probabilities, fold_ids
    )
    second_probabilities = np.asarray(second_probabilities, dtype=np.float64)
    if (
        second_probabilities.shape != first_probabilities.shape
        or not np.isfinite(second_probabilities).all()
        or ((second_probabilities < 0) | (second_probabilities > 1)).any()
    ):
        raise ValueError("second_probabilities must be aligned finite probabilities")
    if weights is None:
        weights = np.linspace(0.0, 1.0, 21)
    weights = _validate_weights(weights)

    predictions = np.zeros(len(y_true), dtype=np.int8)
    fold_results = []
    for fold in np.unique(fold_ids):
        validation = fold_ids == fold
        selection = ~validation
        weight, threshold, selection_score = _select_blend(
            y_true[selection],
            first_probabilities[selection],
            second_probabilities[selection],
            weights,
        )
        blended = (
            weight * first_probabilities[validation]
            + (1.0 - weight) * second_probabilities[validation]
        )
        fold_predictions = (blended >= threshold).astype(np.int8)
        predictions[validation] = fold_predictions
        fold_results.append(
            {
                "fold": int(fold),
                "first_model_weight": float(weight),
                "threshold": float(threshold),
                "selection_macro_f1": float(selection_score),
                "validation_macro_f1": macro_f1(
                    y_true[validation], fold_predictions
                ),
            }
        )

    final_weight, final_threshold, all_oof_selection_score = _select_blend(
        y_true, first_probabilities, second_probabilities, weights
    )
    return {
        "cross_fitted_macro_f1": macro_f1(y_true, predictions),
        "deploy_first_model_weight": float(final_weight),
        "deploy_second_model_weight": float(1.0 - final_weight),
        "deploy_threshold": float(final_threshold),
        "all_oof_selection_macro_f1": float(all_oof_selection_score),
        "folds": fold_results,
    }


def _select_blend(y_true, first, second, weights):
    candidates = []
    for weight in weights:
        probabilities = weight * first + (1.0 - weight) * second
        threshold, score, _ = find_best_threshold(y_true, probabilities)
        candidates.append((float(score), float(weight), float(threshold)))
    score, weight, threshold = min(
        candidates,
        key=lambda row: (-row[0], abs(row[1] - 0.5), abs(row[2] - 0.5)),
    )
    return weight, threshold, score


def _validate_oof_inputs(y_true, probabilities, fold_ids):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    fold_ids = np.asarray(fold_ids)
    if (
        y_true.ndim != 1
        or probabilities.ndim != 1
        or fold_ids.ndim != 1
        or len(y_true) == 0
        or len(y_true) != len(probabilities)
        or len(y_true) != len(fold_ids)
    ):
        raise ValueError("OOF inputs must be non-empty aligned vectors")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must be binary")
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("probabilities must be finite values in [0, 1]")
    if len(np.unique(fold_ids)) < 2:
        raise ValueError("fold_ids must contain at least two folds")
    return y_true.astype(np.int8, copy=False), probabilities, fold_ids


def _validate_weights(weights):
    values = np.asarray(list(weights), dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.isfinite(values).all()
        or ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("weights must be non-empty finite values in [0, 1]")
    return np.unique(values)
