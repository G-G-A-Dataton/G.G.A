"""Produce leakage-free threshold diagnostics from verified shortlist artifacts."""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.metrics import macro_f1
from src.modeling import (
    predictions_from_cross_fitted_selection,
    select_cross_fitted_candidate,
)
from src.oof_artifacts import load_oof_artifacts


DEFAULT_ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ensemble_artifacts")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "threshold_analysis.csv")
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, "docs", "threshold_analysis.md")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze thresholds from hash-verified grouped OOF artifacts"
    )
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--allow-sample",
        action="store_true",
        help="Allow smoke-test artifacts and use sample-specific default outputs",
    )
    return parser.parse_args(argv)


def analyze_thresholds(y_true, probabilities, thresholds):
    """Return descriptive metrics for explicit deploy-threshold candidates."""
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.asarray(list(thresholds), dtype=np.float64)
    if (
        y_true.ndim != 1
        or probabilities.shape != y_true.shape
        or not np.isin(y_true, [0, 1]).all()
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError("y_true and probabilities must be aligned finite binary data")
    if (
        thresholds.ndim != 1
        or len(thresholds) == 0
        or not np.isfinite(thresholds).all()
        or ((thresholds < 0) | (thresholds > 1)).any()
    ):
        raise ValueError("thresholds must be finite values in [0, 1]")

    rows = []
    for threshold in np.unique(thresholds):
        prediction = (probabilities >= threshold).astype(np.int8)
        tn, fp, fn, tp = confusion_matrix(
            y_true, prediction, labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "macro_f1": macro_f1(y_true, prediction),
                "f1_positive": f1_score(
                    y_true, prediction, pos_label=1, zero_division=0
                ),
                "f1_negative": f1_score(
                    y_true, prediction, pos_label=0, zero_division=0
                ),
                "precision": precision_score(
                    y_true, prediction, zero_division=0
                ),
                "recall": recall_score(y_true, prediction, zero_division=0),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "positive_rate": float(prediction.mean()),
            }
        )
    return pd.DataFrame(rows)


def _atomic_write_frame(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _write_report(path, selection, cross_fitted_score, diagnostics, manifest):
    deploy = selection["deploy"]
    default = diagnostics.iloc[(diagnostics["threshold"] - 0.5).abs().argmin()]
    selected = diagnostics.iloc[
        (diagnostics["threshold"] - deploy["threshold"]).abs().argmin()
    ]
    lines = [
        "# Threshold Analysis",
        "",
        "The primary score uses fold-specific thresholds selected without the evaluated fold. The deploy threshold is then fitted on all OOF rows and is not reported as an unbiased validation score.",
        "",
        "## Validation Result",
        "",
        f"- Artifact mode: `{manifest['training_mode']}` training / `{manifest['test_mode']}` test",
        f"- Selected candidate: `{deploy['selected_model']}`",
        f"- Cross-fitted Macro-F1: `{cross_fitted_score:.6f}`",
        f"- Fold count: `{manifest['validation']['n_splits']}` grouped by `term_id`",
        "",
        "## Deploy Parameters",
        "",
        f"- LightGBM weight: `{deploy['lightgbm_weight']:.4f}`",
        f"- XGBoost weight: `{deploy['xgboost_weight']:.4f}`",
        f"- Threshold: `{deploy['threshold']:.8f}`",
        f"- All-OOF diagnostic Macro-F1 at deploy threshold: `{selected['macro_f1']:.6f}`",
        f"- All-OOF diagnostic Macro-F1 at 0.5: `{default['macro_f1']:.6f}`",
        "",
        "## Diagnostic Curve",
        "",
        "| Threshold | Macro-F1 | F1 positive | F1 negative | Precision | Recall | Positive rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    closest = diagnostics.iloc[
        (diagnostics["threshold"] - deploy["threshold"])
        .abs()
        .sort_values()
        .head(11)
        .index
    ].sort_values("threshold")
    for row in closest.itertuples(index=False):
        lines.append(
            f"| {row.threshold:.8f} | {row.macro_f1:.6f} | "
            f"{row.f1_positive:.6f} | {row.f1_negative:.6f} | "
            f"{row.precision:.6f} | {row.recall:.6f} | {row.positive_rate:.4%} |"
        )
    lines.extend(
        [
            "",
            "The complete descriptive curve is stored in `outputs/threshold_analysis.csv`.",
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
    output = args.output or (
        os.path.join(PROJECT_ROOT, "outputs", "threshold_analysis_sample.csv")
        if args.allow_sample
        else DEFAULT_OUTPUT
    )
    report = args.report or (
        os.path.join(PROJECT_ROOT, "docs", "threshold_analysis_sample.md")
        if args.allow_sample
        else DEFAULT_REPORT
    )
    manifest, arrays = load_oof_artifacts(
        args.artifact_dir,
        require_full=not args.allow_sample,
        source_data_dir=DATA_DIR,
    )
    y_true = arrays["y_true.npy"]
    fold_ids = arrays["fold_ids.npy"]
    selection = select_cross_fitted_candidate(
        y_true,
        arrays["oof_lgbm.npy"],
        arrays["oof_xgb.npy"],
        fold_ids,
    )
    deploy = selection["deploy"]
    probabilities = (
        deploy["lightgbm_weight"] * arrays["oof_lgbm.npy"]
        + deploy["xgboost_weight"] * arrays["oof_xgb.npy"]
    )
    cross_fitted_predictions = predictions_from_cross_fitted_selection(
        arrays["oof_lgbm.npy"],
        arrays["oof_xgb.npy"],
        fold_ids,
        selection,
    )
    cross_fitted_score = macro_f1(y_true, cross_fitted_predictions)
    thresholds = np.unique(
        np.concatenate(
            [np.linspace(0.01, 0.99, 99), [0.5, deploy["threshold"]]]
        )
    )
    diagnostics = analyze_thresholds(y_true, probabilities, thresholds)
    _atomic_write_frame(diagnostics, output)
    _write_report(report, selection, cross_fitted_score, diagnostics, manifest)
    print(
        f"selected={deploy['selected_model']} "
        f"cross_fitted_macro_f1={cross_fitted_score:.6f} "
        f"deploy_threshold={deploy['threshold']:.8f}"
    )
    print(f"diagnostics={output}\nreport={report}")
    return output


if __name__ == "__main__":
    main()
