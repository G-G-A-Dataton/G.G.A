"""Compare production shortlist candidates from verified grouped OOF artifacts."""

import argparse
import os
import sys

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.modeling import select_cross_fitted_candidate
from src.oof_artifacts import load_oof_artifacts


DEFAULT_ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ensemble_artifacts")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "ensemble_comparison.csv")
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, "docs", "ensemble_comparison.md")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare hash-verified LightGBM, XGBoost, and blend OOF results"
    )
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--allow-sample", action="store_true")
    return parser.parse_args(argv)


def build_comparison(selection):
    """Convert the common shortlist selection contract into a flat table."""
    labels = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "weighted_blend": "Weighted blend",
    }
    rows = []
    selected = selection["deploy"]["selected_model"]
    for name in ("lightgbm", "xgboost", "weighted_blend"):
        candidate = selection["candidates"][name]
        report = selection["validation"][name]
        rows.append(
            {
                "candidate": name,
                "display_name": labels[name],
                "selected": name == selected,
                "cross_fitted_macro_f1": candidate["cross_fitted_macro_f1"],
                "fold_macro_f1_mean": (
                    report.get("fold_macro_f1_mean")
                    if name != "weighted_blend"
                    else sum(
                        fold["validation_macro_f1"] for fold in report["folds"]
                    )
                    / len(report["folds"])
                ),
                "all_oof_selection_macro_f1": report[
                    "all_oof_selection_macro_f1"
                ],
                "lightgbm_weight": candidate["lightgbm_weight"],
                "xgboost_weight": candidate["xgboost_weight"],
                "deploy_threshold": candidate["threshold"],
            }
        )
    return pd.DataFrame(rows)


def _atomic_write_frame(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _write_report(frame, manifest, path):
    lines = [
        "# Ensemble Candidate Comparison",
        "",
        "Candidate selection uses cross-fitted Macro-F1. All-OOF selection scores and deploy thresholds are included only for reproducibility.",
        "",
        f"Artifact mode: `{manifest['training_mode']}` training / `{manifest['test_mode']}` test.",
        "",
        "| Candidate | Selected | Cross-fitted Macro-F1 | All-OOF selection score | LGB weight | XGB weight | Deploy threshold |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {'yes' if row.selected else 'no'} | "
            f"{row.cross_fitted_macro_f1:.6f} | "
            f"{row.all_oof_selection_macro_f1:.6f} | "
            f"{row.lightgbm_weight:.4f} | {row.xgboost_weight:.4f} | "
            f"{row.deploy_threshold:.8f} |"
        )
    lines.extend(
        [
            "",
            "The selected row is the only candidate eligible for production thresholding.",
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
        os.path.join(PROJECT_ROOT, "outputs", "ensemble_comparison_sample.csv")
        if args.allow_sample
        else DEFAULT_OUTPUT
    )
    report = args.report or (
        os.path.join(PROJECT_ROOT, "docs", "ensemble_comparison_sample.md")
        if args.allow_sample
        else DEFAULT_REPORT
    )
    manifest, arrays = load_oof_artifacts(
        args.artifact_dir,
        require_full=not args.allow_sample,
        source_data_dir=DATA_DIR,
    )
    selection = select_cross_fitted_candidate(
        arrays["y_true.npy"],
        arrays["oof_lgbm.npy"],
        arrays["oof_xgb.npy"],
        arrays["fold_ids.npy"],
    )
    comparison = build_comparison(selection)
    _atomic_write_frame(comparison, output)
    _write_report(comparison, manifest, report)
    selected = comparison.loc[comparison["selected"]].iloc[0]
    print(
        f"selected={selected['candidate']} "
        f"cross_fitted_macro_f1={selected['cross_fitted_macro_f1']:.6f}"
    )
    print(f"comparison={output}\nreport={report}")
    return output


if __name__ == "__main__":
    main()
