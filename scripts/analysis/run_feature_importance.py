"""Aggregate feature importance from hash-verified production LightGBM folds."""

import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.modeling import MODEL_FEATURE_COLS
from src.oof_artifacts import load_oof_artifacts


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "outputs", "ensemble_artifacts")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "feature_importance.csv")
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, "docs", "feature_importance.md")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate verified LightGBM fold feature importances"
    )
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--allow-sample", action="store_true")
    return parser.parse_args(argv)


def aggregate_importance(models, feature_columns):
    """Return fold mean/std gain and split importance with normalized gain."""
    if not models or not feature_columns:
        raise ValueError("models and feature_columns must be non-empty")
    gains = []
    splits = []
    for model in models:
        if model.feature_name() != list(feature_columns):
            raise ValueError("LightGBM model feature names do not match the manifest")
        gains.append(model.feature_importance(importance_type="gain"))
        splits.append(model.feature_importance(importance_type="split"))
    gain_matrix = np.asarray(gains, dtype=np.float64)
    split_matrix = np.asarray(splits, dtype=np.float64)
    if gain_matrix.shape != (len(models), len(feature_columns)):
        raise ValueError("LightGBM importance dimensions do not match features")
    mean_gain = gain_matrix.mean(axis=0)
    total_gain = mean_gain.sum()
    gain_ratio = mean_gain / total_gain if total_gain > 0 else np.zeros_like(mean_gain)
    frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain_mean": mean_gain,
            "gain_std": gain_matrix.std(axis=0),
            "gain_ratio": gain_ratio,
            "split_mean": split_matrix.mean(axis=0),
            "split_std": split_matrix.std(axis=0),
            "nonzero_folds": (gain_matrix > 0).sum(axis=0),
        }
    )
    return frame.sort_values(
        ["gain_mean", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


def _atomic_write_frame(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _write_report(frame, manifest, path):
    lines = [
        "# Feature Importance",
        "",
        "Importances are aggregated across the five hash-verified LightGBM folds. Gain is descriptive and does not establish causal value; ablation is required before removing a feature.",
        "",
        f"- Artifact mode: `{manifest['training_mode']}` training / `{manifest['test_mode']}` test",
        f"- Feature schema: `{manifest['feature_schema_version']}`",
        f"- Candidate sampling schema: `{manifest['candidate_sampling_schema_version']}`",
        "",
        "| Rank | Feature | Gain share | Gain mean | Gain std | Split mean | Nonzero folds |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(frame.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | `{row.feature}` | {row.gain_ratio:.2%} | "
            f"{row.gain_mean:.3f} | {row.gain_std:.3f} | "
            f"{row.split_mean:.1f} | {row.nonzero_folds} |"
        )
    lines.extend(
        [
            "",
            "Features with zero or unstable gain remain in the contract until a grouped, cross-fitted ablation demonstrates a non-negative removal decision.",
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
        os.path.join(PROJECT_ROOT, "outputs", "feature_importance_sample.csv")
        if args.allow_sample
        else DEFAULT_OUTPUT
    )
    report = args.report or (
        os.path.join(PROJECT_ROOT, "docs", "feature_importance_sample.md")
        if args.allow_sample
        else DEFAULT_REPORT
    )
    manifest, _ = load_oof_artifacts(
        args.artifact_dir,
        require_full=not args.allow_sample,
        source_data_dir=DATA_DIR,
    )
    model_files = sorted(
        filename
        for filename in manifest["model_files"]
        if filename.startswith("lgbm_fold_") and filename.endswith(".txt")
    )
    if len(model_files) != manifest["validation"]["n_splits"]:
        raise ValueError("manifest must contain one LightGBM model per fold")
    models = [
        lgb.Booster(model_file=os.path.join(args.artifact_dir, filename))
        for filename in model_files
    ]
    importance = aggregate_importance(models, manifest["feature_columns"])
    if importance["feature"].tolist() == [] or set(importance["feature"]) != set(
        MODEL_FEATURE_COLS
    ):
        raise ValueError("feature importance output violates the model contract")
    _atomic_write_frame(importance, output)
    _write_report(importance, manifest, report)
    print(importance.to_string(index=False))
    print(f"importance={output}\nreport={report}")
    return output


if __name__ == "__main__":
    main()
