"""Cross-fitted ensemble selection and atomic candidate submission creation."""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.modeling import (
    select_cross_fitted_candidate,
)
from src.oof_artifacts import load_oof_artifacts
from src.validate_submission import validate_submission


OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
DEFAULT_ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "ensemble_artifacts")
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, "submission_v2.csv")
DEFAULT_REPORT = os.path.join(PROJECT_ROOT, "docs", "ensemble_selection.md")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Cross-fitted ensemble optimization")
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument(
        "--allow-sample",
        action="store_true",
        help="Allow sample artifacts for pipeline smoke tests",
    )
    return parser.parse_args(argv)


def _atomic_write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, path)


def _write_report(
    path, lgb_report, xgb_report, blend_report, selected_model, deploy, positive_rate
):
    lines = [
        "# Ensemble Model Selection Report",
        "",
        "This report separates held-out model-selection performance from deploy parameters.",
        "",
        "| Candidate | Cross-fitted Macro-F1 |",
        "|---|---:|",
        f"| LightGBM | {lgb_report['cross_fitted_macro_f1']:.6f} |",
        f"| XGBoost | {xgb_report['cross_fitted_macro_f1']:.6f} |",
        f"| Weighted blend | {blend_report['cross_fitted_macro_f1']:.6f} |",
        "",
        "## Deploy Parameters",
        "",
        f"- Selected candidate: `{selected_model}`",
        f"- LightGBM weight: `{deploy['lightgbm_weight']:.4f}`",
        f"- XGBoost weight: `{deploy['xgboost_weight']:.4f}`",
        f"- Threshold: `{deploy['threshold']:.8f}`",
        f"- Candidate positive rate: `{positive_rate:.4%}`",
        "",
        "The all-OOF selection score is recorded for reproducibility only; it is not an unbiased validation estimate.",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
    os.replace(temporary_path, path)


def main(argv=None):
    args = parse_args(argv)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    manifest, arrays = load_oof_artifacts(
        args.artifact_dir,
        require_full=not args.allow_sample,
        source_data_dir=DATA_DIR,
    )
    training_rows = manifest["training"]["rows"]
    test_rows = manifest["test_rows"]

    y_true = arrays["y_true.npy"]
    fold_ids = arrays["fold_ids.npy"]
    selection = select_cross_fitted_candidate(
        y_true,
        arrays["oof_lgbm.npy"],
        arrays["oof_xgb.npy"],
        fold_ids,
    )
    lgb_report = selection["validation"]["lightgbm"]
    xgb_report = selection["validation"]["xgboost"]
    blend_report = selection["validation"]["weighted_blend"]
    selected_model = selection["deploy"]["selected_model"]
    weight = selection["deploy"]["lightgbm_weight"]
    threshold = selection["deploy"]["threshold"]

    if args.allow_sample and os.path.realpath(args.output) == os.path.realpath(
        DEFAULT_OUTPUT
    ):
        raise ValueError("Sample ensemble runs cannot overwrite the production candidate")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary_output = args.output + ".tmp"
    pair_reader = pd.read_csv(
        os.path.join(DATA_DIR, "submission_pairs.csv"),
        usecols=["id"],
        dtype={"id": "string"},
        nrows=test_rows,
        chunksize=args.chunk_size,
    )
    offset = 0
    positive_count = 0
    try:
        for chunk in pair_reader:
            end = offset + len(chunk)
            probabilities = (
                weight * arrays["test_lgbm.npy"][offset:end]
                + (1.0 - weight) * arrays["test_xgb.npy"][offset:end]
            )
            predictions = (probabilities >= threshold).astype(np.int8)
            positive_count += int(predictions.sum())
            pd.DataFrame(
                {"id": chunk["id"].to_numpy(), "prediction": predictions}
            ).to_csv(
                temporary_output,
                mode="w" if offset == 0 else "a",
                header=offset == 0,
                index=False,
            )
            offset = end
    except Exception:
        if os.path.exists(temporary_output):
            os.remove(temporary_output)
        raise
    finally:
        pair_reader.close()
    if offset != test_rows:
        raise RuntimeError(f"Submission rows mismatch: {offset:,} != {test_rows:,}")
    if not validate_submission(
        temporary_output,
        sample_submission_path=os.path.join(DATA_DIR, "sample_submission.csv"),
        expected_rows=test_rows,
        verbose=True,
    ):
        os.remove(temporary_output)
        raise RuntimeError("Ensemble candidate failed submission validation")
    os.replace(temporary_output, args.output)

    positive_rate = positive_count / test_rows
    decision = {
        "validation": {
            "lightgbm": lgb_report,
            "xgboost": xgb_report,
            "ensemble": blend_report,
        },
        "deploy": {
            "selected_model": selected_model,
            "lightgbm_weight": weight,
            "xgboost_weight": 1.0 - weight,
            "threshold": threshold,
            "positive_rate": positive_rate,
            "rows": test_rows,
        },
    }
    _atomic_write_json(
        os.path.join(args.artifact_dir, "ensemble_decision.json"), decision
    )
    _write_report(
        args.report,
        lgb_report,
        xgb_report,
        blend_report,
        selected_model,
        decision["deploy"],
        positive_rate,
    )
    print(
        f"Selected={selected_model} cross-fitted Macro-F1="
        f"{selection['deploy']['cross_fitted_macro_f1']:.6f} "
        f"weight={weight:.3f} threshold={threshold:.8f} output={args.output}"
    )
    return args.output


if __name__ == "__main__":
    main()
