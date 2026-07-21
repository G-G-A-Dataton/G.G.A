"""Build, validate, and manifest the two strongest submission candidates."""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from scripts.training.run_train_full_v2 import git_revision
from src.delivery_artifacts import validate_delivery_manifest
from src.modeling import select_cross_fitted_candidate
from src.oof_artifacts import load_oof_artifacts, sha256_file
from src.validate_submission import validate_submission


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
DEFAULT_ARTIFACT_DIR = os.path.join(OUTPUT_DIR, "ensemble_artifacts")
DEFAULT_CANDIDATE_DIR = os.path.join(OUTPUT_DIR, "final_candidates")
DEFAULT_MANIFEST = os.path.join(DEFAULT_CANDIDATE_DIR, "candidate_set.json")
DEFAULT_ACCEPTED_MANIFEST = os.path.join(OUTPUT_DIR, "submission_v2.manifest.json")
DEFAULT_STRATEGY_REPORT = os.path.join(
    PROJECT_ROOT, "docs", "final_submission_candidates.md"
)
DEFAULT_QA_REPORT = os.path.join(
    PROJECT_ROOT, "docs", "submission_qa_approval.md"
)
DISPLAY_NAMES = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "weighted_blend": "Weighted LightGBM/XGBoost blend",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Publish the top two OOF-ranked final submission candidates"
    )
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--accepted-manifest", default=DEFAULT_ACCEPTED_MANIFEST)
    parser.add_argument("--strategy-report", default=DEFAULT_STRATEGY_REPORT)
    parser.add_argument("--qa-report", default=DEFAULT_QA_REPORT)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    return parser.parse_args(argv)


def rank_candidates(selection, limit=2):
    """Rank deploy candidates by leakage-free cross-fitted Macro-F1."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    candidates = selection.get("candidates", {})
    selected_model = selection.get("deploy", {}).get("selected_model")
    if set(candidates) != set(DISPLAY_NAMES):
        raise ValueError("selection must contain the complete model shortlist")
    ranked = sorted(
        candidates.items(),
        key=lambda item: (
            -float(item[1]["cross_fitted_macro_f1"]),
            item[0] != selected_model,
            item[0],
        ),
    )
    return [
        {"candidate": name, "display_name": DISPLAY_NAMES[name], **parameters}
        for name, parameters in ranked[:limit]
    ]


def candidate_predictions(lightgbm, xgboost, candidate):
    """Apply one candidate's deploy weights and threshold to aligned arrays."""
    lightgbm = np.asarray(lightgbm, dtype=np.float64)
    xgboost = np.asarray(xgboost, dtype=np.float64)
    if lightgbm.ndim != 1 or xgboost.shape != lightgbm.shape:
        raise ValueError("candidate probability arrays must be aligned vectors")
    if (
        not np.isfinite(lightgbm).all()
        or not np.isfinite(xgboost).all()
        or ((lightgbm < 0.0) | (lightgbm > 1.0)).any()
        or ((xgboost < 0.0) | (xgboost > 1.0)).any()
    ):
        raise ValueError("candidate probabilities must be finite values in [0, 1]")
    lightgbm_weight = float(candidate["lightgbm_weight"])
    xgboost_weight = float(candidate["xgboost_weight"])
    threshold = float(candidate["threshold"])
    if (
        not 0.0 <= lightgbm_weight <= 1.0
        or not 0.0 <= xgboost_weight <= 1.0
        or not np.isclose(lightgbm_weight + xgboost_weight, 1.0)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("candidate deploy parameters are invalid")
    probabilities = lightgbm_weight * lightgbm + xgboost_weight * xgboost
    return (probabilities >= threshold).astype(np.int8)


def _assert_primary_matches_accepted(primary, accepted):
    deploy = accepted["deploy"]
    checks = (
        primary["candidate"] == deploy["selected_model"],
        np.isclose(primary["lightgbm_weight"], deploy["lightgbm_weight"]),
        np.isclose(primary["xgboost_weight"], deploy["xgboost_weight"]),
        np.isclose(primary["threshold"], deploy["threshold"]),
    )
    if not all(checks):
        raise RuntimeError("Current OOF selection does not match accepted delivery")


def _atomic_write(path, content):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        output_file.write(content)
    os.replace(temporary_path, path)


def _write_manifest(path, payload):
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _write_strategy_report(path, payload):
    rows = payload["candidates"]
    lines = [
        "# Final Submission Candidate Strategy",
        "",
        f"- Status: **{payload['status']}**",
        f"- Source revision: `{payload['source_revision']}`",
        f"- Artifact revision: `{payload['artifact_revision']}`",
        "- Ranking signal: cross-fitted grouped Macro-F1 only",
        "- Leaderboard signal used: no",
        "",
        "## Ranked Candidates",
        "",
        "| Rank | Candidate | Cross-fitted Macro-F1 | Weights (LGB/XGB) | Threshold | Positives | Positive rate |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['display_name']} | "
            f"{row['cross_fitted_macro_f1']:.6f} | "
            f"{row['lightgbm_weight']:.2f}/{row['xgboost_weight']:.2f} | "
            f"{row['threshold']:.8f} | {row['positive_rows']:,} | "
            f"{row['positive_rate']:.4%} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Candidate 1 remains the default upload because it has the highest",
            "leakage-free cross-fitted score and is byte-identical to the accepted",
            "15 July delivery. Candidate 2 is the strongest single-family fallback;",
            "it provides model diversity without using leaderboard feedback.",
            "",
            f"The candidates disagree on `{payload['pairwise_disagreement']['rows']:,}` "
            f"rows (`{payload['pairwise_disagreement']['rate']:.4%}`). This is a",
            "controlled alternative, not evidence that the runner-up is expected to",
            "score higher. Kaggle upload and observed scores remain account-bound.",
            "",
            "## Integrity",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- Candidate {row['rank']} SHA-256: `{row['sha256']}`")
    lines.append("")
    _atomic_write(path, "\n".join(lines))


def _write_qa_report(path, payload):
    lines = [
        "# Final Submission QA Approval",
        "",
        f"- Status: **{payload['status']}**",
        f"- Approved candidates: `{len(payload['candidates'])}/{len(payload['candidates'])}`",
        f"- Reference rows: `{payload['submission_rows']:,}`",
        f"- Sample submission SHA-256: `{payload['sample_submission_sha256']}`",
        "",
        "## Mandatory Checks",
        "",
        "Both candidates passed bounded-memory validation for:",
        "",
        "- exact `id,prediction` column order;",
        "- exactly 3,359,679 data rows;",
        "- integer predictions restricted to `{0, 1}` with no nulls;",
        "- exact sample-submission ID order and global ID uniqueness;",
        "- no accidental index column;",
        "- recorded row counts, class balance, and SHA-256.",
        "",
        "## Approved Files",
        "",
        "| Rank | File | Rows | Positives | SHA-256 |",
        "|---:|---|---:|---:|---|",
    ]
    for row in payload["candidates"]:
        lines.append(
            f"| {row['rank']} | `{row['path']}` | {row['rows']:,} | "
            f"{row['positive_rows']:,} | `{row['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "Candidate 1 additionally matches the accepted delivery SHA-256 byte for",
            "byte. Any later modification invalidates this approval and requires the",
            "candidate-set command to be rerun before upload.",
            "",
        ]
    )
    _atomic_write(path, "\n".join(lines))


def main(argv=None):
    args = parse_args(argv)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    revision = git_revision()
    accepted = validate_delivery_manifest(
        args.accepted_manifest,
        project_root=PROJECT_ROOT,
        source_data_dir=DATA_DIR,
        require_full=True,
    )
    artifact_manifest, arrays = load_oof_artifacts(
        args.artifact_dir,
        require_full=True,
        source_data_dir=DATA_DIR,
    )
    selection = select_cross_fitted_candidate(
        arrays["y_true.npy"],
        arrays["oof_lgbm.npy"],
        arrays["oof_xgb.npy"],
        arrays["fold_ids.npy"],
    )
    candidates = rank_candidates(selection)
    _assert_primary_matches_accepted(candidates[0], accepted)

    os.makedirs(args.output_dir, exist_ok=True)
    targets = [
        os.path.join(
            args.output_dir,
            f"submission_candidate_{rank}_{candidate['candidate']}.csv",
        )
        for rank, candidate in enumerate(candidates, start=1)
    ]
    temporary_paths = [path + ".tmp" for path in targets]
    for path in temporary_paths:
        if os.path.exists(path):
            os.remove(path)

    test_rows = artifact_manifest["test_rows"]
    positive_counts = [0] * len(candidates)
    disagreement_rows = 0
    offset = 0
    pair_reader = pd.read_csv(
        os.path.join(DATA_DIR, "submission_pairs.csv"),
        usecols=["id"],
        dtype={"id": "string"},
        nrows=test_rows,
        chunksize=args.chunk_size,
    )
    try:
        for id_chunk in pair_reader:
            end = offset + len(id_chunk)
            predictions = [
                candidate_predictions(
                    arrays["test_lgbm.npy"][offset:end],
                    arrays["test_xgb.npy"][offset:end],
                    candidate,
                )
                for candidate in candidates
            ]
            disagreement_rows += int(np.count_nonzero(predictions[0] != predictions[1]))
            for index, prediction in enumerate(predictions):
                positive_counts[index] += int(prediction.sum())
                pd.DataFrame(
                    {"id": id_chunk["id"].to_numpy(), "prediction": prediction}
                ).to_csv(
                    temporary_paths[index],
                    mode="w" if offset == 0 else "a",
                    header=offset == 0,
                    index=False,
                )
            offset = end
    except Exception:
        for path in temporary_paths:
            if os.path.exists(path):
                os.remove(path)
        raise
    finally:
        pair_reader.close()

    if offset != test_rows:
        raise RuntimeError(f"Submission rows mismatch: {offset:,} != {test_rows:,}")
    for path in temporary_paths:
        if not validate_submission(
            path,
            sample_submission_path=os.path.join(DATA_DIR, "sample_submission.csv"),
            expected_rows=test_rows,
            verbose=False,
            chunk_size=args.chunk_size,
        ):
            raise RuntimeError(f"Final candidate failed QA: {path}")

    primary_hash = sha256_file(temporary_paths[0])
    accepted_hash = accepted["files"]["submission"]["sha256"]
    if primary_hash != accepted_hash:
        raise RuntimeError("Primary candidate is not byte-identical to accepted delivery")
    for temporary_path, target in zip(temporary_paths, targets):
        os.replace(temporary_path, target)

    records = []
    for rank, (candidate, target, positives) in enumerate(
        zip(candidates, targets, positive_counts), start=1
    ):
        records.append(
            {
                "rank": rank,
                **candidate,
                "path": os.path.relpath(target, PROJECT_ROOT),
                "sha256": sha256_file(target),
                "rows": test_rows,
                "positive_rows": positives,
                "positive_rate": positives / test_rows,
                "qa_status": "PASS",
            }
        )
    payload = {
        "candidate_set_schema_version": 1,
        "status": "PASS",
        "source_revision": revision,
        "artifact_revision": artifact_manifest["code_revision"],
        "selection_contract": "cross_fitted_grouped_macro_f1",
        "leaderboard_signal_used": False,
        "submission_rows": test_rows,
        "sample_submission_sha256": sha256_file(
            os.path.join(DATA_DIR, "sample_submission.csv")
        ),
        "accepted_primary_sha256": accepted_hash,
        "candidates": records,
        "pairwise_disagreement": {
            "rows": disagreement_rows,
            "rate": disagreement_rows / test_rows,
        },
        "mandatory_qa": [
            "columns",
            "row_count",
            "binary_integer_predictions",
            "id_order",
            "id_uniqueness",
            "no_index_column",
            "sha256",
        ],
    }
    _write_manifest(args.manifest, payload)
    _write_strategy_report(args.strategy_report, payload)
    _write_qa_report(args.qa_report, payload)
    print(
        f"status=PASS candidates={len(records)} rows={test_rows} "
        f"disagreement_rows={disagreement_rows} manifest={args.manifest}"
    )
    return payload


if __name__ == "__main__":
    main()
