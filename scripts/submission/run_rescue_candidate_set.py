"""Publish QA-verified structural-semantic leaderboard rescue candidates."""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.oof_artifacts import EXPECTED_TEST_ROWS, sha256_file
from src.rescue import (
    band_override_predictions,
    prediction_summary,
    threshold_predictions,
)
from src.validate_submission import validate_submission


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
STATIC_DIR = os.path.join(OUTPUT_DIR, "static_semantic")
E5_DIR = os.path.join(OUTPUT_DIR, "e5_rescue")
DEFAULT_DESTINATION = os.path.join(OUTPUT_DIR, "rescue_candidates")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Publish the two independently validated leaderboard rescue files"
    )
    parser.add_argument(
        "--meta-probabilities", default=os.path.join(STATIC_DIR, "meta_test.npy")
    )
    parser.add_argument(
        "--meta-report", default=os.path.join(STATIC_DIR, "meta_report.json")
    )
    parser.add_argument(
        "--e5-scores", default=os.path.join(E5_DIR, "test_e5_score.npy")
    )
    parser.add_argument(
        "--e5-band-probabilities", default=os.path.join(E5_DIR, "band_test.npy")
    )
    parser.add_argument(
        "--e5-band-report", default=os.path.join(E5_DIR, "band_report.json")
    )
    parser.add_argument("--output-dir", default=DEFAULT_DESTINATION)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    return parser.parse_args(argv)


def _load_json(path):
    with open(path, encoding="utf-8") as input_file:
        return json.load(input_file)


def _git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _atomic_json(path, payload):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, path)


def _write_submission(path, predictions, chunk_size):
    temporary_path = path + ".tmp"
    if os.path.exists(temporary_path):
        os.remove(temporary_path)
    offset = 0
    reader = pd.read_csv(
        os.path.join(DATA_DIR, "submission_pairs.csv"),
        usecols=["id"],
        dtype={"id": "string"},
        chunksize=chunk_size,
    )
    try:
        for id_chunk in reader:
            end = offset + len(id_chunk)
            pd.DataFrame(
                {
                    "id": id_chunk["id"].to_numpy(),
                    "prediction": predictions[offset:end],
                }
            ).to_csv(
                temporary_path,
                mode="w" if offset == 0 else "a",
                header=offset == 0,
                index=False,
            )
            offset = end
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    finally:
        reader.close()
    if offset != len(predictions):
        raise RuntimeError(
            f"Submission pair count {offset:,} does not match "
            f"predictions {len(predictions):,}"
        )
    if not validate_submission(
        temporary_path,
        sample_submission_path=os.path.join(DATA_DIR, "sample_submission.csv"),
        expected_rows=EXPECTED_TEST_ROWS,
        verbose=False,
        chunk_size=chunk_size,
    ):
        raise RuntimeError(f"Rescue candidate failed submission QA: {path}")
    os.replace(temporary_path, path)


def main(argv=None):
    args = parse_args(argv)
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    required = [
        args.meta_probabilities,
        args.meta_report,
        args.e5_scores,
        args.e5_band_probabilities,
        args.e5_band_report,
    ]
    missing = [path for path in required if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing rescue evidence artifacts: {missing}")

    meta_probabilities = np.load(args.meta_probabilities, mmap_mode="r")
    meta_report = _load_json(args.meta_report)["report"]
    if len(meta_probabilities) != EXPECTED_TEST_ROWS:
        raise ValueError("Meta probabilities do not cover the complete submission")
    primary = threshold_predictions(
        meta_probabilities, meta_report["deploy_threshold"]
    )

    e5_scores = np.load(args.e5_scores, mmap_mode="r")
    band_mask = np.isfinite(e5_scores)
    band_probabilities = np.load(args.e5_band_probabilities, mmap_mode="r")
    band_payload = _load_json(args.e5_band_report)
    band_report = band_payload["band_report"]
    secondary = band_override_predictions(
        meta_probabilities,
        meta_report["deploy_threshold"],
        band_probabilities,
        band_mask,
        band_report["deploy_threshold"],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    candidates = [
        {
            "rank": 1,
            "name": "structural_semantic_stack",
            "filename": "submission_rescue_1_structural_semantic_stack.csv",
            "predictions": primary,
            "cross_fitted_macro_f1": meta_report["cross_fitted_macro_f1"],
            "deploy_threshold": meta_report["deploy_threshold"],
            "risk": "preferred",
        },
        {
            "rank": 2,
            "name": "e5_gated_stack",
            "filename": "submission_rescue_2_e5_gated_stack.csv",
            "predictions": secondary,
            "cross_fitted_macro_f1": band_payload["combined_macro_f1"],
            "deploy_threshold": band_report["deploy_threshold"],
            "risk": "feedback_only",
        },
    ]
    records = []
    for candidate in candidates:
        path = os.path.join(args.output_dir, candidate["filename"])
        _write_submission(path, candidate["predictions"], args.chunk_size)
        records.append(
            {
                key: value
                for key, value in candidate.items()
                if key != "predictions"
            }
            | prediction_summary(candidate["predictions"])
            | {
                "path": os.path.relpath(path, PROJECT_ROOT),
                "sha256": sha256_file(path),
            }
        )

    disagreement = int(np.count_nonzero(primary != secondary))
    manifest = {
        "schema_version": 1,
        "status": "QA_VERIFIED_AWAITING_LEADERBOARD",
        "source_revision": _git_revision(),
        "submission_rows": EXPECTED_TEST_ROWS,
        "selection_rule": (
            "Upload rank 1 first; evaluate rank 2 only after recording rank 1 "
            "Public Score and only when a materially different probe is justified."
        ),
        "public_scores": None,
        "candidates": records,
        "pairwise_disagreement": {
            "rows": disagreement,
            "rate": disagreement / EXPECTED_TEST_ROWS,
        },
        "evidence_artifacts": {
            os.path.relpath(path, PROJECT_ROOT): sha256_file(path)
            for path in required
        },
    }
    manifest_path = os.path.join(args.output_dir, "rescue_candidate_set.json")
    _atomic_json(manifest_path, manifest)
    print(
        f"manifest={manifest_path}\n"
        f"primary={records[0]['path']} positives={records[0]['positive_rows']:,}\n"
        f"secondary={records[1]['path']} positives={records[1]['positive_rows']:,}\n"
        f"disagreement={disagreement:,}"
    )
    return manifest_path


if __name__ == "__main__":
    main()
