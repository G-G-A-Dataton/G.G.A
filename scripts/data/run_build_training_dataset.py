"""Build a versioned, test-shaped training candidate data set."""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.candidate_sampling import (
    CANDIDATE_SAMPLING_SCHEMA_VERSION,
    build_test_shaped_training_set,
    candidate_distribution,
    sample_complete_terms,
)
from src.data import load_items, load_terms
from src.data_freeze import sha256_file, verify_data_freeze
from src.oof_artifacts import (
    EXPECTED_POSITIVE_ROWS,
    EXPECTED_TRAINING_ROWS,
    EXPECTED_TRAINING_TERMS,
    PRODUCTION_CANDIDATE_SAMPLING,
)


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
FREEZE_CONFIG = os.path.join(PROJECT_ROOT, "configs", "final_v1.json")
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "outputs", "train_final_candidates.csv")
SOURCE_FILES = ("terms.csv", "items.csv", "training_pairs.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build leakage-free candidates matching submission query density"
    )
    parser.add_argument("--sample-terms", type=int, default=None)
    parser.add_argument("--min-candidates", type=int, default=100)
    parser.add_argument("--dense-multiplier", type=float, default=2.0)
    parser.add_argument("--bm25-hard-fraction", type=float, default=0.20)
    parser.add_argument("--category-hard-fraction", type=float, default=0.50)
    parser.add_argument("--bm25-top-n", type=int, default=200)
    parser.add_argument("--bm25-max-df-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _git_revision(require_clean=True):
    if require_clean:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            raise RuntimeError("Refusing to version a data set from a dirty worktree")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return revision.stdout.strip()


def _atomic_write_frame(frame, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    frame.to_csv(temporary_path, index=False)
    os.replace(temporary_path, path)


def _atomic_write_json(payload, path):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, path)


def _validate_args(args):
    if args.sample_terms is not None and args.sample_terms <= 0:
        raise ValueError("--sample-terms must be positive")
    if args.min_candidates <= 0 or args.dense_multiplier < 1.0:
        raise ValueError("candidate density parameters are invalid")
    if args.bm25_top_n <= 0 or not 0.0 < args.bm25_max_df_ratio <= 1.0:
        raise ValueError("BM25 parameters are invalid")
    fractions = (args.bm25_hard_fraction, args.category_hard_fraction)
    if any(value < 0.0 or value > 1.0 for value in fractions) or sum(fractions) > 1.0:
        raise ValueError("hard-negative fractions must be in [0, 1] and sum to <= 1")
    output_path = os.path.realpath(os.path.abspath(args.output))
    if (
        os.path.commonpath([PROJECT_ROOT, output_path]) != PROJECT_ROOT
        or not output_path.lower().endswith(".csv")
        or os.path.isdir(output_path)
    ):
        raise ValueError("--output must be a CSV path inside the project")


def _candidate_config(args):
    return {
        "min_candidates": args.min_candidates,
        "dense_multiplier": args.dense_multiplier,
        "bm25_hard_fraction": args.bm25_hard_fraction,
        "category_hard_fraction": args.category_hard_fraction,
        "bm25_top_n": args.bm25_top_n,
        "bm25_max_df_ratio": args.bm25_max_df_ratio,
        "random_state": args.seed,
    }


def main(argv=None):
    args = parse_args(argv)
    _validate_args(args)
    config = _candidate_config(args)
    if args.sample_terms is None:
        expected_config = {
            key: value
            for key, value in PRODUCTION_CANDIDATE_SAMPLING.items()
            if key not in {"strategy", "positive_reference_rows"}
        }
        if config != expected_config:
            raise ValueError(
                "Full data generation requires the locked production sampling contract"
            )
    output_path = os.path.abspath(args.output)
    manifest_path = output_path + ".manifest.json"
    if not args.force and (os.path.exists(output_path) or os.path.exists(manifest_path)):
        raise FileExistsError("Output already exists; pass --force to replace it atomically")

    revision = _git_revision()
    verify_data_freeze(FREEZE_CONFIG, DATA_DIR)
    terms = load_terms(os.path.join(DATA_DIR, "terms.csv"))
    items = load_items(os.path.join(DATA_DIR, "items.csv"))
    positives = pd.read_csv(
        os.path.join(DATA_DIR, "training_pairs.csv"),
        dtype={"id": "string", "term_id": "string", "item_id": "string", "label": "int8"},
    )
    if len(positives) != EXPECTED_POSITIVE_ROWS or not (positives["label"] == 1).all():
        raise ValueError("training_pairs.csv does not satisfy the frozen positive contract")
    selected = (
        sample_complete_terms(positives, args.sample_terms, args.seed)
        if args.sample_terms is not None
        else positives
    )
    candidates = build_test_shaped_training_set(
        selected,
        items,
        terms_df=terms,
        positive_reference_df=positives,
        **config,
    )
    stats = candidate_distribution(candidates)
    if args.sample_terms is None and (
        stats["rows"] != EXPECTED_TRAINING_ROWS
        or stats["terms"] != EXPECTED_TRAINING_TERMS
    ):
        raise RuntimeError("Full candidate data set does not match production row contracts")
    _atomic_write_frame(candidates, output_path)
    manifest = {
        "artifact_schema_version": 1,
        "candidate_sampling_schema_version": CANDIDATE_SAMPLING_SCHEMA_VERSION,
        "training_mode": "sample" if args.sample_terms is not None else "full",
        "code_revision": revision,
        "candidate_sampling": {
            "strategy": "test_shaped_bm25_category_random",
            **config,
            "positive_reference_rows": len(positives),
        },
        "training": stats,
        "source_data_sha256": {
            name: sha256_file(os.path.join(DATA_DIR, name)) for name in SOURCE_FILES
        },
        "artifact": os.path.basename(output_path),
        "sha256": sha256_file(output_path),
    }
    if (
        args.sample_terms is None
        and manifest["candidate_sampling"] != PRODUCTION_CANDIDATE_SAMPLING
    ):
        raise RuntimeError("Full candidate data set must use the locked production sampling contract")
    _atomic_write_json(manifest, manifest_path)
    print(f"dataset={output_path}\nmanifest={manifest_path}\nrows={len(candidates):,}")
    return manifest_path


if __name__ == "__main__":
    main()
