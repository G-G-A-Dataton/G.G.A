"""Validate the repository data-loading and merge contracts."""

import os
import sys
from itertools import zip_longest

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_items, load_terms, merge_pairs


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
EXPECTED_COUNTS = {
    "terms": 50_153,
    "items": 962_873,
    "training_pairs": 250_000,
    "submission_pairs": 3_359_679,
    "sample_submission": 3_359_679,
}


def verify_submission_contract(
    submission_pairs_path,
    sample_submission_path,
    expected_rows=EXPECTED_COUNTS["submission_pairs"],
):
    pair_reader = pd.read_csv(
        submission_pairs_path,
        dtype={"id": "string", "term_id": "string", "item_id": "string"},
        chunksize=250_000,
    )
    sample_reader = pd.read_csv(
        sample_submission_path,
        dtype={"id": "string"},
        chunksize=250_000,
    )
    total_rows = 0
    with pair_reader, sample_reader:
        for pair_chunk, sample_chunk in zip_longest(pair_reader, sample_reader):
            if pair_chunk is None or sample_chunk is None:
                raise ValueError("Submission pairs and sample submission lengths differ")
            if pair_chunk.columns.tolist() != ["id", "term_id", "item_id"]:
                raise ValueError("submission_pairs.csv has an invalid column contract")
            if sample_chunk.columns.tolist() != ["id", "prediction"]:
                raise ValueError("sample_submission.csv has an invalid column contract")
            if pair_chunk.isna().any().any() or sample_chunk.isna().any().any():
                raise ValueError("Submission input files contain null values")
            if not pair_chunk["id"].reset_index(drop=True).equals(
                sample_chunk["id"].reset_index(drop=True)
            ):
                raise ValueError("Submission pair IDs do not match the sample ID order")
            total_rows += len(pair_chunk)

    if total_rows != expected_rows:
        raise ValueError(
            f"Unexpected submission row count: {total_rows:,}"
        )
    return total_rows


def main():
    paths = {
        "terms": os.path.join(DATA_DIR, "terms.csv"),
        "items": os.path.join(DATA_DIR, "items.csv"),
        "training_pairs": os.path.join(DATA_DIR, "training_pairs.csv"),
        "submission_pairs": os.path.join(DATA_DIR, "submission_pairs.csv"),
        "sample_submission": os.path.join(DATA_DIR, "sample_submission.csv"),
    }
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {missing}")

    terms_df = load_terms(paths["terms"])
    items_df = load_items(paths["items"])
    merged_df = merge_pairs(
        paths["training_pairs"], terms_df, items_df, is_train=True
    )

    if terms_df["term_id"].duplicated().any():
        raise ValueError("terms.csv contains duplicate term_id values")
    if items_df["item_id"].duplicated().any():
        raise ValueError("items.csv contains duplicate item_id values")
    if merged_df[["query", "title"]].isna().any().any():
        raise ValueError("Training pairs contain unresolved term_id or item_id values")
    if set(merged_df["label"].unique()) != {1}:
        raise ValueError("training_pairs.csv must contain only positive labels")
    actual_counts = {
        "terms": len(terms_df),
        "items": len(items_df),
        "training_pairs": len(merged_df),
    }
    for name, actual_count in actual_counts.items():
        if actual_count != EXPECTED_COUNTS[name]:
            raise ValueError(
                f"Unexpected {name} row count: {actual_count:,}; "
                f"expected {EXPECTED_COUNTS[name]:,}"
            )
    submission_rows = verify_submission_contract(
        paths["submission_pairs"], paths["sample_submission"]
    )

    print("Pipeline verified successfully.")
    print(f"  Terms: {len(terms_df):,}")
    print(f"  Items: {len(items_df):,}")
    print(f"  Training pairs: {len(merged_df):,}")
    print(f"  Submission pairs: {submission_rows:,}")


if __name__ == "__main__":
    main()
