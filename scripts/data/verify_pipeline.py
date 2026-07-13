"""Validate the repository data-loading and merge contracts."""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_items, load_terms, merge_pairs


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")


def main():
    paths = {
        "terms": os.path.join(DATA_DIR, "terms.csv"),
        "items": os.path.join(DATA_DIR, "items.csv"),
        "training_pairs": os.path.join(DATA_DIR, "training_pairs.csv"),
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

    print("Pipeline verified successfully.")
    print(f"  Terms: {len(terms_df):,}")
    print(f"  Items: {len(items_df):,}")
    print(f"  Training pairs: {len(merged_df):,}")


if __name__ == "__main__":
    main()
