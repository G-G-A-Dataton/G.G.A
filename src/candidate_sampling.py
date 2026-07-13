"""Test-shaped, leakage-free candidate generation for positive-only training data."""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.negative_sampling import generate_random_negatives


CANDIDATE_SAMPLING_SCHEMA_VERSION = 1


def sample_complete_terms(positive_df, n_terms, random_state=42):
    """Sample complete query groups instead of truncating positive rows."""
    _validate_positive_pairs(positive_df, "positive_df")
    if isinstance(n_terms, bool) or not isinstance(n_terms, int) or n_terms <= 0:
        raise ValueError("n_terms must be a positive integer")
    term_ids = positive_df["term_id"].drop_duplicates().to_numpy()
    if n_terms >= len(term_ids):
        return positive_df.copy().reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    selected = set(rng.choice(term_ids, size=n_terms, replace=False).tolist())
    return positive_df[positive_df["term_id"].isin(selected)].copy().reset_index(
        drop=True
    )


def candidate_targets(positive_df, min_candidates=100, dense_multiplier=2.0):
    """Return the target candidate and negative count for every query."""
    _validate_positive_pairs(positive_df, "positive_df")
    if (
        isinstance(min_candidates, bool)
        or not isinstance(min_candidates, int)
        or min_candidates <= 0
    ):
        raise ValueError("min_candidates must be a positive integer")
    if (
        isinstance(dense_multiplier, bool)
        or not isinstance(dense_multiplier, (int, float, np.integer, np.floating))
        or not np.isfinite(dense_multiplier)
        or dense_multiplier < 1.0
    ):
        raise ValueError("dense_multiplier must be a finite number >= 1")

    positive_counts = positive_df.groupby("term_id", sort=False).size().astype("int64")
    dense_targets = np.ceil(positive_counts * float(dense_multiplier)).astype("int64")
    total_targets = dense_targets.clip(lower=min_candidates)
    total_targets = pd.concat([total_targets, positive_counts], axis=1).max(axis=1)
    return pd.DataFrame(
        {
            "positive_count": positive_counts,
            "candidate_count": total_targets.astype("int64"),
            "negative_count": (total_targets - positive_counts).astype("int64"),
        }
    )


def build_test_shaped_training_set(
    positive_df,
    items_df,
    *,
    positive_reference_df=None,
    min_candidates=100,
    dense_multiplier=2.0,
    category_hard_fraction=0.5,
    random_state=42,
    verbose=True,
):
    """Build per-query candidate sets that mirror the submission distribution.

    Known positives are retained. A configurable share of the negative quota is
    sampled from the positive products' level-2 category; the remainder is
    filled from the full catalog. Every pair is unique and all known positives
    are excluded from both negative sources.
    """
    _validate_positive_pairs(positive_df, "positive_df")
    positive_reference_df = (
        positive_df if positive_reference_df is None else positive_reference_df
    )
    _validate_positive_pairs(positive_reference_df, "positive_reference_df")
    _validate_catalog(items_df)
    if (
        isinstance(category_hard_fraction, bool)
        or not isinstance(
            category_hard_fraction, (int, float, np.integer, np.floating)
        )
        or not np.isfinite(category_hard_fraction)
        or not 0.0 <= float(category_hard_fraction) <= 1.0
    ):
        raise ValueError("category_hard_fraction must be in [0, 1]")

    selected_pairs = positive_df[["term_id", "item_id"]].drop_duplicates()
    reference_pairs = positive_reference_df[["term_id", "item_id"]].drop_duplicates()
    selected_keys = pd.MultiIndex.from_frame(selected_pairs)
    reference_keys = pd.MultiIndex.from_frame(reference_pairs)
    if not selected_keys.isin(reference_keys).all():
        raise ValueError("positive_reference_df must contain every selected positive pair")

    selected_terms = set(selected_pairs["term_id"].tolist())
    complete_reference = reference_pairs[
        reference_pairs["term_id"].isin(selected_terms)
    ]
    if len(complete_reference) != len(selected_pairs):
        raise ValueError(
            "positive_df must contain complete positive groups for every selected term"
        )

    targets = candidate_targets(
        selected_pairs,
        min_candidates=min_candidates,
        dense_multiplier=dense_multiplier,
    )
    blocked_by_term = {
        term_id: set(group["item_id"].tolist())
        for term_id, group in complete_reference.groupby("term_id", sort=False)
    }

    catalog = items_df[["item_id", "category"]].copy()
    catalog["category_key"] = _category_keys(catalog["category"])
    category_pools = {
        key: group["item_id"].to_numpy()
        for key, group in catalog[catalog["category_key"] != ""].groupby(
            "category_key", sort=False, observed=True
        )
    }
    positive_categories = selected_pairs.merge(
        catalog[["item_id", "category_key"]],
        on="item_id",
        how="left",
        validate="many_to_one",
    )
    if positive_categories["category_key"].isna().any():
        raise ValueError("positive_df contains item_id values missing from items_df")
    categories_by_term = {
        term_id: group.loc[group["category_key"] != "", "category_key"]
        .drop_duplicates()
        .tolist()
        for term_id, group in positive_categories.groupby("term_id", sort=False)
    }

    rng = np.random.default_rng(random_state)
    hard_terms = []
    hard_items = []
    hard_counts = defaultdict(int)
    for term_id, row in targets.sort_index().iterrows():
        desired = int(round(row["negative_count"] * float(category_hard_fraction)))
        if desired == 0:
            continue
        category_keys = categories_by_term.get(term_id, [])
        if not category_keys:
            continue
        selected = set()
        blocked = blocked_by_term.get(term_id, set())
        max_attempts = max(100, desired * 30)
        attempts = 0
        while len(selected) < desired and attempts < max_attempts:
            category_key = category_keys[int(rng.integers(len(category_keys)))]
            pool = category_pools.get(category_key)
            if pool is None or len(pool) == 0:
                break
            candidate = pool[int(rng.integers(len(pool)))]
            attempts += 1
            if candidate not in blocked and candidate not in selected:
                selected.add(candidate)
        if selected:
            ordered = sorted(selected, key=str)
            hard_terms.extend([term_id] * len(ordered))
            hard_items.extend(ordered)
            hard_counts[term_id] = len(ordered)

    category_negatives = pd.DataFrame(
        {"term_id": hard_terms, "item_id": hard_items, "label": 0}
    )
    category_negatives["neg_source"] = "category"

    missing_counts = targets["negative_count"].sub(
        pd.Series(hard_counts, dtype="int64"), fill_value=0
    ).astype("int64")
    missing_terms = np.repeat(missing_counts.index.to_numpy(), missing_counts.to_numpy())
    if len(missing_terms):
        quota = pd.DataFrame({"term_id": missing_terms, "item_id": pd.NA})
        random_negatives = generate_random_negatives(
            quota,
            items_df,
            ratio=1,
            random_state=random_state,
            verbose=False,
            positive_reference_df=positive_reference_df,
            excluded_pairs_df=category_negatives,
        )
        random_negatives["neg_source"] = "random"
    else:
        random_negatives = pd.DataFrame(
            columns=["term_id", "item_id", "label", "neg_source"]
        )

    positives = selected_pairs.copy()
    positives["label"] = 1
    positives["neg_source"] = "positive"
    result = pd.concat(
        [positives, category_negatives, random_negatives],
        ignore_index=True,
    )
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    _validate_result(result, reference_pairs, targets)

    if verbose:
        source_counts = result["neg_source"].value_counts()
        print(
            "[candidate_sampling] "
            f"{len(targets):,} terms, {len(result):,} candidates, "
            f"{source_counts.get('category', 0):,} category negatives, "
            f"{source_counts.get('random', 0):,} random negatives"
        )
    return result


def candidate_distribution(frame):
    """Summarize the per-query candidate distribution for manifests/reports."""
    if not {"term_id", "label"}.issubset(frame.columns) or frame.empty:
        raise ValueError("frame must contain non-empty term_id and label columns")
    counts = frame.groupby("term_id").size()
    positives = frame.groupby("term_id")["label"].sum()
    return {
        "terms": int(len(counts)),
        "rows": int(len(frame)),
        "positive_rows": int(frame["label"].sum()),
        "negative_rows": int((frame["label"] == 0).sum()),
        "candidate_mean": float(counts.mean()),
        "candidate_median": float(counts.median()),
        "candidate_p99": float(counts.quantile(0.99)),
        "candidate_max": int(counts.max()),
        "positive_prevalence": float(frame["label"].mean()),
        "positive_per_term_mean": float(positives.mean()),
    }


def _category_key(value):
    if not isinstance(value, str):
        return ""
    levels = [level.strip().casefold() for level in value.split("/") if level.strip()]
    return "/".join(levels[:2])


def _category_keys(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = series.cat.categories
        key_by_code = np.asarray([_category_key(value) for value in categories])
        codes = series.cat.codes.to_numpy()
        values = np.full(len(series), "", dtype=object)
        valid = codes >= 0
        values[valid] = key_by_code[codes[valid]]
        return pd.Series(values, index=series.index, dtype="string")
    return series.map(_category_key).astype("string")


def _validate_positive_pairs(frame, name):
    required = {"term_id", "item_id"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError(f"{name} must contain non-empty term_id and item_id columns")
    if frame[["term_id", "item_id"]].isna().any().any():
        raise ValueError(f"{name} contains null pair values")
    if frame.duplicated(["term_id", "item_id"]).any():
        raise ValueError(f"{name} contains duplicate term-item pairs")
    if "label" in frame.columns and not (frame["label"] == 1).all():
        raise ValueError(f"{name} must contain only known positives")


def _validate_catalog(items_df):
    required = {"item_id", "category"}
    if not required.issubset(items_df.columns) or items_df.empty:
        raise ValueError("items_df must contain non-empty item_id and category columns")
    if items_df["item_id"].isna().any() or items_df["item_id"].duplicated().any():
        raise ValueError("items_df item_id values must be non-null and unique")


def _validate_result(result, reference_pairs, targets):
    if result.duplicated(["term_id", "item_id"]).any():
        raise RuntimeError("Candidate set contains duplicate term-item pairs")
    negatives = result[result["label"] == 0]
    reference_index = pd.MultiIndex.from_frame(reference_pairs)
    negative_index = pd.MultiIndex.from_frame(negatives[["term_id", "item_id"]])
    if negative_index.isin(reference_index).any():
        raise RuntimeError("Candidate negatives overlap known positive pairs")
    actual = result.groupby("term_id").size().sort_index()
    expected = targets["candidate_count"].sort_index()
    if not actual.equals(expected):
        raise RuntimeError("Per-term candidate quotas were not satisfied")
