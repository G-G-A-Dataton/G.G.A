"""Candidate-set relative features shared by training and inference."""

import numpy as np
import pandas as pd


CONTEXT_FEATURE_SCHEMA_VERSION = 1
CONTEXT_FEATURE_COLS = [
    "candidate_count_log1p",
    "query_title_overlap_rank",
    "query_title_overlap_gap",
    "query_title_overlap_delta_mean",
    "query_title_coverage_rank",
    "query_category_overlap_rank",
    "tfidf_cosine_rank",
    "tfidf_cosine_gap",
    "tfidf_cosine_delta_mean",
]


def add_context_features(frame, group_column="term_id", copy=True):
    """Add stable relative scores within each complete query candidate set."""
    score_columns = [
        "query_title_overlap",
        "query_title_coverage",
        "query_category_overlap",
        "tfidf_cosine",
    ]
    required = {group_column, *score_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Context feature input is missing columns: {missing}")
    if frame.empty or frame[group_column].isna().any():
        raise ValueError("Context feature input must contain non-null candidate groups")
    for column in score_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Context feature column {column} must be finite numeric")

    out = frame.copy() if copy else frame
    grouped = out.groupby(group_column, sort=False, observed=True)
    group_sizes = grouped[group_column].transform("size").astype("float32")
    out["candidate_count_log1p"] = np.log1p(group_sizes).astype("float32")

    for source, prefix in (
        ("query_title_overlap", "query_title_overlap"),
        ("query_title_coverage", "query_title_coverage"),
        ("query_category_overlap", "query_category_overlap"),
        ("tfidf_cosine", "tfidf_cosine"),
    ):
        rank = grouped[source].rank(method="average", ascending=False)
        denominator = (group_sizes - 1.0).clip(lower=1.0)
        out[f"{prefix}_rank"] = (1.0 - (rank - 1.0) / denominator).astype(
            "float32"
        )

    for source in ("query_title_overlap", "tfidf_cosine"):
        maximum = grouped[source].transform("max")
        mean = grouped[source].transform("mean")
        out[f"{source}_gap"] = (maximum - out[source]).astype("float32")
        out[f"{source}_delta_mean"] = (out[source] - mean).astype("float32")

    return out
