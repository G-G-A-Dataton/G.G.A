"""Disk-backed feature generation for non-contiguous submission query groups."""

import os

import numpy as np
import pandas as pd

from src.context_features import CONTEXT_FEATURE_COLS
from src.features import build_features
from src.modeling import BASE_MODEL_FEATURE_COLS, MODEL_FEATURE_COLS
from src.tfidf_features import add_tfidf_features


def build_base_feature_store(
    pairs_path,
    terms_df,
    items_df,
    vectorizer,
    *,
    row_count,
    batch_size,
    output_prefix,
):
    """Write base features and term codes in source row order."""
    if row_count <= 0 or batch_size <= 0:
        raise ValueError("row_count and batch_size must be positive")
    os.makedirs(os.path.dirname(os.path.abspath(output_prefix)), exist_ok=True)
    base_path = output_prefix + "_base.npy"
    codes_path = output_prefix + "_term_codes.npy"
    temporary_base = base_path + ".tmp"
    temporary_codes = codes_path + ".tmp"
    base_store = np.lib.format.open_memmap(
        temporary_base,
        mode="w+",
        dtype="float32",
        shape=(row_count, len(BASE_MODEL_FEATURE_COLS)),
    )
    code_store = np.lib.format.open_memmap(
        temporary_codes, mode="w+", dtype="int32", shape=(row_count,)
    )
    term_code_map = pd.Series(
        np.arange(len(terms_df), dtype=np.int32), index=terms_df["term_id"]
    )
    reader = pd.read_csv(
        pairs_path,
        nrows=row_count,
        chunksize=batch_size,
        dtype={"id": "string", "term_id": "string", "item_id": "string"},
    )
    offset = 0
    try:
        for source_batch in reader:
            end = offset + len(source_batch)
            if source_batch.columns.tolist() != ["id", "term_id", "item_id"]:
                raise ValueError("submission_pairs.csv has an invalid column contract")
            term_codes = source_batch["term_id"].map(term_code_map)
            if term_codes.isna().any():
                raise ValueError("submission_pairs.csv contains unresolved term_id values")
            batch = source_batch.merge(
                terms_df, on="term_id", how="left", validate="many_to_one"
            ).merge(items_df, on="item_id", how="left", validate="many_to_one")
            if batch[["query", "title"]].isna().any().any():
                raise ValueError("submission_pairs.csv contains unresolved item_id values")
            batch = build_features(batch, verbose=False, copy=False)
            batch = add_tfidf_features(
                batch, vectorizer, verbose=False, copy=False
            )
            values = batch[BASE_MODEL_FEATURE_COLS].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Base submission features contain non-finite values")
            base_store[offset:end] = values
            code_store[offset:end] = term_codes.to_numpy(dtype=np.int32)
            offset = end
    except Exception:
        for path in (temporary_base, temporary_codes):
            if os.path.exists(path):
                os.remove(path)
        raise
    finally:
        reader.close()
    if offset != row_count:
        raise RuntimeError(f"Feature store row mismatch: {offset:,} != {row_count:,}")
    base_store.flush()
    code_store.flush()
    del base_store, code_store
    os.replace(temporary_base, base_path)
    os.replace(temporary_codes, codes_path)
    return base_path, codes_path


def build_context_feature_store(base_path, codes_path, output_prefix):
    """Compute exact group-relative features over the complete stored dataset."""
    base = np.load(base_path, mmap_mode="r")
    codes = np.load(codes_path, mmap_mode="r")
    if base.ndim != 2 or base.shape[1] != len(BASE_MODEL_FEATURE_COLS):
        raise ValueError("Base feature store has an invalid shape")
    if codes.ndim != 1 or len(codes) != len(base) or (codes < 0).any():
        raise ValueError("Term code store has an invalid shape or values")
    context_path = output_prefix + "_context.npy"
    temporary_path = context_path + ".tmp"
    context = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype="float32",
        shape=(len(base), len(CONTEXT_FEATURE_COLS)),
    )
    group_count = int(codes.max()) + 1
    counts = np.bincount(codes, minlength=group_count).astype(np.float32)
    row_counts = counts[codes]
    context[:, 0] = np.log1p(row_counts)

    specifications = [
        ("query_title_overlap", 1, 2, 3),
        ("query_title_coverage", 4, None, None),
        ("query_category_overlap", 5, None, None),
        ("tfidf_cosine", 6, 7, 8),
    ]
    for source, rank_column, gap_column, delta_column in specifications:
        source_index = BASE_MODEL_FEATURE_COLS.index(source)
        values = np.asarray(base[:, source_index], dtype=np.float32)
        ranks = (
            pd.Series(values)
            .groupby(np.asarray(codes), sort=False)
            .rank(method="average", ascending=False)
            .to_numpy(dtype=np.float32)
        )
        denominator = np.maximum(row_counts - 1.0, 1.0)
        context[:, rank_column] = 1.0 - (ranks - 1.0) / denominator
        if gap_column is not None:
            maxima = np.full(group_count, -np.inf, dtype=np.float32)
            np.maximum.at(maxima, codes, values)
            sums = np.bincount(codes, weights=values, minlength=group_count)
            means = np.divide(
                sums,
                counts,
                out=np.zeros(group_count, dtype=np.float64),
                where=counts != 0,
            )
            context[:, gap_column] = maxima[codes] - values
            context[:, delta_column] = values - means[codes]

    if not np.isfinite(context).all():
        raise RuntimeError("Context feature store contains non-finite values")
    context.flush()
    del context, base, codes
    os.replace(temporary_path, context_path)
    return context_path


def load_feature_batch(base_store, context_store, start, end):
    """Return a named in-memory model batch from two memory-mapped stores."""
    if start < 0 or end <= start or end > len(base_store):
        raise ValueError("Invalid feature batch bounds")
    values = np.concatenate(
        [base_store[start:end], context_store[start:end]], axis=1
    )
    return pd.DataFrame(values, columns=MODEL_FEATURE_COLS)


def remove_feature_stores(*paths):
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)
