"""Backward-compatible facade for the canonical mixed-negative sampler."""

from src.train_mix_v2 import (
    build_mixed_training_set as _build_mixed_training_set,
    verify_mix_no_leakage,
)


def build_mixed_training_set(
    train_df,
    terms_df,
    items_df,
    ratio=3,
    top_n=50,
    max_df_ratio=0.15,
    random_state=42,
    verbose=True,
    positive_reference_df=None,
):
    """Delegate to the quota-checked, leakage-free canonical implementation."""
    return _build_mixed_training_set(
        train_df,
        terms_df,
        items_df,
        ratio=ratio,
        bm25_top_n=top_n,
        bm25_max_df_ratio=max_df_ratio,
        random_state=random_state,
        verbose=verbose,
        positive_reference_df=positive_reference_df,
    )


def build_mixed_negative_set(*args, **kwargs):
    """Return only negatives from the canonical mixed training set."""
    frame = build_mixed_training_set(*args, **kwargs)
    return frame.loc[frame["label"] == 0].reset_index(drop=True)
