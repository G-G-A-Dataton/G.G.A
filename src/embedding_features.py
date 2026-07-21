"""Backward-compatible imports for verified embedding production.

Live row-by-row encoding was removed because it duplicated texts, could change
the feature contract silently, and had no artifact integrity checks.
"""

from src.embedding_batch import DEFAULT_MODEL, load_model
from src.embedding_cosine import (
    EmbeddingIndex,
    add_embedding_cosine_feature,
    compute_cosine_batch,
    load_embedding_indexes,
)


def load_embedding_model(model_name=DEFAULT_MODEL, offline=True):
    """Load a sentence-transformer explicitly; offline is the safe default."""
    return load_model(model_name=model_name, offline=offline)


cosine_similarity_pair = compute_cosine_batch
