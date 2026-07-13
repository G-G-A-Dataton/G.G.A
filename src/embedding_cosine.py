"""Strict manifest-backed embedding lookup and cosine features."""

import json
import os

import numpy as np
import pandas as pd

from src.embedding_batch import EMBEDDING_ARTIFACT_SCHEMA_VERSION, sha256_file


class EmbeddingIndex:
    """Memory-mapped embedding matrix with vectorized, strict ID lookup."""

    def __init__(self, emb_path, ids_path, manifest_path=None, expected_target=None):
        manifest_path = manifest_path or _infer_manifest_path(emb_path)
        for path in (emb_path, ids_path, manifest_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing embedding artifact: {path}")
        with open(manifest_path, encoding="utf-8") as manifest_file:
            self.manifest = json.load(manifest_file)
        errors = []
        if (
            self.manifest.get("artifact_schema_version")
            != EMBEDDING_ARTIFACT_SCHEMA_VERSION
        ):
            errors.append("unsupported artifact schema")
        if expected_target and self.manifest.get("target") != expected_target:
            errors.append("target mismatch")
        expected_artifacts = {
            "embeddings": os.path.basename(emb_path),
            "ids": os.path.basename(ids_path),
        }
        if self.manifest.get("artifacts") != expected_artifacts:
            errors.append("artifact filename mismatch")
        for path in (emb_path, ids_path):
            filename = os.path.basename(path)
            if self.manifest.get("sha256", {}).get(filename) != sha256_file(path):
                errors.append(f"SHA-256 mismatch: {filename}")
        if errors:
            raise ValueError("Invalid embedding manifest: " + "; ".join(errors))

        self.embeddings = np.load(emb_path, mmap_mode="r", allow_pickle=False)
        ids = np.load(ids_path, mmap_mode="r", allow_pickle=False)
        if (
            self.embeddings.ndim != 2
            or ids.ndim != 1
            or len(ids) != len(self.embeddings)
            or self.embeddings.shape
            != (self.manifest.get("rows"), self.manifest.get("dimension"))
            or not self.manifest.get("normalized")
        ):
            raise ValueError("Embedding matrix shape does not match its manifest")
        self.ids = pd.Index(ids.astype(str))
        if self.ids.has_duplicates:
            raise ValueError("Embedding IDs must be unique")
        self.dimension = self.embeddings.shape[1]

    def get(self, id_):
        positions = self.ids.get_indexer([str(id_)])
        if positions[0] < 0:
            raise KeyError(f"Unknown embedding ID: {id_}")
        return self.embeddings[positions[0]]

    def get_batch(self, ids):
        requested = np.asarray(ids, dtype=str)
        if requested.ndim != 1:
            raise ValueError("Embedding lookup IDs must be one-dimensional")
        positions = self.ids.get_indexer(requested)
        missing = positions < 0
        if missing.any():
            examples = requested[missing][:5].tolist()
            raise KeyError(
                f"Embedding index is missing {int(missing.sum())} IDs; examples={examples}"
            )
        return np.asarray(self.embeddings[positions], dtype=np.float32)


def compute_cosine_batch(query_embs, item_embs):
    query_embs = np.asarray(query_embs)
    item_embs = np.asarray(item_embs)
    if (
        query_embs.ndim != 2
        or item_embs.ndim != 2
        or query_embs.shape != item_embs.shape
        or len(query_embs) == 0
        or not np.isfinite(query_embs).all()
        or not np.isfinite(item_embs).all()
    ):
        raise ValueError("Cosine inputs must be aligned finite non-empty matrices")
    return np.einsum("nd,nd->n", query_embs, item_embs).astype(np.float32)


def add_embedding_cosine_feature(
    df,
    term_index,
    item_index,
    term_id_col="term_id",
    item_id_col="item_id",
    out_col="embedding_cosine",
    copy=True,
):
    if term_index is None or item_index is None:
        raise ValueError("Both verified embedding indexes are required")
    if term_index.dimension != item_index.dimension:
        raise ValueError("Term and item embedding dimensions do not match")
    missing = sorted({term_id_col, item_id_col} - set(df.columns))
    if missing or df.empty:
        raise ValueError(f"Embedding feature input is invalid; missing={missing}")
    out = df.copy() if copy else df
    query_embs = term_index.get_batch(out[term_id_col].astype(str).to_numpy())
    item_embs = item_index.get_batch(out[item_id_col].astype(str).to_numpy())
    out[out_col] = compute_cosine_batch(query_embs, item_embs)
    return out


def load_embedding_indexes(project_root):
    emb_dir = os.path.join(project_root, "outputs", "embeddings")
    term_index = EmbeddingIndex(
        os.path.join(emb_dir, "term_embeddings.npy"),
        os.path.join(emb_dir, "term_ids.npy"),
        os.path.join(emb_dir, "term_manifest.json"),
        expected_target="term",
    )
    item_index = EmbeddingIndex(
        os.path.join(emb_dir, "item_embeddings.npy"),
        os.path.join(emb_dir, "item_ids.npy"),
        os.path.join(emb_dir, "item_manifest.json"),
        expected_target="item",
    )
    if term_index.manifest.get("model") != item_index.manifest.get("model"):
        raise ValueError("Term and item embeddings were produced by different models")
    return term_index, item_index


def _infer_manifest_path(embedding_path):
    suffix = "_embeddings.npy"
    if not embedding_path.endswith(suffix):
        raise ValueError("manifest_path is required for non-standard embedding filenames")
    return embedding_path[: -len(suffix)] + "_manifest.json"
