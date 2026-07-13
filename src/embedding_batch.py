"""Resumable, manifest-verified sentence embedding production."""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.item_text import build_item_texts, clean_text


DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
EMB_DIR = os.path.join(PROJECT_ROOT, "outputs", "embeddings")
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 10_000
BATCH_SIZE = 64
EMBEDDING_ARTIFACT_SCHEMA_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids):
    digest = hashlib.sha256()
    for value in ids:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def load_model(model_name=DEFAULT_MODEL, offline=False):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers is required for embeddings") from exc
    print(f"[embedding_batch] loading model: {model_name}")
    model = SentenceTransformer(model_name, local_files_only=offline)
    print(f"[embedding_batch] device: {model.device}")
    return model


def _atomic_save_array(path, values):
    temporary_path = path + ".tmp"
    with open(temporary_path, "wb") as output_file:
        np.save(output_file, values, allow_pickle=False)
    os.replace(temporary_path, path)


def _atomic_write_json(path, payload):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, path)


def _embedding_dimension(model):
    dimension = model.get_sentence_embedding_dimension()
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("Embedding model returned an invalid dimension")
    return dimension


def _validate_embedding_array(values, expected_rows, expected_dimension, source):
    if values.shape != (expected_rows, expected_dimension):
        raise ValueError(
            f"{source} shape mismatch: {values.shape} != "
            f"{(expected_rows, expected_dimension)}"
        )
    if values.dtype.kind != "f" or not np.isfinite(values).all():
        raise ValueError(f"{source} must contain finite floating-point values")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-3, atol=1e-3):
        raise ValueError(f"{source} contains non-normalized embeddings")


def encode_in_chunks(
    model,
    texts,
    ids,
    out_prefix,
    *,
    chunk_size=CHUNK_SIZE,
    batch_size=BATCH_SIZE,
    model_name=DEFAULT_MODEL,
    target="unknown",
    source_path=None,
    text_provider=None,
    keep_chunks=False,
):
    """Encode rows with verified checkpoints and disk-backed final assembly."""
    if chunk_size <= 0 or batch_size <= 0:
        raise ValueError("chunk_size and batch_size must be positive")
    ids = np.asarray(ids, dtype=str)
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("ids must be a non-empty one-dimensional unique sequence")
    if text_provider is None:
        if texts is None or len(texts) != len(ids):
            raise ValueError("texts and ids must have equal length")
        text_provider = lambda start, end: texts[start:end]
    dimension = _embedding_dimension(model)
    total_rows = len(ids)
    chunk_count = (total_rows + chunk_size - 1) // chunk_size
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    started_at = time.time()
    chunk_paths = []

    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, total_rows)
        chunk_path = f"{out_prefix}_chunk_{chunk_index:05d}.npy"
        metadata_path = chunk_path + ".json"
        expected_metadata = {
            "schema_version": EMBEDDING_ARTIFACT_SCHEMA_VERSION,
            "model": model_name,
            "target": target,
            "start": start,
            "end": end,
            "rows": end - start,
            "dimension": dimension,
            "ids_sha256": sha256_ids(ids[start:end]),
            "normalized": True,
        }
        if os.path.exists(chunk_path) or os.path.exists(metadata_path):
            if not (os.path.exists(chunk_path) and os.path.exists(metadata_path)):
                raise ValueError(f"Incomplete embedding checkpoint: {chunk_path}")
            with open(metadata_path, encoding="utf-8") as metadata_file:
                existing_metadata = json.load(metadata_file)
            if existing_metadata != expected_metadata:
                raise ValueError(f"Stale embedding checkpoint metadata: {chunk_path}")
            existing = np.load(chunk_path, mmap_mode="r", allow_pickle=False)
            _validate_embedding_array(existing, end - start, dimension, chunk_path)
            print(f"[embedding_batch] verified checkpoint {chunk_index + 1}/{chunk_count}")
            chunk_paths.append(chunk_path)
            continue

        chunk_texts = list(text_provider(start, end))
        if len(chunk_texts) != end - start or any(
            not isinstance(text, str) for text in chunk_texts
        ):
            raise ValueError("text_provider returned an invalid chunk")
        values = np.asarray(
            model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        _validate_embedding_array(values, end - start, dimension, chunk_path)
        _atomic_save_array(chunk_path, values)
        _atomic_write_json(metadata_path, expected_metadata)
        chunk_paths.append(chunk_path)
        print(f"[embedding_batch] encoded {end:,}/{total_rows:,}")

    final_path = f"{out_prefix}_embeddings.npy"
    ids_path = f"{out_prefix}_ids.npy"
    temporary_final = final_path + ".tmp"
    final_store = np.lib.format.open_memmap(
        temporary_final,
        mode="w+",
        dtype="float32",
        shape=(total_rows, dimension),
    )
    offset = 0
    for chunk_path in chunk_paths:
        chunk = np.load(chunk_path, mmap_mode="r", allow_pickle=False)
        end = offset + len(chunk)
        final_store[offset:end] = chunk
        offset = end
    final_store.flush()
    del final_store
    os.replace(temporary_final, final_path)
    _atomic_save_array(ids_path, ids)

    manifest_path = f"{out_prefix}_manifest.json"
    manifest = {
        "artifact_schema_version": EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        "model": model_name,
        "target": target,
        "rows": total_rows,
        "dimension": dimension,
        "normalized": True,
        "source": os.path.basename(source_path) if source_path else None,
        "source_sha256": sha256_file(source_path) if source_path else None,
        "artifacts": {
            "embeddings": os.path.basename(final_path),
            "ids": os.path.basename(ids_path),
        },
        "sha256": {
            os.path.basename(final_path): sha256_file(final_path),
            os.path.basename(ids_path): sha256_file(ids_path),
        },
    }
    _atomic_write_json(manifest_path, manifest)

    if not keep_chunks:
        for chunk_path in chunk_paths:
            os.remove(chunk_path)
            os.remove(chunk_path + ".json")
    print(
        f"[embedding_batch] completed {target}: shape=({total_rows}, {dimension}) "
        f"seconds={time.time() - started_at:.1f}"
    )
    return np.load(final_path, mmap_mode="r", allow_pickle=False)


def produce_item_embeddings(
    model,
    *,
    chunk_size=CHUNK_SIZE,
    batch_size=BATCH_SIZE,
    model_name=DEFAULT_MODEL,
    keep_chunks=False,
):
    source_path = os.path.join(DATA_DIR, "items.csv")
    items = pd.read_csv(source_path, dtype={"item_id": "string"})
    return encode_in_chunks(
        model,
        None,
        items["item_id"].to_numpy(dtype=str),
        os.path.join(EMB_DIR, "item"),
        chunk_size=chunk_size,
        batch_size=batch_size,
        model_name=model_name,
        target="item",
        source_path=source_path,
        text_provider=lambda start, end: build_item_texts(
            items.iloc[start:end], include_attrs=True
        ),
        keep_chunks=keep_chunks,
    )


def produce_term_embeddings(
    model,
    *,
    chunk_size=CHUNK_SIZE,
    batch_size=BATCH_SIZE,
    model_name=DEFAULT_MODEL,
    keep_chunks=False,
):
    source_path = os.path.join(DATA_DIR, "terms.csv")
    terms = pd.read_csv(source_path, dtype={"term_id": "string"})
    texts = [clean_text(query) for query in terms["query"]]
    return encode_in_chunks(
        model,
        texts,
        terms["term_id"].to_numpy(dtype=str),
        os.path.join(EMB_DIR, "term"),
        chunk_size=chunk_size,
        batch_size=batch_size,
        model_name=model_name,
        target="term",
        source_path=source_path,
        keep_chunks=keep_chunks,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Produce verified text embeddings")
    parser.add_argument("--target", choices=["items", "terms", "both"], default="both")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--keep-chunks", action="store_true")
    args = parser.parse_args(argv)
    if args.chunk_size <= 0 or args.batch_size <= 0:
        raise ValueError("chunk and batch sizes must be positive")
    model = load_model(args.model, offline=args.offline)
    kwargs = {
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "model_name": args.model,
        "keep_chunks": args.keep_chunks,
    }
    if args.target in ("items", "both"):
        produce_item_embeddings(model, **kwargs)
    if args.target in ("terms", "both"):
        produce_term_embeddings(model, **kwargs)


if __name__ == "__main__":
    main()
