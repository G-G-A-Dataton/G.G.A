"""Versioned source-data freeze verification."""

import hashlib
import json
import os

import pandas as pd


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows_and_columns(path, chunk_size=250_000):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    reader = pd.read_csv(path, chunksize=chunk_size)
    columns = None
    rows = 0
    with reader:
        for chunk in reader:
            if columns is None:
                columns = chunk.columns.tolist()
            elif chunk.columns.tolist() != columns:
                raise ValueError(f"CSV column order changed within {path}")
            rows += len(chunk)
    return rows, columns or []


def verify_data_freeze(config_path, data_dir, chunk_size=250_000):
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    if config.get("config_schema_version") != 1:
        raise ValueError("Unsupported final config schema")
    freeze = config.get("data_freeze", {})
    expected_files = freeze.get("files")
    if not freeze.get("version") or not isinstance(expected_files, dict):
        raise ValueError("Final config has no valid data freeze")

    results = {}
    errors = []
    for filename, expected in expected_files.items():
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            errors.append(f"missing file: {filename}")
            continue
        rows, columns = count_csv_rows_and_columns(path, chunk_size=chunk_size)
        actual = {
            "bytes": os.path.getsize(path),
            "rows": rows,
            "columns": columns,
            "sha256": sha256_file(path),
        }
        results[filename] = actual
        for field in ("bytes", "rows", "columns", "sha256"):
            if actual[field] != expected.get(field):
                errors.append(
                    f"{filename} {field} mismatch: "
                    f"{actual[field]} != {expected.get(field)}"
                )
    if errors:
        raise ValueError("Data freeze verification failed: " + "; ".join(errors))
    return {"version": freeze["version"], "files": results}
