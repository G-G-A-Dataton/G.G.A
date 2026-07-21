"""Manifest the exact deterministic candidates generated during step 2."""

import argparse
import json
import os
import sys

import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.oof_artifacts import sha256_file


def build_manifest(candidate_path, model_dump_path):
    manifest_path = os.path.join(model_dump_path, "oof_manifest.json")
    with open(manifest_path, encoding="utf-8") as manifest_file:
        model_manifest = json.load(manifest_file)
    columns = None
    rows = 0
    label_counts = {"0": 0, "1": 0}
    source_counts = {}
    reader = pd.read_csv(
        candidate_path,
        dtype={"term_id": "string", "item_id": "string", "label": "int8"},
        chunksize=250_000,
    )
    try:
        for chunk in reader:
            if columns is None:
                columns = chunk.columns.tolist()
            if chunk.columns.tolist() != [
                "term_id",
                "item_id",
                "label",
                "neg_source",
            ]:
                raise ValueError("Generated candidate CSV has an invalid schema")
            if chunk.isna().any().any() or not chunk["label"].isin([0, 1]).all():
                raise ValueError("Generated candidate CSV contains invalid values")
            rows += len(chunk)
            for label, count in chunk["label"].value_counts().items():
                label_counts[str(int(label))] += int(count)
            for source, count in chunk["neg_source"].value_counts().items():
                source_counts[str(source)] = source_counts.get(str(source), 0) + int(
                    count
                )
    finally:
        reader.close()
    expected = model_manifest["training"]
    if (
        rows != expected["rows"]
        or label_counts["1"] != expected["positive_rows"]
        or label_counts["0"] != expected["negative_rows"]
    ):
        raise ValueError("Generated candidates do not match the model manifest")
    return {
        "extra_data_schema_version": 1,
        "external_data_used": False,
        "synthetic_data_used": False,
        "pseudo_labeled_data_used": False,
        "generator": "scripts/training/run_model_shortlist.py",
        "file": os.path.basename(candidate_path),
        "columns": columns,
        "rows": rows,
        "label_rows": label_counts,
        "source_rows": dict(sorted(source_counts.items())),
        "candidate_sampling": model_manifest["candidate_sampling"],
        "sha256": sha256_file(candidate_path),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Manifest generated training data")
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--model-dump-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = build_manifest(
        os.path.abspath(args.candidate_path),
        os.path.abspath(args.model_dump_path),
    )
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, output_path)
    print(f"extra_data_manifest={output_path} rows={payload['rows']}")
    return output_path


if __name__ == "__main__":
    main()
