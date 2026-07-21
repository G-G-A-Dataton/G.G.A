"""Integrity contract for the final submission and its selection evidence."""

import json
import os
import re

import pandas as pd

from src.oof_artifacts import sha256_file, validate_oof_artifacts


DELIVERY_SCHEMA_VERSION = 1


def write_delivery_manifest(
    path,
    *,
    project_root,
    submission_path,
    decision_path,
    oof_manifest_path,
    code_revision,
    submission_rows,
    positive_rows,
):
    """Atomically bind a candidate CSV to its OOF and deploy decision."""
    project_root = os.path.realpath(project_root)
    paths = {
        "submission": os.path.realpath(submission_path),
        "decision": os.path.realpath(decision_path),
        "oof_manifest": os.path.realpath(oof_manifest_path),
    }
    for name, artifact_path in paths.items():
        if not _is_within(artifact_path, project_root) or not os.path.isfile(
            artifact_path
        ):
            raise ValueError(f"{name} must be an existing file inside project_root")
    if (
        not isinstance(submission_rows, int)
        or isinstance(submission_rows, bool)
        or submission_rows <= 0
        or not isinstance(positive_rows, int)
        or isinstance(positive_rows, bool)
        or not 0 <= positive_rows <= submission_rows
    ):
        raise ValueError("submission row counts are invalid")
    if not re.fullmatch(r"[0-9a-f]{40}", str(code_revision)):
        raise ValueError("code_revision must be a full lowercase Git SHA")

    with open(paths["decision"], encoding="utf-8") as decision_file:
        decision = json.load(decision_file)
    with open(paths["oof_manifest"], encoding="utf-8") as oof_file:
        oof_manifest = json.load(oof_file)
    deploy = decision.get("deploy", {})
    deploy_positive_rate = deploy.get("positive_rate")
    if deploy.get("rows") != submission_rows:
        raise ValueError("decision and submission row counts do not match")
    if (
        isinstance(deploy_positive_rate, bool)
        or not isinstance(deploy_positive_rate, (int, float))
        or abs(deploy_positive_rate - positive_rows / submission_rows) > 1e-15
    ):
        raise ValueError("decision and submission positive rates do not match")

    manifest = {
        "delivery_schema_version": DELIVERY_SCHEMA_VERSION,
        "code_revision": code_revision,
        "oof_code_revision": oof_manifest.get("code_revision"),
        "files": {
            name: {
                "path": os.path.relpath(artifact_path, project_root),
                "sha256": sha256_file(artifact_path),
            }
            for name, artifact_path in paths.items()
        },
        "submission": {
            "rows": submission_rows,
            "positive_rows": positive_rows,
            "positive_rate": positive_rows / submission_rows,
            "columns": ["id", "prediction"],
        },
        "deploy": deploy,
        "source_data_sha256": oof_manifest.get("source_data_sha256"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")
    os.replace(temporary_path, path)
    return path


def validate_delivery_manifest(
    path,
    *,
    project_root,
    source_data_dir=None,
    require_full=False,
):
    """Validate hashes, cross-file metadata, and submission row content."""
    project_root = os.path.realpath(project_root)
    with open(path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    errors = []
    if manifest.get("delivery_schema_version") != DELIVERY_SCHEMA_VERSION:
        errors.append("unsupported delivery schema")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("code_revision", ""))):
        errors.append("invalid delivery code revision")
    if not re.fullmatch(
        r"[0-9a-f]{40}", str(manifest.get("oof_code_revision", ""))
    ):
        errors.append("invalid OOF code revision")
    files = manifest.get("files", {})
    if set(files) != {"submission", "decision", "oof_manifest"}:
        errors.append("delivery file contract mismatch")
        paths = {}
    else:
        paths = {}
        for name, record in files.items():
            relative_path = record.get("path") if isinstance(record, dict) else None
            artifact_path = (
                os.path.realpath(os.path.join(project_root, relative_path))
                if isinstance(relative_path, str)
                else ""
            )
            if (
                not artifact_path
                or not _is_within(artifact_path, project_root)
                or not os.path.isfile(artifact_path)
            ):
                errors.append(f"invalid delivery path: {name}")
                continue
            paths[name] = artifact_path
            if sha256_file(artifact_path) != record.get("sha256"):
                errors.append(f"SHA-256 mismatch: {name}")
    if errors:
        raise ValueError("Invalid delivery manifest: " + "; ".join(errors))

    oof_dir = os.path.dirname(paths["oof_manifest"])
    oof_manifest = validate_oof_artifacts(
        oof_dir,
        require_full=require_full,
        source_data_dir=source_data_dir,
    )
    with open(paths["decision"], encoding="utf-8") as decision_file:
        decision = json.load(decision_file)
    submission = manifest.get("submission", {})
    deploy = decision.get("deploy", {})
    if (
        manifest.get("oof_code_revision") != oof_manifest.get("code_revision")
        or manifest.get("source_data_sha256")
        != oof_manifest.get("source_data_sha256")
        or manifest.get("deploy") != deploy
        or submission.get("columns") != ["id", "prediction"]
        or deploy.get("rows") != submission.get("rows")
        or deploy.get("positive_rate") != submission.get("positive_rate")
    ):
        raise ValueError("Invalid delivery manifest: cross-file metadata mismatch")

    rows = 0
    positive_rows = 0
    reader = pd.read_csv(
        paths["submission"],
        dtype={"id": "string", "prediction": "int8"},
        chunksize=250_000,
    )
    try:
        for chunk in reader:
            if chunk.columns.tolist() != ["id", "prediction"]:
                raise ValueError("submission column contract mismatch")
            if chunk["id"].isna().any() or not chunk["prediction"].isin([0, 1]).all():
                raise ValueError("submission contains invalid IDs or predictions")
            rows += len(chunk)
            positive_rows += int(chunk["prediction"].sum())
    finally:
        reader.close()
    if rows != submission.get("rows") or positive_rows != submission.get(
        "positive_rows"
    ):
        raise ValueError("Invalid delivery manifest: submission statistics mismatch")
    return manifest


def _is_within(path, root):
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False
