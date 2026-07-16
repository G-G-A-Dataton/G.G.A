"""Verify the complete local evidence required by the 17 July freeze."""

import argparse
import json
import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from scripts.verify_environment import verify_environment
from src.data_freeze import verify_data_freeze
from src.delivery_artifacts import validate_delivery_manifest
from src.oof_artifacts import sha256_file
from src.solution_contract import validate_deploy_decision
from src.validate_submission import validate_submission


def _project_path(relative_path):
    path = os.path.realpath(os.path.join(PROJECT_ROOT, relative_path))
    if os.path.commonpath([PROJECT_ROOT, path]) != PROJECT_ROOT:
        raise ValueError(f"Artifact path escapes the project: {relative_path}")
    return path


def _validate_candidate_set(candidate_manifest_path, data_dir, accepted):
    with open(candidate_manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("status") != "PASS" or manifest.get("submission_rows") != 3_359_679:
        raise ValueError("Final candidate manifest is not approved for full delivery")
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("Exactly two final candidates are required")
    verified = []
    for expected_rank, candidate in enumerate(candidates, start=1):
        if candidate.get("rank") != expected_rank or candidate.get("qa_status") != "PASS":
            raise ValueError("Final candidate ranking or QA status is invalid")
        path = _project_path(candidate["path"])
        actual_hash = sha256_file(path)
        if actual_hash != candidate.get("sha256"):
            raise ValueError(f"Final candidate SHA-256 mismatch: {path}")
        if not validate_submission(
            path,
            sample_submission_path=os.path.join(data_dir, "sample_submission.csv"),
            expected_rows=manifest["submission_rows"],
            verbose=False,
        ):
            raise ValueError(f"Final candidate submission QA failed: {path}")
        verified.append(
            {
                "rank": expected_rank,
                "path": candidate["path"],
                "sha256": actual_hash,
            }
        )
    accepted_hash = accepted["files"]["submission"]["sha256"]
    if verified[0]["sha256"] != accepted_hash:
        raise ValueError("Primary final candidate does not match accepted delivery")
    return verified


def _verify_entrypoints():
    required = [
        "step1.sh",
        "step2.sh",
        "step3.sh",
        "SOLUTION_README.md",
        "RUNBOOK.md",
    ]
    for relative_path in required:
        path = _project_path(relative_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing delivery file: {relative_path}")
        if relative_path.endswith(".sh") and not os.access(path, os.X_OK):
            raise PermissionError(f"Delivery entrypoint is not executable: {relative_path}")
    return required


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify the July 17 local freeze")
    parser.add_argument(
        "--output",
        default=os.path.join(PROJECT_ROOT, "outputs", "july_17_freeze.json"),
    )
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args(argv)
    if args.require_clean:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            raise RuntimeError("July 17 freeze requires a clean source worktree")

    data_dir = os.path.join(PROJECT_ROOT, "datasets")
    model_dir = os.path.join(PROJECT_ROOT, "outputs", "ensemble_artifacts")
    environment = verify_environment(
        os.path.join(PROJECT_ROOT, "requirements.txt"),
        os.path.join(PROJECT_ROOT, "configs", "final_v1.json"),
        lock_path=os.path.join(PROJECT_ROOT, "requirements.lock"),
    )
    data = verify_data_freeze(
        os.path.join(PROJECT_ROOT, "configs", "final_v1.json"),
        data_dir,
        250_000,
    )
    accepted = validate_delivery_manifest(
        os.path.join(PROJECT_ROOT, "outputs", "submission_v2.manifest.json"),
        project_root=PROJECT_ROOT,
        source_data_dir=data_dir,
        require_full=True,
    )
    decision = validate_deploy_decision(model_dir, data_dir, require_full=True)
    candidates = _validate_candidate_set(
        os.path.join(
            PROJECT_ROOT,
            "outputs",
            "final_candidates",
            "candidate_set.json",
        ),
        data_dir,
        accepted,
    )
    result = {
        "freeze_schema_version": 1,
        "status": "PASS",
        "environment": environment,
        "data_files": data,
        "deploy": decision["deploy"],
        "candidates": candidates,
        "entrypoints": _verify_entrypoints(),
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, output_path)
    print(f"status=PASS freeze={output_path}")
    return result


if __name__ == "__main__":
    main()
