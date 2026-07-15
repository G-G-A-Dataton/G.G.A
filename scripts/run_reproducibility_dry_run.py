"""Reproduce the accepted submission from a clean local clone, offline."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILES = (
    "terms.csv",
    "items.csv",
    "training_pairs.csv",
    "submission_pairs.csv",
    "sample_submission.csv",
)
ACCEPTED_OUTPUT_FILES = (
    "submission_v2.csv",
    "submission_v2.manifest.json",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_test_count(output):
    match = re.search(r"Ran (\d+) tests?", output)
    if not match:
        raise ValueError("test output does not contain a unittest count")
    return int(match.group(1))


def hardlink_or_copy(source, destination):
    """Materialize an external runtime asset without mutating the source."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def materialize_tree(source, destination):
    methods = set()
    source = Path(source)
    for path in sorted(source.rglob("*")):
        if path.is_file():
            methods.add(
                hardlink_or_copy(
                    path,
                    Path(destination) / path.relative_to(source),
                )
            )
    return sorted(methods)


def run_step(name, command, *, cwd, env):
    started = time.monotonic()
    print(f"[reproducibility] {name}: {shlex.join(map(str, command))}", flush=True)
    completed = subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    duration = time.monotonic() - started
    if completed.returncode != 0:
        print(completed.stdout, file=sys.stderr)
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    return {
        "name": name,
        "command": shlex.join(map(str, command)),
        "duration_seconds": round(duration, 3),
        "output": completed.stdout,
    }


def require_clean_revision():
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("Reproducibility dry-run requires a clean source worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def isolated_environment(python, workspace):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        {
            "PATH": os.pathsep.join(
                [str(Path(python).resolve().parent), "/usr/bin", "/bin"]
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": str(workspace / "matplotlib"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HOME": str(workspace / "home"),
        }
    )
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def clone_clean_revision(revision, destination, env):
    clone_env = env.copy()
    clone_env["GIT_LFS_SKIP_SMUDGE"] = "1"
    subprocess.run(
        [
            "git",
            "clone",
            "--local",
            "--no-hardlinks",
            "--quiet",
            "--no-checkout",
            str(PROJECT_ROOT),
            str(destination),
        ],
        check=True,
        env=clone_env,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "--detach", revision],
        cwd=destination,
        check=True,
        env=clone_env,
    )


def materialize_runtime_assets(snapshot):
    methods = set()
    for filename in DATA_FILES:
        methods.add(
            hardlink_or_copy(
                PROJECT_ROOT / "datasets" / filename,
                snapshot / "datasets" / filename,
            )
        )
    methods.update(
        materialize_tree(
            PROJECT_ROOT / "outputs" / "ensemble_artifacts",
            snapshot / "outputs" / "ensemble_artifacts",
        )
    )
    for filename in ACCEPTED_OUTPUT_FILES:
        methods.add(
            hardlink_or_copy(
                PROJECT_ROOT / "outputs" / filename,
                snapshot / "outputs" / filename,
            )
        )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=snapshot,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"Clean clone changed after asset materialization: {status}")
    return sorted(methods)


def write_reports(payload, markdown_path, json_path):
    markdown_path = Path(markdown_path)
    json_path = Path(json_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_json, json_path)

    lines = [
        "# July 16 Reproducibility Dry-Run",
        "",
        f"- Status: **{payload['status']}**",
        f"- Source revision: `{payload['source_revision']}`",
        f"- Artifact revision: `{payload['artifact_revision']}`",
        f"- Python: `{payload['python_version']}`",
        f"- Hash-locked packages: `{payload['locked_packages']}`",
        f"- Tests: `{payload['tests_passed']}/{payload['tests_passed']}` passed",
        f"- Network during dry-run: `{payload['network_access']}`",
        f"- Asset materialization: `{', '.join(payload['asset_materialization'])}`",
        "",
        "## Deterministic Delivery Check",
        "",
        f"- Accepted submission SHA-256: `{payload['accepted_submission_sha256']}`",
        f"- Reproduced submission SHA-256: `{payload['reproduced_submission_sha256']}`",
        f"- Byte-identical: **{str(payload['byte_identical']).lower()}**",
        f"- Rows: `{payload['submission_rows']:,}`",
        f"- Positive rows: `{payload['positive_rows']:,}`",
        "",
        "## Executed Gates",
        "",
        "| Gate | Seconds |",
        "|---|---:|",
    ]
    for step in payload["steps"]:
        lines.append(f"| `{step['name']}` | {step['duration_seconds']:.3f} |")
    lines.extend(
        [
            "",
            "The dry-run used a detached local clone, disabled Python user-site",
            "packages and online model access, revalidated frozen data and all",
            "delivery hashes, and rebuilt the final CSV from the accepted OOF/test",
            "probabilities. Full model retraining remains a separate long-running",
            "operation documented in `acceptance_runs.md`.",
            "",
        ]
    )
    temporary_markdown = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    temporary_markdown.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary_markdown, markdown_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify the accepted delivery from a clean, offline local clone"
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--report", default=str(PROJECT_ROOT / "docs" / "reproducibility_dry_run.md")
    )
    parser.add_argument(
        "--json-report",
        default=str(PROJECT_ROOT / "outputs" / "reproducibility_dry_run.json"),
    )
    parser.add_argument("--keep-workdir", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    python = Path(args.python).resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Python interpreter does not exist: {python}")
    revision = require_clean_revision()
    workspace = Path(tempfile.mkdtemp(prefix="gga-reproducibility-"))
    snapshot = workspace / "repository"
    env = isolated_environment(python, workspace)
    steps = []
    started = time.monotonic()
    try:
        clone_clean_revision(revision, snapshot, env)
        materialization = materialize_runtime_assets(snapshot)
        commands = [
            (
                "compile",
                [
                    python,
                    "-m",
                    "compileall",
                    "-q",
                    "src",
                    "scripts",
                    "pipeline",
                    "notebooks",
                    "tests",
                ],
            ),
            ("tests", [python, "-m", "unittest", "discover", "-s", "tests"]),
            (
                "environment",
                [python, "scripts/verify_environment.py", "--lock", "requirements.lock"],
            ),
            ("data_freeze", [python, "scripts/data/verify_data_freeze.py"]),
            ("data_pipeline", [python, "scripts/data/verify_pipeline.py"]),
            (
                "accepted_delivery",
                [
                    python,
                    "-c",
                    "from src.delivery_artifacts import validate_delivery_manifest; "
                    "validate_delivery_manifest('outputs/submission_v2.manifest.json', "
                    "project_root='.', source_data_dir='datasets', require_full=True); "
                    "print('accepted_delivery=valid')",
                ],
            ),
            (
                "submission_reproduction",
                [
                    python,
                    "scripts/analysis/run_ensemble_optimization.py",
                    "--artifact-dir",
                    "outputs/ensemble_artifacts",
                    "--output",
                    "outputs/reproduced_submission_v2.csv",
                    "--report",
                    "docs/reproduced_ensemble_selection.md",
                    "--manifest",
                    "outputs/reproduced_submission_v2.manifest.json",
                ],
            ),
            (
                "submission_qa",
                [
                    python,
                    "scripts/submission/run_submission_qa.py",
                    "outputs/reproduced_submission_v2.csv",
                ],
            ),
        ]
        for name, command in commands:
            steps.append(run_step(name, command, cwd=snapshot, env=env))

        accepted_hash = sha256_file(snapshot / "outputs" / "submission_v2.csv")
        reproduced_hash = sha256_file(
            snapshot / "outputs" / "reproduced_submission_v2.csv"
        )
        if accepted_hash != reproduced_hash:
            raise RuntimeError("Reproduced submission is not byte-identical")
        environment_result = json.loads(steps[2]["output"])
        tests_passed = parse_test_count(steps[1]["output"])
        with open(
            snapshot / "outputs" / "submission_v2.manifest.json", encoding="utf-8"
        ) as manifest_file:
            delivery = json.load(manifest_file)
        payload = {
            "status": "PASS",
            "source_revision": revision,
            "artifact_revision": delivery["oof_code_revision"],
            "python": str(python),
            "python_version": environment_result["python"],
            "lock_sha256": sha256_file(PROJECT_ROOT / "requirements.lock"),
            "locked_packages": len(environment_result["packages"]),
            "tests_passed": tests_passed,
            "network_access": "disabled",
            "asset_materialization": materialization,
            "accepted_submission_sha256": accepted_hash,
            "reproduced_submission_sha256": reproduced_hash,
            "byte_identical": True,
            "submission_rows": delivery["submission"]["rows"],
            "positive_rows": delivery["submission"]["positive_rows"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "steps": [
                {key: value for key, value in step.items() if key != "output"}
                for step in steps
            ],
        }
        write_reports(payload, args.report, args.json_report)
        print(
            f"status=PASS revision={revision} tests={tests_passed} "
            f"submission_sha256={reproduced_hash} report={args.report}"
        )
        return payload
    finally:
        if args.keep_workdir:
            print(f"workdir={workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
