"""Verify the pinned Python environment without importing heavy packages."""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalize_package_name(name):
    """Return the PEP 503 comparison form for a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def expected_requirements(path):
    requirements = {}
    with open(path, encoding="utf-8") as requirements_file:
        for raw_line in requirements_file:
            line = raw_line.split("#")[0].strip()
            if not line:
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
            if not match:
                raise ValueError(f"Requirement must be exactly pinned: {line}")
            package = normalize_package_name(match.group(1))
            if package in requirements:
                raise ValueError(f"Duplicate direct requirement: {package}")
            requirements[package] = match.group(2)
    return requirements


def expected_lock_requirements(path):
    """Read package pins from a hash-locked pip requirements file."""
    requirements = {}
    with open(path, encoding="utf-8") as lock_file:
        for raw_line in lock_file:
            if raw_line[:1].isspace() or raw_line.lstrip().startswith("#"):
                continue
            match = re.match(r"([A-Za-z0-9_.-]+)==([^\\\s]+)", raw_line)
            if not match:
                continue
            package, version = match.groups()
            package = normalize_package_name(package)
            if package in requirements:
                raise ValueError(f"Duplicate package in lock file: {package}")
            requirements[package] = version
    if not requirements:
        raise ValueError(f"Lock file contains no pinned packages: {path}")
    return requirements


def resolved_requirements(requirements_path, lock_path=None):
    direct = expected_requirements(requirements_path)
    if lock_path is None:
        return direct
    locked = expected_lock_requirements(lock_path)
    mismatches = {
        package: (version, locked.get(package))
        for package, version in direct.items()
        if locked.get(package) != version
    }
    if mismatches:
        raise ValueError(
            "Lock file does not preserve direct requirements: "
            + ", ".join(
                f"{name} {actual!r} != {expected!r}"
                for name, (expected, actual) in sorted(mismatches.items())
            )
        )
    return locked


def verify_environment(
    requirements_path,
    config_path,
    require_embedding_model=False,
    lock_path=None,
    ignore_mismatch=False,
):
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    errors = []
    expected_python = config["environment"]["python"]
    actual_python = platform.python_version()
    if actual_python != expected_python:
        errors.append(f"Python mismatch: {actual_python} != {expected_python}")
    installed = {}
    expected = resolved_requirements(requirements_path, lock_path)
    for package, expected_version in expected.items():
        try:
            actual_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing package: {package}=={expected_version}")
            continue
        installed[package] = actual_version
        if actual_version != expected_version:
            errors.append(
                f"Package mismatch: {package} {actual_version} != {expected_version}"
            )
    if require_embedding_model:
        model_path = os.path.join(
            PROJECT_ROOT, "models", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        if not os.path.isdir(model_path):
            errors.append(f"Missing offline embedding model: {model_path}")
    if errors:
        message = "Environment verification failed: " + "; ".join(errors)
        if ignore_mismatch:
            print(f"\n[UYARI] {message}\n")
        else:
            raise ValueError(message)
    return {
        "python": actual_python,
        "packages": installed,
        "lock_file": os.path.basename(lock_path) if lock_path else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify pinned project environment")
    parser.add_argument("--require-embedding-model", action="store_true")
    parser.add_argument("--ignore-mismatch", action="store_true")
    parser.add_argument(
        "--lock",
        default=None,
        help="Also verify every transitive package pinned in this lock file",
    )
    args = parser.parse_args(argv)
    lock_path = os.path.abspath(args.lock) if args.lock and os.path.exists(args.lock) else None
    result = verify_environment(
        os.path.join(PROJECT_ROOT, "requirements.txt"),
        os.path.join(PROJECT_ROOT, "configs", "final_v1.json"),
        require_embedding_model=args.require_embedding_model,
        lock_path=lock_path,
        ignore_mismatch=args.ignore_mismatch,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
