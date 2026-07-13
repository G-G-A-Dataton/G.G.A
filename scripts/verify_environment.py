"""Verify the pinned Python environment without importing heavy packages."""

import argparse
import importlib.metadata
import json
import os
import platform
import re
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expected_requirements(path):
    requirements = {}
    with open(path, encoding="utf-8") as requirements_file:
        for raw_line in requirements_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line)
            if not match:
                raise ValueError(f"Requirement must be exactly pinned: {line}")
            requirements[match.group(1)] = match.group(2)
    return requirements


def verify_environment(requirements_path, config_path, require_embedding_model=False, ignore_mismatch=False):
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    errors = []
    expected_python = config["environment"]["python"]
    actual_python = platform.python_version()
    if actual_python != expected_python:
        errors.append(f"Python mismatch: {actual_python} != {expected_python}")
    installed = {}
    for package, expected_version in expected_requirements(requirements_path).items():
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
    return {"python": actual_python, "packages": installed}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify pinned project environment")
    parser.add_argument("--require-embedding-model", action="store_true")
    parser.add_argument("--ignore-mismatch", action="store_true")
    args = parser.parse_args(argv)
    result = verify_environment(
        os.path.join(PROJECT_ROOT, "requirements.txt"),
        os.path.join(PROJECT_ROOT, "configs", "final_v1.json"),
        require_embedding_model=args.require_embedding_model,
        ignore_mismatch=args.ignore_mismatch,
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
