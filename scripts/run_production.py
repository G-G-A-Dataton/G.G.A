"""One-command verified training, benchmark, ablation and inference runner."""

import argparse
import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(command):
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run verified production workflow")
    parser.add_argument(
        "--stage",
        choices=["verify", "train", "predict", "benchmark", "ablation", "full-e2e", "all"],
        default="all",
        help="Pipeline stage to execute",
    )
    parser.add_argument(
        "--pipeline",
        choices=["shortlist", "lightgbm"],
        default="shortlist",
        help="Final shortlist is the default; lightgbm is a single-model fallback",
    )
    parser.add_argument(
        "--ignore-env-mismatch", action="store_true", help="Ignore environment check mismatches"
    )
    args = parser.parse_args(argv)
    python = sys.executable

    if args.stage in ("verify", "all"):
        run([python, "-m", "pytest", "tests/", "-v"])
        verify_cmd = [python, "scripts/verify_environment.py"]
        if args.ignore_env_mismatch:
            verify_cmd.append("--ignore-mismatch")
        else:
            lock_path = os.path.join(PROJECT_ROOT, "requirements.lock")
            if os.path.exists(lock_path):
                verify_cmd.extend(["--lock", "requirements.lock"])
        run(verify_cmd)
        run([python, "scripts/data/verify_data_freeze.py"])
        run([python, "scripts/data/verify_pipeline.py"])
        run([python, "scripts/data/run_data_quality_report.py"])

    if args.stage in ("benchmark", "full-e2e"):
        run([python, "scripts/training/run_hybrid_reranker_benchmark.py"])

    if args.stage in ("ablation", "full-e2e"):
        run([python, "scripts/analysis/run_ablation_matrix.py"])

    if args.stage in ("train", "all"):
        training_script = (
            "scripts/training/run_model_shortlist.py"
            if args.pipeline == "shortlist"
            else "scripts/training/run_train_full_v2.py"
        )
        run([python, training_script])

    if args.stage in ("predict", "all"):
        if args.pipeline == "shortlist":
            run(
                [
                    python,
                    "scripts/analysis/run_ensemble_optimization.py",
                    "--output",
                    "outputs/submission_v2.csv",
                    "--report",
                    "docs/ensemble_selection.md",
                ]
            )
        else:
            run([python, "pipeline/inference.py", "--mode", "predict"])


if __name__ == "__main__":
    main()
