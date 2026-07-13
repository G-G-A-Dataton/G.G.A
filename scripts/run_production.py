"""One-command verified training and inference runner."""

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
        "--stage", choices=["verify", "train", "predict", "all"], default="all"
    )
    args = parser.parse_args(argv)
    python = sys.executable
    if args.stage in ("verify", "all"):
        run([python, "-m", "unittest", "discover", "-s", "tests", "-v"])
        run([python, "scripts/verify_environment.py"])
        run([python, "scripts/data/verify_data_freeze.py"])
        run([python, "scripts/data/verify_pipeline.py"])
    if args.stage in ("train", "all"):
        run([python, "scripts/training/run_train_full_v2.py"])
    if args.stage in ("predict", "all"):
        run([python, "scripts/submission/run_pipeline.py", "--mode", "predict"])


if __name__ == "__main__":
    main()
