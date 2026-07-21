#!/usr/bin/env python
"""Compatibility entry point for the verified production workflow.

The historical implementation in this path used row-level validation and
random-only negatives. Both violate the production model-selection contract.
This command now delegates to the canonical grouped shortlist pipeline.
"""

import argparse
import sys

from scripts.run_production import main as run_production


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the verified final model workflow")
    parser.add_argument(
        "--stage", choices=["verify", "train", "predict", "all"], default="all"
    )
    parser.add_argument(
        "--pipeline", choices=["shortlist", "lightgbm"], default="shortlist"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Deprecated compatibility flag; the all/predict stages write output",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_production(["--stage", args.stage, "--pipeline", args.pipeline])


if __name__ == "__main__":
    main(sys.argv[1:])
