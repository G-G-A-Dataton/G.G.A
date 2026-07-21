#!/usr/bin/env python
"""Compatibility entry point for manifest-backed candidate generation.

The production builder combines compact BM25, category-hard, and deterministic
random sources under exact test-shaped per-query quotas. It replaces the legacy
unmanifested BM25-only CSV export.
"""

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.data.run_build_training_dataset import main as build_candidates  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build hash-manifested production candidate data"
    )
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    delegated = []
    if args.sample is not None:
        if args.sample <= 0:
            raise ValueError("--sample must be positive")
        delegated.extend(["--sample-terms", str(args.sample)])
    output = args.output
    if output is None and args.sample is not None:
        output = os.path.join(
            PROJECT_ROOT, "outputs", f"training_candidates_sample_{args.sample}.csv"
        )
    if output is not None:
        delegated.extend(["--output", output])
    return build_candidates(delegated)


if __name__ == "__main__":
    main()
