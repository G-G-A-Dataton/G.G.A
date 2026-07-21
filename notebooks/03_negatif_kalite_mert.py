#!/usr/bin/env python
"""Compatibility entry point for the canonical frozen-data quality gates.

Candidate-level leakage and quota assertions live in `src.candidate_sampling`
and are covered by the regression suite. This entry point validates the frozen
source files and all merge/submission relationships without import-time work.
"""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.data.verify_data_freeze import main as verify_freeze  # noqa: E402
from scripts.data.verify_pipeline import main as verify_pipeline  # noqa: E402


def main():
    verify_freeze()
    return verify_pipeline()


if __name__ == "__main__":
    main()
