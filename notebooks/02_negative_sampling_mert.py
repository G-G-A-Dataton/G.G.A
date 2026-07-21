#!/usr/bin/env python
"""Compatibility entry point for grouped negative-ratio ablations.

Legacy fixed-ratio files are not production candidates. This command retains
the historical 1:1/3:1/5:1 experiment surface while using complete query groups,
complete-positive exclusion, and cross-fitted threshold evaluation.
"""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.analysis.run_deney_matrisi_v2 import main as run_matrix  # noqa: E402


def main(argv=None):
    extra_args = sys.argv[1:] if argv is None else argv
    return run_matrix(["--ratios", "1", "3", "5", *extra_args])


if __name__ == "__main__":
    main()
