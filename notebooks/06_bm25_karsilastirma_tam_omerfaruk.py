#!/usr/bin/env python
"""Compatibility entry point for candidate-distribution shift analysis.

The former script generated a fixed 3:1 BM25 data set at import time. The
maintained comparison selects a BM25 share against unlabeled submission feature
marginals and leaves final promotion to grouped OOF validation.
"""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.analysis.run_candidate_shift_analysis import main as run_analysis  # noqa: E402


def main(argv=None):
    return run_analysis(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
