"""Compatibility entry point for verified cross-fitted threshold analysis."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.analysis.run_threshold_analysis import main


if __name__ == "__main__":
    main()
