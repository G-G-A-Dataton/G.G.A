"""Compatibility entry point for verified shortlist model comparison."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.analysis.run_ensemble_comparison import main


if __name__ == "__main__":
    main()
