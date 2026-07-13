"""Compatibility entry point for the leakage-free grouped lexical baseline."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from scripts.analysis.run_deney_matrisi_v2 import main


if __name__ == "__main__":
    main(
        [
            "--ratios",
            "3",
            "--feature-sets",
            "lexical_core",
            "--output",
            "outputs/baseline_grouped.csv",
            "--report",
            "docs/baseline_grouped.md",
            *sys.argv[1:],
        ]
    )
