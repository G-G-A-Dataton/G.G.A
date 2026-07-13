"""CLI wrapper for the canonical inference pipeline."""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.inference import main


if __name__ == "__main__":
    main()
