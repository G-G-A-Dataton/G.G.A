"""Backward-compatible CLI wrapper for full v2 submission generation."""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.inference import main


if __name__ == "__main__":
    main()
