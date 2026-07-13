"""Compatibility entry point for verified term embedding production."""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.embedding_batch import main


if __name__ == "__main__":
    main(["--target", "terms", *sys.argv[1:]])
