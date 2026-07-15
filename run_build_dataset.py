#!/usr/bin/env python
"""Compatibility entry point for manifest-backed training-set generation."""

import sys

from scripts.data.run_build_training_dataset import main


if __name__ == "__main__":
    main(sys.argv[1:])
