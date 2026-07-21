#!/usr/bin/env python
"""Compatibility entry point for the canonical data-quality verifier.

The original exported notebook performed full CSV reads at import time and
reported diagnostics without failing every invalid contract. The maintained
verifier is import-safe and fails closed on schema, reference, label, row-count,
and submission-order violations.
"""

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.data.verify_pipeline import main as verify_pipeline  # noqa: E402


def main():
    return verify_pipeline()


if __name__ == "__main__":
    main()
