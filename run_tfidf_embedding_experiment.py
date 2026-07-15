#!/usr/bin/env python
"""Compatibility entry point for the verified embedding ablation.

Embedding experiments require complete, hash-manifested term and item
matrices. Missing model artifacts fail explicitly instead of silently changing
the feature contract.
"""

import sys

from scripts.embedding.run_embedding_score_comparison import main


if __name__ == "__main__":
    main(sys.argv[1:])
