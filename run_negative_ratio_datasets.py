#!/usr/bin/env python
"""Compatibility entry point for grouped negative-ratio ablations.

The canonical experiment stores compact result evidence instead of persisting
several multi-gigabyte intermediate CSV files.
"""

import sys

from scripts.analysis.run_deney_matrisi_v2 import main


if __name__ == "__main__":
    main(["--ratios", "1", "2", "3", "5", *sys.argv[1:]])
