"""
src/features/__init__.py
========================
Re-exports legacy features from src/features.py and new TabularFeatureExtractor.
"""

import os
import sys
import importlib.util

# Load legacy src/features.py file if present
_current_dir = os.path.dirname(os.path.abspath(__file__))
_legacy_path = os.path.join(os.path.dirname(_current_dir), "features.py")

if os.path.exists(_legacy_path):
    spec = importlib.util.spec_from_file_location("_src_features_legacy", _legacy_path)
    if spec and spec.loader:
        _mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mod)
        for _attr in dir(_mod):
            if not _attr.startswith("__"):
                globals()[_attr] = getattr(_mod, _attr)

# Export new 40+ tabular feature extractor
from .feature_extractor import TabularFeatureExtractor
