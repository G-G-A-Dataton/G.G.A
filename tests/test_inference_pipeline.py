"""
tests/test_inference_pipeline.py
=================================
G.G.A Takımı — Inference Pipeline Unit Tests
"""

import os
import pytest
from src.inference.pipeline import InferencePipeline


def test_inference_pipeline_from_config(tmp_path):
    config_file = tmp_path / "test_inf.json"
    config_file.write_text('{"manifest_path": "outputs/model_manifest_v2.json", "batch_size": 50000}')

    pipeline = InferencePipeline.from_config(str(config_file))
    assert pipeline.batch_size == 50000
    assert pipeline.threshold == 0.5
