import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.context_features import CONTEXT_FEATURE_COLS, add_context_features
from src.modeling import BASE_MODEL_FEATURE_COLS, MODEL_FEATURE_COLS
from src.out_of_core_features import (
    build_context_feature_store,
    load_feature_batch,
)


class OutOfCoreContextTests(unittest.TestCase):
    def test_disk_context_matches_in_memory_context_for_shuffled_groups(self):
        frame = pd.DataFrame(
            {
                "term_id": ["t1", "t2", "t1", "t3", "t2", "t1"],
                "query_title_overlap": [0.9, 0.2, 0.4, 0.1, 0.8, 0.4],
                "query_title_coverage": [1.0, 0.2, 0.5, 0.1, 0.9, 0.5],
                "query_category_overlap": [0.8, 0.1, 0.3, 0.0, 0.7, 0.3],
                "tfidf_cosine": [0.9, 0.2, 0.5, 0.1, 0.8, 0.5],
            }
        )
        expected = add_context_features(frame, copy=True)
        base_values = np.zeros(
            (len(frame), len(BASE_MODEL_FEATURE_COLS)), dtype=np.float32
        )
        for column in (
            "query_title_overlap",
            "query_title_coverage",
            "query_category_overlap",
            "tfidf_cosine",
        ):
            base_values[:, BASE_MODEL_FEATURE_COLS.index(column)] = frame[column]
        codes = pd.factorize(frame["term_id"], sort=False)[0].astype(np.int32)

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = os.path.join(temp_dir, "base.npy")
            codes_path = os.path.join(temp_dir, "codes.npy")
            np.save(base_path, base_values, allow_pickle=False)
            np.save(codes_path, codes, allow_pickle=False)
            context_path = build_context_feature_store(
                base_path, codes_path, os.path.join(temp_dir, "features")
            )
            base_store = np.load(base_path, mmap_mode="r")
            context_store = np.load(context_path, mmap_mode="r")
            actual = load_feature_batch(base_store, context_store, 0, len(frame))
            if hasattr(base_store, "base") and base_store.base is not None:
                base_store.base.close()
            if hasattr(context_store, "base") and context_store.base is not None:
                context_store.base.close()

        self.assertEqual(actual.columns.tolist(), MODEL_FEATURE_COLS)
        np.testing.assert_allclose(
            actual[CONTEXT_FEATURE_COLS].to_numpy(),
            expected[CONTEXT_FEATURE_COLS].to_numpy(),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
