import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.embedding_batch import encode_in_chunks
from src.embedding_cosine import EmbeddingIndex, add_embedding_cosine_feature


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        values = []
        for index, text in enumerate(texts):
            vector = np.array([len(text) + 1, index + 1, 1.0], dtype=np.float32)
            values.append(vector / np.linalg.norm(vector))
        return np.asarray(values, dtype=np.float32)


class EmbeddingArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.indices = []

    def tearDown(self):
        for index in self.indices:
            index.close()
        self.temp_dir.cleanup()

    def produce(self, target="term", keep_chunks=False):
        prefix = os.path.join(self.temp_dir.name, target)
        encode_in_chunks(
            FakeEmbeddingModel(),
            ["one", "two words", "three"],
            ["a", "b", "c"],
            prefix,
            chunk_size=2,
            batch_size=1,
            model_name="fake/model",
            target=target,
            keep_chunks=keep_chunks,
        )
        return prefix

    def test_production_writes_pickle_free_ids_and_hash_manifest(self):
        prefix = self.produce()
        ids = np.load(prefix + "_ids.npy", allow_pickle=False)
        self.assertEqual(ids.tolist(), ["a", "b", "c"])
        with open(prefix + "_manifest.json", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        self.assertEqual(manifest["rows"], 3)
        self.assertEqual(manifest["dimension"], 3)
        index = EmbeddingIndex(
            prefix + "_embeddings.npy",
            prefix + "_ids.npy",
            prefix + "_manifest.json",
            expected_target="term",
        )
        self.indices.append(index)
        self.assertEqual(index.get_batch(["c", "a"]).shape, (2, 3))

    def test_unknown_ids_fail_instead_of_returning_zero_vectors(self):
        prefix = self.produce()
        index = EmbeddingIndex(
            prefix + "_embeddings.npy",
            prefix + "_ids.npy",
            prefix + "_manifest.json",
        )
        self.indices.append(index)
        with self.assertRaisesRegex(KeyError, "missing"):
            index.get_batch(["missing"])
        with self.assertRaisesRegex(ValueError, "Both verified"):
            add_embedding_cosine_feature(
                pd.DataFrame({"term_id": ["a"], "item_id": ["a"]}),
                index,
                None,
            )

    def test_stale_checkpoint_metadata_is_rejected(self):
        prefix = self.produce(keep_chunks=True)
        metadata_path = prefix + "_chunk_00000.npy.json"
        with open(metadata_path, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        metadata["model"] = "other/model"
        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file)
        with self.assertRaisesRegex(ValueError, "Stale"):
            encode_in_chunks(
                FakeEmbeddingModel(),
                ["one", "two words", "three"],
                ["a", "b", "c"],
                prefix,
                chunk_size=2,
                batch_size=1,
                model_name="fake/model",
                target="term",
                keep_chunks=True,
            )


if __name__ == "__main__":
    unittest.main()
