import unittest

import numpy as np

from src.bm25_hard_negative import BM25Index


class Bm25IndexTests(unittest.TestCase):
    def test_ranks_relevant_documents_with_compact_postings(self):
        index = BM25Index(
            np.array(["i1", "i2", "i3"]),
            ["nike koşu ayakkabı", "nike tişört", "mutfak tencere"],
            max_df_ratio=1.0,
        )
        result = index.top_n("nike koşu", n=2)
        self.assertEqual(result[0], "i1")
        documents, frequencies = index.inverted_index["nike"]
        self.assertEqual(documents.dtype, np.uint32)
        self.assertEqual(frequencies.dtype, np.uint32)

    def test_filters_high_document_frequency_postings(self):
        index = BM25Index(
            np.array(["i1", "i2", "i3", "i4"]),
            ["ortak a", "ortak b", "ortak c", "ortak d"],
            max_df_ratio=0.5,
        )
        self.assertNotIn("ortak", index.inverted_index)
        self.assertEqual(len(index.top_n("ortak", n=2)), 0)

    def test_validates_parameters_and_empty_documents(self):
        with self.assertRaisesRegex(ValueError, "max_df_ratio"):
            BM25Index(np.array(["i1"]), ["text"], max_df_ratio=0)
        index = BM25Index(np.array(["i1"]), [""], max_df_ratio=1.0)
        self.assertEqual(len(index.top_n("query", n=1)), 0)


if __name__ == "__main__":
    unittest.main()
