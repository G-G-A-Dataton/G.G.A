import contextlib
import importlib
import io
import unittest
from unittest import mock

import pandas as pd

import run_final_model
from scripts.data import run_build_training_dataset
from src import attribute_features, negative_mix


class CompatibilityEntrypointTests(unittest.TestCase):
    def test_root_entrypoints_are_import_safe(self):
        modules = (
            "run_build_dataset",
            "run_final_model",
            "run_negative_ratio_datasets",
            "run_tfidf_embedding_experiment",
        )
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            for module_name in modules:
                module = importlib.import_module(module_name)
                self.assertTrue(callable(module.main))
        self.assertEqual(captured.getvalue(), "")

    def test_final_entrypoint_delegates_to_verified_shortlist(self):
        with mock.patch.object(run_final_model, "run_production") as runner:
            run_final_model.main(["--submit"])
        runner.assert_called_once_with(
            ["--stage", "all", "--pipeline", "shortlist"]
        )

    def test_data_builder_rejects_unversioned_full_configuration_early(self):
        with mock.patch.object(
            run_build_training_dataset, "_git_revision"
        ) as revision:
            with self.assertRaisesRegex(ValueError, "locked production"):
                run_build_training_dataset.main(
                    ["--bm25-hard-fraction", "0.1"]
                )
        revision.assert_not_called()

    def test_data_builder_restricts_artifacts_to_project_csv_paths(self):
        args = run_build_training_dataset.parse_args(["--output", "/tmp/out.csv"])
        with self.assertRaisesRegex(ValueError, "inside the project"):
            run_build_training_dataset._validate_args(args)

    def test_attribute_facade_uses_canonical_feature_contract(self):
        frame = pd.DataFrame(
            {
                "query": ["siyah deri ayakkabi"],
                "attributes": ["renk: siyah, materyal: deri"],
            }
        )
        result = attribute_features.add_attribute_features(frame, verbose=False)
        self.assertEqual(
            result[attribute_features.ATTRIBUTE_FEATURE_COLS].iloc[0].tolist(),
            [1, 0, 1],
        )

    def test_negative_mix_facade_preserves_canonical_quotas(self):
        canonical = pd.DataFrame(
            {
                "term_id": ["t1", "t1"],
                "item_id": ["i1", "i2"],
                "label": [1, 0],
                "neg_source": ["positive", "random"],
            }
        )
        with mock.patch.object(
            negative_mix, "_build_mixed_training_set", return_value=canonical
        ) as builder:
            negatives = negative_mix.build_mixed_negative_set(
                object(), object(), object(), top_n=75, max_df_ratio=0.2
            )
        self.assertEqual(negatives["item_id"].tolist(), ["i2"])
        self.assertEqual(builder.call_args.kwargs["bm25_top_n"], 75)
        self.assertEqual(builder.call_args.kwargs["bm25_max_df_ratio"], 0.2)


if __name__ == "__main__":
    unittest.main()
