import unittest
from unittest import mock

from scripts import run_production


class ProductionRunnerTests(unittest.TestCase):
    def test_default_training_uses_dual_model_shortlist(self):
        with mock.patch.object(run_production, "run") as runner:
            run_production.main(["--stage", "train"])
        command = runner.call_args.args[0]
        self.assertEqual(command[1], "scripts/training/run_model_shortlist.py")

    def test_default_prediction_uses_cross_fitted_selection(self):
        with mock.patch.object(run_production, "run") as runner:
            run_production.main(["--stage", "predict"])
        command = runner.call_args.args[0]
        self.assertEqual(
            command[1], "scripts/analysis/run_ensemble_optimization.py"
        )
        self.assertIn("outputs/submission_v2.csv", command)

    def test_lightgbm_fallback_is_explicit(self):
        with mock.patch.object(run_production, "run") as runner:
            run_production.main(
                ["--stage", "predict", "--pipeline", "lightgbm"]
            )
        self.assertEqual(
            runner.call_args.args[0][1], "scripts/submission/run_pipeline.py"
        )


if __name__ == "__main__":
    unittest.main()
