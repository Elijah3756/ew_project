import pathlib
import sys
import unittest

import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tune_train  # noqa: E402


class TuneTrainHelpersTest(unittest.TestCase):
    def test_build_trial_matrix_includes_weight_decay_dimension(self):
        trials = tune_train.build_trial_matrix(
            lrs=[1e-3, 1e-4],
            batch_sizes=[128],
            epochs_list=[40, 60],
            weight_decays=[0.0, 1e-4],
        )

        self.assertEqual(len(trials), 8)
        self.assertEqual(
            {(t["lr"], t["batch_size"], t["epochs"], t["weight_decay"]) for t in trials},
            {
                (1e-3, 128, 40, 0.0),
                (1e-3, 128, 40, 1e-4),
                (1e-3, 128, 60, 0.0),
                (1e-3, 128, 60, 1e-4),
                (1e-4, 128, 40, 0.0),
                (1e-4, 128, 40, 1e-4),
                (1e-4, 128, 60, 0.0),
                (1e-4, 128, 60, 1e-4),
            },
        )

    def test_rank_top_results_filters_failures_and_sorts_by_best_val(self):
        df = pd.DataFrame(
            [
                {"lr": 1e-3, "batch_size": 128, "epochs": 60, "weight_decay": 1e-4, "best_val_acc": 0.71, "status": "ok"},
                {"lr": 1e-4, "batch_size": 256, "epochs": 80, "weight_decay": 1e-4, "best_val_acc": 0.83, "status": "ok"},
                {"lr": 5e-4, "batch_size": 128, "epochs": 40, "weight_decay": 0.0, "best_val_acc": None, "status": "failed_exit_1"},
            ]
        )

        ranked = tune_train.rank_top_results(df, top_k=2)

        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0]["lr"], 1e-4)
        self.assertEqual(ranked[1]["lr"], 1e-3)

    def test_build_refined_candidates_expands_and_deduplicates(self):
        top_rows = [
            {"lr": 1e-3, "batch_size": 128, "epochs": 60, "weight_decay": 1e-4},
            {"lr": 1e-3, "batch_size": 256, "epochs": 80, "weight_decay": 1e-4},
        ]

        refined = tune_train.build_refined_candidates(
            top_rows,
            lr_factors=[0.5, 1.0, 2.0],
            batch_multipliers=[0.5, 1.0, 2.0],
            epoch_offsets=[-20, 0, 20],
            weight_decay_factors=[0.1, 1.0, 10.0],
        )

        self.assertEqual(refined["lr"], [5e-4, 1e-3, 2e-3])
        self.assertEqual(refined["batch_size"], [64, 128, 256, 512])
        self.assertEqual(refined["epochs"], [40, 60, 80, 100])
        self.assertEqual(refined["weight_decay"], [1e-5, 1e-4, 1e-3])


class DefenseAndWorkflowShapeTest(unittest.TestCase):
    def test_tune_defenses_module_exposes_shared_and_specific_grid_builder(self):
        import tune_defenses  # noqa: E402

        grids = tune_defenses.build_defense_search_spaces(
            shared={
                "lr": [1e-3],
                "batch_size": [128],
                "epochs": [60],
                "weight_decay": [1e-4],
            },
            rho_values=[0.005, 0.01],
            pgd_steps_values=[3, 5],
            sigma_values=[0.01, 0.02],
        )

        self.assertIn("adv_train", grids)
        self.assertIn("noise_inject", grids)
        self.assertTrue(any(cfg["rho"] == 0.005 for cfg in grids["adv_train"]))
        self.assertTrue(any(cfg["sigma"] == 0.02 for cfg in grids["noise_inject"]))

    def test_tune_defenses_best_results_by_defense_keeps_one_winner_per_defense(self):
        import tune_defenses  # noqa: E402

        df = pd.DataFrame(
            [
                {"defense": "adv_train", "best_val_acc": 0.70, "status": "ok", "rho": 0.01},
                {"defense": "adv_train", "best_val_acc": 0.75, "status": "ok", "rho": 0.02},
                {"defense": "noise_inject", "best_val_acc": 0.65, "status": "ok", "sigma": 0.01},
                {"defense": "noise_inject", "best_val_acc": 0.64, "status": "ok", "sigma": 0.02},
            ]
        )

        best = tune_defenses.best_results_by_defense(df)

        self.assertEqual(best["adv_train"]["rho"], 0.02)
        self.assertEqual(best["noise_inject"]["sigma"], 0.01)

    def test_run_tuning_workflow_exposes_expected_stage_sequence(self):
        import run_tuning_workflow  # noqa: E402

        stages = run_tuning_workflow.build_stage_sequence()
        self.assertEqual(
            [stage["name"] for stage in stages],
            [
                "baseline2016_coarse",
                "baseline2016_refine",
                "baseline2016_weight_decay",
                "baseline2018_coarse",
                "baseline2018_refine",
                "defense_tuning",
            ],
        )


class TrainCliTest(unittest.TestCase):
    def test_train_parser_supports_skip_snr_sweep(self):
        import train  # noqa: E402

        parser = train.build_arg_parser()
        args = parser.parse_args(["--skip_snr_sweep"])

        self.assertTrue(args.skip_snr_sweep)


if __name__ == "__main__":
    unittest.main()
