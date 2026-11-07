"""Numerical and protocol checks for the supplied experiment implementation."""

import contextlib
import json
import os
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyspiel
from open_spiel.python.algorithms import exploitability

from psro_v2.tabular_eval import TabularEvaluator
from psro_v2.meta_strategies import VDOMetaSolver
from vdo import experiment, matrix, runs

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def test_matrix_equilibrium_metric(self):
        payoffs = np.array([[1.0, -1.0], [-1.0, 1.0]])
        self.assertAlmostEqual(matrix.restricted_ne(payoffs, [[0, 1], [0, 1]]), 0.0)
        self.assertAlmostEqual(matrix.restricted_ne(payoffs, [[0], [0]]), 1.0)

    def test_saved_matrix_trajectories(self):
        for mode in ["odo", "vdo", "dvdo_b5", "fp_hardbr", "fixed_lambda10"]:
            with self.subTest(mode=mode):
                full_events = 3000 if mode == "fixed_lambda10" else 200
                size = 50 if mode == "fixed_lambda10" else 25
                saved = runs.load_json(
                    ROOT
                    / "data/supplement"
                    / matrix.filename((size, mode, 1, full_events))
                )
                fresh = matrix.run_spec((size, mode, 1, 12))
                self.assertEqual(fresh["raw_matrix_sha256"], saved["raw_matrix_sha256"])
                for a, b in zip(fresh["points"], saved["points"]):
                    np.testing.assert_allclose(
                        [a["online"], a["rne"]],
                        [b["online"], b["rne"]],
                        atol=1e-10,
                        rtol=0,
                    )
                    self.assertEqual(a["learning_rounds"], b["learning_rounds"])

    def test_rediscovery_has_no_population_effect(self):
        payoffs = np.random.default_rng(123).uniform(-1, 1, (8, 7))
        for mode in ["odo", "vdo", "dvdo_b5", "fp_hardbr"]:
            a = matrix.run(payoffs, mode, 1, 20)
            b = matrix.run(payoffs, mode, 1, 20, keep_duplicates=True)
            for x, y in zip(a["points"], b["points"]):
                np.testing.assert_allclose(
                    [x["online"], x["rne"]], [y["online"], y["rne"]], atol=1e-10, rtol=0
                )
                self.assertEqual(x["learning_rounds"], y["learning_rounds"])

    def test_per_player_reset(self):
        class Observed(VDOMetaSolver):
            resets = []

            def _reinitialise(self, payoffs, average, players):
                before = [a.copy() for a in self._log_weights]
                super()._reinitialise(payoffs, average, players)
                for p in set(range(2)) - set(players):
                    np.testing.assert_array_equal(before[p], self._log_weights[p])
                self.resets.append(list(players))

        game = matrix.MatrixGame(np.array([[1.0, -1.0], [-1.0, 1.0]]), 1)
        solver = Observed(strategy_key_fn=lambda p, a: a, per_player_windows=True)
        solver(game)
        game.ids[0].append(1 - game.ids[0][0])
        solver(game)
        self.assertEqual(solver.resets[-1], [0])

    def test_endpoint_recording_preserves_learning(self):
        for oracle in ["exact", "eps", "sampled"]:
            cfg = runs.load_json(ROOT / f"configs/psro/{oracle}/kuhn.json")["global"]
            cfg["max_br"] = 12
            method = {
                "meta_strategy_method": "vdo",
                "vdo_beta": 5,
                "vdo_eta": 1 / 11,
                "vdo_window_scale": 50,
                "normalize_payoffs": True,
            }
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                full = experiment.run_single(cfg, method, 1)[-1]
                end = experiment.run_single(cfg, method, 1, endpoint_only=True)[-1]
            for key in ["exploitability", "rne_exploitability"]:
                self.assertAlmostEqual(full[key], end[key], places=11)

    def test_realisation_equivalent_aggregation(self):
        game = pyspiel.load_game("kuhn_poker")
        evaluator = TabularEvaluator(game)
        rng = np.random.RandomState(3)
        policies = [
            [experiment._random_pure_policy(game, rng) for _ in range(3)]
            for _ in range(2)
        ]
        weights = [np.array([0.2, 0.3, 0.5]), np.array([0.4, 0.5, 0.1])]
        aggregate = evaluator.aggregate(game, policies, weights)
        expected = sum(
            weights[0][i] * weights[1][j] * evaluator.expected_returns([p, q])
            for i, p in enumerate(policies[0])
            for j, q in enumerate(policies[1])
        )
        np.testing.assert_allclose(
            evaluator.expected_returns([aggregate, aggregate]), expected, atol=1e-12
        )
        self.assertAlmostEqual(
            evaluator.nash_conv(policies, weights),
            exploitability.nash_conv(game, aggregate),
            places=11,
        )

    def test_all_psro_seed_aggregates(self):
        manifest = runs.load_json(ROOT / "data/psro_manifest.json")
        count = 0
        for dataset in manifest["datasets"]:
            aggregate = runs.load_json(ROOT / dataset["aggregate"])
            for method, entry in aggregate.items():
                if not isinstance(entry, dict):
                    continue
                files = [p for p in dataset["checkpoints"] if p["method"] == method]
                self.assertEqual(len(files), entry["num_runs"])
                self.assertEqual(
                    sorted(p["seed"] for p in files), list(range(1, len(files) + 1))
                )
                samples = [runs.load_json(ROOT / p["file"]) for p in files]
                count += len(samples)
                for sample in samples:
                    self.assertEqual(
                        [r["br_count"] for r in sample], list(range(1, 201))
                    )
                for metric in ["exploitability", "rne_exploitability"]:
                    values = np.array(
                        [[row[metric] for row in sample] for sample in samples]
                    )
                    rows = entry["results_by_br"]
                    np.testing.assert_allclose(
                        values.mean(0),
                        [r["avg_" + metric] for r in rows],
                        atol=1e-13,
                        rtol=0,
                    )
                    np.testing.assert_allclose(
                        values.std(0, ddof=0),
                        [r["std_" + metric] for r in rows],
                        atol=1e-13,
                        rtol=0,
                    )
        self.assertEqual(count, 885)

    def test_table_matches_full_psro_records(self):
        endpoints = runs.load_json(ROOT / "data/table2_endpoints.json")
        manifest = runs.load_json(ROOT / "data/psro_manifest.json")
        for dataset in manifest["datasets"]:
            aggregate = runs.load_json(ROOT / dataset["aggregate"])
            for method, entry in aggregate.items():
                if not isinstance(entry, dict):
                    continue
                last = entry["results_by_br"][-1]
                expected = [
                    entry["num_runs"],
                    last["br_count"],
                    last["avg_exploitability"],
                    last["avg_rne_exploitability"],
                ]
                self.assertEqual(
                    endpoints[dataset["regime"]][dataset["game"]][method], expected
                )

    def test_checkpoint_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.json"
            runs.cached_run(path, {"eta": 0.1}, lambda: {"points": []})
            with self.assertRaises(ValueError):
                runs.cached_run(path, {"eta": 0.2}, lambda: {"points": []})

    def test_fresh_results_feed_table_and_goofspiel_panel(self):
        from vdo import figures, table

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for dataset in runs.load_json(ROOT / "data/psro_manifest.json")["datasets"]:
                target = root / dataset["regime"] / dataset["game"] / "all_results.json"
                target.parent.mkdir(parents=True)
                shutil.copyfile(ROOT / dataset["aggregate"], target)
            source = runs.psro_result_path(root, "exact", "goofspiel4")
            changed = runs.load_json(source)
            changed["vdo_beta_50"]["results_by_br"][-1]["avg_exploitability"] = 0.2468
            runs.write_json(source, changed)
            rows = table.generate(root / "figures", root)
            row = next(
                r
                for r in rows
                if (r["game"], r["oracle"], r["method"], r["metric"])
                == ("goofspiel4", "exact", "vdo_beta_50", "online")
            )
            self.assertEqual(float(row["normalised_half_nashconv"]), 0.1234)
            with patch.object(figures, "PSRO_DATA", root):
                figures.reported_goofspiel.cache_clear()
                self.assertEqual(figures.endpoint("goof4", "dvdo_b50")[0], 0.1234)
            figures.reported_goofspiel.cache_clear()
            figures.SUMMARY.clear()


if __name__ == "__main__":
    unittest.main()
