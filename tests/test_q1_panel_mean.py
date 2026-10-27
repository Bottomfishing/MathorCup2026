"""Unit tests for beer_models.q1_panel_mean (task-9, task-15 leak-free baseline)."""

import json
import re
import unittest
from pathlib import Path

import numpy as np

from beer_models.q1_panel_mean import (
    absolute_errors,
    exact_rate,
    gate_verdicts,
    macro_spearman,
    paired_beer_bootstrap,
    panel_mean_from_distribution,
    panel_mean_metrics,
    panel_median_from_distribution,
    per_beer_mean_error,
    round_grades,
    target_family,
    within_one_rate,
)


class TruthTest(unittest.TestCase):
    def test_panel_mean_from_distribution(self):
        distribution = np.zeros((2, 3, 7))
        distribution[0, 0, 0] = 1.0
        distribution[0, 1, 6] = 1.0
        distribution[0, 2, 2] = 1.0
        distribution[1, 0, 3] = 1.0
        distribution[1, 1, 2] = 0.5
        distribution[1, 1, 4] = 0.5
        distribution[1, 2, 1] = 1.0
        means = panel_mean_from_distribution(distribution)
        np.testing.assert_allclose(means[0], [0.0, 6.0, 2.0])
        self.assertAlmostEqual(float(means[1, 1]), 3.0)
        self.assertAlmostEqual(float(means[1, 2]), 1.0)

    def test_panel_median_is_lower_median(self):
        distribution = np.zeros((1, 1, 7))
        distribution[0, 0, 2] = 0.5
        distribution[0, 0, 4] = 0.5
        self.assertEqual(float(panel_median_from_distribution(distribution)[0, 0]), 2.0)

    def test_distribution_validation(self):
        with self.assertRaises(ValueError):
            panel_mean_from_distribution(np.zeros((1, 6)))
        with self.assertRaises(ValueError):
            panel_mean_from_distribution(np.full((1, 7), 1.0))
        with self.assertRaises(ValueError):
            panel_mean_from_distribution(np.full((1, 7), -1.0))


class RoundingTest(unittest.TestCase):
    def test_bankers_rounding_and_clip(self):
        values = np.array([-1.0, 0.4, 0.5, 1.5, 2.5, 3.5, 6.4, 7.0])
        np.testing.assert_array_equal(round_grades(values), [0, 0, 0, 2, 2, 4, 6, 6])

    def test_round_grades_requires_array(self):
        with self.assertRaises(ValueError):
            round_grades(3.0)


class ErrorTest(unittest.TestCase):
    def test_absolute_and_per_beer_error(self):
        predictions = np.array([[0.0, 2.0], [4.0, 1.0]])
        truth = np.array([[1.0, 2.0], [4.0, 3.0]])
        np.testing.assert_allclose(absolute_errors(predictions, truth), [[1, 0], [0, 2]])
        np.testing.assert_allclose(per_beer_mean_error(predictions, truth), [0.5, 1.0])

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            absolute_errors(np.zeros(3), np.zeros(4))


class BootstrapTest(unittest.TestCase):
    def test_positive_constant_difference(self):
        differences = np.full(20, 0.1)
        styles = np.array(["lager"] * 10 + ["ale"] * 10)
        result = paired_beer_bootstrap(differences, styles, n_boot=200, seed=7)
        self.assertAlmostEqual(result["point"], 0.1)
        self.assertGreater(result["lower"], 0.0)
        self.assertLessEqual(result["lower"], result["upper"])

    def test_zero_difference_interval_contains_zero(self):
        differences = np.zeros(20)
        styles = np.array(["lager"] * 10 + ["ale"] * 10)
        result = paired_beer_bootstrap(differences, styles, n_boot=200, seed=7)
        self.assertEqual(result["point"], 0.0)
        self.assertLessEqual(result["lower"], 0.0)
        self.assertGreaterEqual(result["upper"], 0.0)

    def test_deterministic_and_validates(self):
        differences = np.linspace(-0.2, 0.2, 20)
        styles = np.array(["a"] * 10 + ["b"] * 10)
        first = paired_beer_bootstrap(differences, styles, n_boot=100, seed=11)
        second = paired_beer_bootstrap(differences, styles, n_boot=100, seed=11)
        self.assertEqual(first, second)
        with self.assertRaises(ValueError):
            paired_beer_bootstrap(differences, styles[:5], n_boot=10, seed=1)


class CorrelationTest(unittest.TestCase):
    def test_macro_spearman_perfect_and_constant_column(self):
        truth = np.arange(6, dtype=float).reshape(6, 1)
        predictions = np.column_stack([truth[:, 0], np.ones(6)])
        truth = np.column_stack([truth[:, 0], np.arange(6, dtype=float)])
        rho, count = macro_spearman(predictions, truth)
        self.assertAlmostEqual(rho, 1.0)
        self.assertEqual(count, 1)

    def test_macro_spearman_negative(self):
        truth = np.arange(6, dtype=float).reshape(6, 1)
        predictions = -truth
        rho, count = macro_spearman(predictions, truth)
        self.assertAlmostEqual(rho, -1.0)
        self.assertEqual(count, 1)


class MetricBundleTest(unittest.TestCase):
    def test_rates_and_bundle(self):
        predicted_grades = np.array([0, 2, 3])
        median = np.array([0, 1, 5])
        self.assertAlmostEqual(exact_rate(predicted_grades, median), 1 / 3)
        self.assertAlmostEqual(within_one_rate(predicted_grades, median), 2 / 3)
        predictions = np.array([[0.0, 2.0], [5.0, 6.0]])
        truth_mean = np.array([[0.0, 2.0], [5.0, 6.0]])
        truth_median = np.array([[0, 2], [5, 6]])
        metrics = panel_mean_metrics(predictions, truth_mean, truth_median, truth_mean)
        self.assertAlmostEqual(metrics["panel_mean_mae"], 0.0)
        self.assertAlmostEqual(metrics["panel_mean_rmse"], 0.0)
        self.assertAlmostEqual(metrics["panel_median_exact"], 1.0)
        self.assertAlmostEqual(metrics["panel_median_within_one"], 1.0)
        self.assertAlmostEqual(metrics["pearson_with_calibrated_baseline"], 1.0)


class GateTest(unittest.TestCase):
    def test_all_gates_pass(self):
        verdict = gate_verdicts(
            pooled_mae=0.44,
            repeat_maes=[0.44, 0.45, 0.43],
            bootstrap_lower=0.005,
            pooled_within_one=0.92,
            baseline_mae=0.4622,
        )
        self.assertTrue(verdict["all_pass"])
        self.assertTrue(all(verdict["gates"].values()))

    def test_any_gate_fails(self):
        verdict = gate_verdicts(
            pooled_mae=0.455,
            repeat_maes=[0.44, 0.47, 0.45],
            bootstrap_lower=-0.001,
            pooled_within_one=0.90,
            baseline_mae=0.4622,
        )
        self.assertFalse(verdict["all_pass"])
        self.assertFalse(verdict["gates"]["gate_1_point_improvement"])
        self.assertFalse(verdict["gates"]["gate_2_bootstrap"])
        self.assertFalse(verdict["gates"]["gate_3_every_repeat"])
        self.assertFalse(verdict["gates"]["gate_4_within_one"])

    def test_empty_repeats_raise(self):
        with self.assertRaises(ValueError):
            gate_verdicts(0.4, [], 0.1, 0.9, 0.46)


class FamilyTest(unittest.TestCase):
    def test_target_family(self):
        self.assertEqual(target_family("A_malt_all"), "Aroma")
        self.assertEqual(target_family("F_hops_all"), "Flavour")
        self.assertEqual(target_family("body"), "Supplementary")


class LeakFreeBaselineProtocolTest(unittest.TestCase):
    """Task-15: the comparison baseline and gates must use the task-12 nested numbers."""

    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "configs/q1_panel_mean_20261028.json"
        cls.config = json.loads(path.read_text(encoding="utf-8"))

    def test_baseline_values_are_the_nested_ones(self):
        baseline = self.config["reported_baseline"]
        self.assertAlmostEqual(baseline["panel_mean_mae"], 0.46607189992522735, places=12)
        self.assertAlmostEqual(baseline["panel_mean_mae_linear"], 0.5054899408175908, places=12)
        self.assertAlmostEqual(
            baseline["panel_median_within_one_rounded"], 0.91416, places=12
        )
        self.assertAlmostEqual(
            baseline["voided_pre_fix"]["panel_mean_mae"], 0.4622337899166068, places=12
        )

    def test_gates_reference_the_nested_baseline(self):
        gates = self.config["adoption_gates"]
        baseline = self.config["reported_baseline"]["panel_mean_mae"]
        self.assertIn(repr(baseline), gates["gate_1_point_improvement"])
        self.assertIn(repr(baseline), gates["gate_3_every_repeat"])
        self.assertNotIn("0.4622337899166068", gates["gate_1_point_improvement"])
        # gate 4 keeps its absolute pre-registered floor (0.9169 - 0.5 pt); it is
        # not re-anchored to the corrected baseline.
        floor = float(re.search(r">=\s*([0-9.]+)", gates["gate_4_within_one"]).group(1))
        self.assertAlmostEqual(floor, 0.9119, places=12)

    def test_leak_fix_is_declared(self):
        fix = self.config["leak_fix"]
        self.assertEqual(fix["old_baseline_path"].split(" ")[0], "fold_style_mean")
        self.assertIn("inner OOF cells", fix["new_baseline_path"])
        self.assertIn("fit_nested_linear", fix["reproduction"])

    def test_gate4_keeps_absolute_floor(self):
        # ET-direct's pooled within-one (0.91136) misses the pre-registered
        # absolute floor 0.9119 by 0.00054, but would pass a relative floor
        # anchored to the corrected baseline (0.91416 - 0.005).  Only the
        # absolute floor is used, so the verdict must stay do_not_adopt.
        common = dict(
            pooled_mae=0.4518585187795302,
            repeat_maes=[0.45197225088862586, 0.4512178825142204, 0.4523854229357453],
            bootstrap_lower=0.008208117710517127,
            pooled_within_one=0.91136,
            baseline_mae=0.46607189992522735,
            minimum_improvement=0.010,
        )
        absolute = gate_verdicts(**common, within_one_floor=0.9119)
        relative_only = gate_verdicts(**common, within_one_floor=0.90916)
        self.assertFalse(absolute["gates"]["gate_4_within_one"])
        self.assertFalse(absolute["all_pass"])
        self.assertTrue(relative_only["gates"]["gate_4_within_one"])


if __name__ == "__main__":
    unittest.main()
