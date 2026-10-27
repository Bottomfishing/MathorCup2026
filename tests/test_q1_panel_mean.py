"""Unit tests for beer_models.q1_panel_mean (task-9)."""

import unittest

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


if __name__ == "__main__":
    unittest.main()
