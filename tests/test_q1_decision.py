"""Unit tests for :mod:`beer_models.q1_decision` on synthetic data.

Run with::

    $env:PYTHONPATH='src'
    .venv/Scripts/python.exe -m unittest tests.test_q1_decision -v
"""
from __future__ import annotations

import unittest

import numpy as np

from beer_models.q1_decision import (
    apply_nested_linear,
    calibrate_panel_mean,
    enforce_parent_child,
    fit_nested_linear,
    fold_style_mean,
    fold_training_mean,
    fold_training_rate,
    high_grade_stats,
    isotonic_calibrate,
    linear_to_grade,
    observed_high_grade_rates,
    quantile_calibrate,
    runtime_decision,
    topk_decision,
)


class TopKDecisionTest(unittest.TestCase):
    def test_zero_training_rate_keeps_capped_middle_grades(self):
        n = 6
        grades = topk_decision(
            p_ge5=np.full(n, 0.9),
            p6=np.full(n, 0.9),
            base_grades=np.array([3.0, 4.0, 5.0, 2.0, 0.0, 1.0]),
            rate_ge5=np.zeros(n),
            rate_6=np.zeros(n),
            group_ids=np.array(['g'] * n, dtype=object),
        )
        np.testing.assert_array_equal(grades, np.array([3, 4, 4, 2, 0, 1]))

    def test_full_training_rate_promotes_everything_to_six(self):
        n = 4
        grades = topk_decision(
            p_ge5=np.array([0.1, 0.2, 0.3, 0.4]),
            p6=np.array([0.1, 0.2, 0.3, 0.4]),
            base_grades=np.zeros(n),
            rate_ge5=np.ones(n),
            rate_6=np.ones(n),
            group_ids=np.array(['g'] * n, dtype=object),
        )
        np.testing.assert_array_equal(grades, np.array([6, 6, 6, 6]))

    def test_grade_five_is_chosen_only_from_remaining_cells(self):
        # k6 = 1 (0.25 * 4), k5 = 1 (0.25 * 4); the cell promoted to 6 must not
        # consume the 5-slot.
        grades = topk_decision(
            p_ge5=np.array([0.9, 0.0, 0.8, 0.1]),
            p6=np.array([0.1, 0.9, 0.2, 0.3]),
            base_grades=np.zeros(4),
            rate_ge5=np.full(4, 0.25),
            rate_6=np.full(4, 0.25),
            group_ids=np.array(['g'] * 4, dtype=object),
        )
        np.testing.assert_array_equal(grades, np.array([5, 6, 0, 0]))

    def test_ties_are_broken_by_ascending_row_position(self):
        grades = topk_decision(
            p_ge5=np.array([0.5, 0.5, 0.5]),
            p6=np.array([0.5, 0.5, 0.5]),
            base_grades=np.zeros(3),
            rate_ge5=np.full(3, 1.0 / 3.0),
            rate_6=np.full(3, 1.0 / 3.0),
            group_ids=np.array(['g'] * 3, dtype=object),
        )
        # k6 = 1 and k5 = 1 for a 3-cell group: grade 6 takes position 0 first,
        # then grade 5 takes the first remaining position (1).
        self.assertEqual(grades.tolist(), [6, 5, 0])

    def test_groups_use_their_own_rates(self):
        # group a: no high grades observed -> nothing promoted; group b: 50% -> 2 cells.
        grades = topk_decision(
            p_ge5=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
            p6=np.zeros(8),
            base_grades=np.ones(8),
            rate_ge5=np.array([0.0] * 4 + [0.5] * 4),
            rate_6=np.zeros(8),
            group_ids=np.array(['a'] * 4 + ['b'] * 4, dtype=object),
        )
        np.testing.assert_array_equal(grades, np.array([1, 1, 1, 1, 5, 5, 1, 1]))

    def test_small_rate_rounds_to_zero_k(self):
        grades = topk_decision(
            p_ge5=np.full(10, 0.99), p6=np.zeros(10), base_grades=np.zeros(10),
            rate_ge5=np.full(10, 0.04), rate_6=np.zeros(10),
            group_ids=np.array(['g'] * 10, dtype=object))
        np.testing.assert_array_equal(grades, np.zeros(10, dtype=int))

    def test_rejects_out_of_range_probabilities(self):
        with self.assertRaises(ValueError):
            topk_decision(np.array([1.2, 0.1]), np.array([0.1, 0.1]),
                          np.zeros(2), np.zeros(2), np.zeros(2),
                          np.array(['g', 'g'], dtype=object))

    def test_rejects_misaligned_lengths(self):
        with self.assertRaises(ValueError):
            topk_decision(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2]),
                          np.zeros(3), np.zeros(3), np.zeros(3),
                          np.array(['g'] * 3, dtype=object))

    def test_deterministic(self):
        kwargs = dict(
            p_ge5=np.array([0.9, 0.8, 0.7, 0.6]),
            p6=np.array([0.4, 0.3, 0.2, 0.1]),
            base_grades=np.array([1.0, 2.0, 3.0, 4.0]),
            rate_ge5=np.full(4, 0.5),
            rate_6=np.full(4, 0.25),
            group_ids=np.array(['g'] * 4, dtype=object))
        first = topk_decision(**kwargs)
        second = topk_decision(**kwargs)
        np.testing.assert_array_equal(first, second)


class FoldRateTest(unittest.TestCase):
    def test_rate_excludes_the_cells_own_fold(self):
        labels = np.array([0, 0, 5, 6, 6, 6], dtype=float)
        folds = np.array([0, 0, 1, 1, 2, 2], dtype=object)
        groups = np.array(['t'] * 6, dtype=object)
        rate = fold_training_rate(labels, groups, folds, 5)
        # fold 0 trains on folds 1+2: [5, 6, 6, 6] -> 1.0
        np.testing.assert_allclose(rate[:2], [1.0, 1.0])
        # fold 1 trains on folds 0+2: [0, 0, 6, 6] -> 0.5
        np.testing.assert_allclose(rate[2:4], [0.5, 0.5])
        # fold 2 trains on folds 0+1: [0, 0, 5, 6] -> 0.5
        np.testing.assert_allclose(rate[4:], [0.5, 0.5])

    def test_mean_and_style_mean_follow_the_training_folds(self):
        values = np.array([1.0, 1.0, 5.0, 7.0], dtype=float)
        folds = np.array([0, 0, 1, 1], dtype=object)
        groups = np.array(['t'] * 4, dtype=object)
        styles = np.array(['stout', 'ale', 'stout', 'ale'], dtype=object)
        np.testing.assert_allclose(fold_training_mean(values, groups, folds),
                                   [6.0, 6.0, 1.0, 1.0])
        style_mean = fold_style_mean(values, styles, groups, folds)
        np.testing.assert_allclose(style_mean, [5.0, 7.0, 1.0, 1.0])

    def test_unseen_style_falls_back_to_training_global_mean(self):
        values = np.array([1.0, 3.0, 5.0, 5.0], dtype=float)
        folds = np.array([0, 0, 1, 1], dtype=object)
        groups = np.array(['t'] * 4, dtype=object)
        styles = np.array(['stout', 'ale', 'lager', 'lager'], dtype=object)
        predicted = fold_style_mean(values, styles, groups, folds)
        np.testing.assert_allclose(predicted[2:], [2.0, 2.0])
        np.testing.assert_allclose(predicted[:2], [5.0, 5.0])

    def test_single_fold_group_is_rejected(self):
        with self.assertRaises(ValueError):
            fold_training_rate(np.array([0.0, 1.0]), np.array(['t', 't'], dtype=object),
                               np.array([0, 0], dtype=object), 1)


class CalibrationTest(unittest.TestCase):
    def _affine_frame(self, groups=2, per_group=4):
        rows = []
        for group in range(groups):
            for fold in (0, 1):
                for index in range(per_group):
                    x = float(index)
                    rows.append((group, fold, x, 3.0 * x + 2.0))
        data = np.array(rows)
        return (data[:, 2], data[:, 3],
                np.array(['g%d' % value for value in data[:, 0]], dtype=object),
                np.array(['f%d' % value for value in data[:, 1]], dtype=object))

    def test_linear_calibration_recovers_affine_map(self):
        x, y, groups, folds = self._affine_frame()
        prediction, coefficients = calibrate_panel_mean(x, y, groups, folds)
        np.testing.assert_allclose(prediction, 3.0 * x + 2.0, atol=1e-9)
        np.testing.assert_allclose(coefficients['slope'].ravel(), 3.0, atol=1e-9)
        np.testing.assert_allclose(coefficients['intercept'], 2.0, atol=1e-9)

    def test_linear_calibration_rank_deficient_falls_back_to_training_mean(self):
        x = np.full(4, 2.0)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        groups = np.array(['g'] * 4, dtype=object)
        folds = np.array(['f0', 'f0', 'f1', 'f1'], dtype=object)
        prediction, coefficients = calibrate_panel_mean(x, y, groups, folds)
        np.testing.assert_allclose(prediction[:2], [3.5, 3.5])
        np.testing.assert_allclose(prediction[2:], [1.5, 1.5])
        np.testing.assert_allclose(coefficients['slope'], 0.0)

    def test_isotonic_calibration_is_monotone_and_clipped(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])
        y = np.array([2.0, 0.0, 1.0, 3.0, 2.0, 0.0, 1.0, 3.0])
        groups = np.array(['g'] * 8, dtype=object)
        folds = np.array(['f0'] * 4 + ['f1'] * 4, dtype=object)
        prediction = isotonic_calibrate(x, y, groups, folds)
        # train on fold 0 (with one violator pair at x=0,1 -> pooled 1.0)
        np.testing.assert_allclose(prediction[4:], [1.0, 1.0, 1.0, 3.0])
        self.assertTrue(np.all(np.diff(prediction[4:]) >= 0))
        self.assertTrue(np.all(prediction >= 0.0))

    def test_isotonic_constant_scores_use_training_mean(self):
        x = np.full(4, 1.0)
        y = np.array([1.0, 3.0, 2.0, 4.0])
        groups = np.array(['g'] * 4, dtype=object)
        folds = np.array(['f0', 'f0', 'f1', 'f1'], dtype=object)
        prediction = isotonic_calibrate(x, y, groups, folds)
        np.testing.assert_allclose(prediction[:2], [3.0, 3.0])
        np.testing.assert_allclose(prediction[2:], [2.0, 2.0])

    def test_quantile_calibration_returns_training_reference_values(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0])
        median = np.array([0.0, 1.0, 2.0, 4.0, 0.0, 1.0, 2.0, 4.0])
        groups = np.array(['g'] * 8, dtype=object)
        folds = np.array(['f0'] * 4 + ['f1'] * 4, dtype=object)
        prediction = quantile_calibrate(x, median, groups, folds)
        np.testing.assert_allclose(prediction[4:], [0.0, 1.0, 1.0, 2.0])
        self.assertTrue(np.all(np.isin(prediction[4:], np.array([0.0, 1.0, 2.0, 4.0]))))

    def test_linear_to_grade_rounds_ties_to_even_and_clips(self):
        np.testing.assert_array_equal(
            linear_to_grade(np.array([-1.0, 0.4, 2.5, 3.5, 7.0])),
            np.array([0, 0, 2, 4, 6]))

    def test_high_grade_stats(self):
        stats = high_grade_stats(np.array([0, 4, 5, 5, 6]), np.array([0, 0, 5, 6, 6]), 5)
        self.assertEqual(stats['n_predicted'], 3)
        self.assertEqual(stats['n_observed'], 3)
        self.assertEqual(stats['true_positive'], 3)
        self.assertAlmostEqual(stats['precision'], 1.0)
        self.assertAlmostEqual(stats['recall'], 1.0)
        self.assertAlmostEqual(stats['f1'], 1.0)

    def test_calibration_is_deterministic(self):
        x, y, groups, folds = self._affine_frame()
        first, first_coefficients = calibrate_panel_mean(x, y, groups, folds)
        second, second_coefficients = calibrate_panel_mean(x, y, groups, folds)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_coefficients['slope'],
                                      second_coefficients['slope'])


class ParentChildCapTest(unittest.TestCase):
    def test_child_is_capped_by_parent(self):
        grades = np.array([[0, 5, 6], [3, 5, 2]])
        capped = enforce_parent_child(grades, [0, 0], [1, 2])
        np.testing.assert_array_equal(capped, np.array([[0, 0, 0], [3, 3, 2]]))

    def test_child_is_capped_to_a_lower_parent(self):
        grades = np.array([[2, 4, 6], [1, 5, 6]])
        capped = enforce_parent_child(grades, [0], [2])
        np.testing.assert_array_equal(capped, np.array([[2, 4, 2], [1, 5, 1]]))

    def test_original_matrix_is_not_mutated(self):
        grades = np.array([[0, 5]])
        enforce_parent_child(grades, [0], [1])
        np.testing.assert_array_equal(grades, np.array([[0, 5]]))

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            enforce_parent_child(np.zeros((2, 3)), [0], [1, 2])


class ObservedHighGradeRatesTest(unittest.TestCase):
    def test_rates_are_fractions_of_the_panel(self):
        panel_median = np.array([[0, 5, 6], [4, 5, 3], [0, 0, 2], [6, 5, 0]])
        rate_ge5, rate_6 = observed_high_grade_rates(panel_median)
        np.testing.assert_allclose(rate_ge5, np.array([0.25, 0.75, 0.25]))
        np.testing.assert_allclose(rate_6, np.array([0.25, 0.0, 0.25]))

    def test_empty_or_off_scale_input_rejected(self):
        with self.assertRaises(ValueError):
            observed_high_grade_rates(np.zeros((0, 3)))
        with self.assertRaises(ValueError):
            observed_high_grade_rates(np.array([[0, 7]]))


class RuntimeDecisionTest(unittest.TestCase):
    def _inputs(self):
        p_ge5 = np.array([[0.1, 0.9], [0.8, 0.2], [0.95, 0.7], [0.3, 0.4]])
        p6 = np.array([[0.0, 0.8], [0.7, 0.1], [0.9, 0.6], [0.1, 0.05]])
        base = np.array([[0, 5], [4, 0], [3, 6], [2, 2]])
        rates5 = np.array([0.5, 0.5])
        rates6 = np.array([0.25, 0.25])
        return p_ge5, p6, base, rates5, rates6

    def test_matches_topk_then_cap_implementation(self):
        p_ge5, p6, base, rates5, rates6 = self._inputs()
        targets = ['A_a', 'A_b']
        grades = runtime_decision(p_ge5, p6, base, rates5, rates6, targets)
        expected = topk_decision(p_ge5.ravel(), p6.ravel(), np.minimum(base, 4).ravel(),
                                 np.tile(rates5, len(p_ge5)),
                                 np.tile(rates6, len(p_ge5)),
                                 np.tile(np.array(targets, object), len(p_ge5))).reshape(4, 2)
        np.testing.assert_array_equal(grades, expected)

    def test_parent_cap_applies_after_topk(self):
        p_ge5 = np.full((3, 2), 0.9)
        p6 = np.array([[0.0, 0.9], [0.0, 0.9], [0.0, 0.9]])
        base = np.zeros((3, 2), dtype=int)
        grades = runtime_decision(p_ge5, p6, base, np.array([1.0, 1.0]),
                                  np.array([0.0, 0.0]), ['A_a', 'A_b'], [0], [1])
        self.assertTrue((grades[:, 1] <= grades[:, 0]).all())

    def test_zero_rate_targets_never_emit_high_grades(self):
        p_ge5, p6, base, _, _ = self._inputs()
        grades = runtime_decision(p_ge5, p6, base, np.zeros(2), np.zeros(2), ['A_a', 'A_b'])
        self.assertTrue((grades <= 4).all())

    def test_shape_and_rate_validation(self):
        p_ge5, p6, base, rates5, rates6 = self._inputs()
        with self.assertRaises(ValueError):
            runtime_decision(p_ge5, p6[:, :1], base, rates5, rates6, ['A_a', 'A_b'])
        with self.assertRaises(ValueError):
            runtime_decision(p_ge5, p6, base, np.array([1.5, 0.5]), rates6, ['A_a', 'A_b'])


class NestedCalibrationTest(unittest.TestCase):
    def test_fit_uses_only_rows_of_its_own_fold(self):
        scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])[:, None]
        reference = 0.5 + 0.8 * scores[:, 0]
        groups = np.array(['A'] * 4 + ['B'] * 4, dtype=object)
        folds = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        models = fit_nested_linear(scores, reference, groups, folds)
        self.assertEqual(sorted(models), [('A', '0'), ('A', '1'), ('B', '0'), ('B', '1')])
        for key in [('A', '0'), ('A', '1'), ('B', '0'), ('B', '1')]:
            np.testing.assert_allclose(models[key][1], 0.8)
            np.testing.assert_allclose(models[key][0], 0.5)

    def test_perturbing_held_out_reference_does_not_change_predictions(self):
        rng = np.random.default_rng(7)
        n_fit, n_held = 32, 12
        scores_fit = rng.integers(0, 5, size=n_fit).astype(float)[:, None]
        reference_fit = 0.4 + 0.9 * scores_fit[:, 0] + rng.normal(0, 0.02, n_fit)
        groups_fit = np.array(['A', 'B'] * (n_fit // 2), dtype=object)
        folds_fit = np.arange(n_fit) % 2
        models = fit_nested_linear(scores_fit, reference_fit, groups_fit, folds_fit)
        scores_held = rng.integers(0, 5, size=n_held).astype(float)[:, None]
        groups_held = np.array(['A', 'B'] * (n_held // 2), dtype=object)
        folds_held = np.arange(n_held) % 2
        base = apply_nested_linear(scores_held, groups_held, folds_held, models)
        perturbed = scores_held.copy()
        perturbed += 0.25
        moved = apply_nested_linear(perturbed, groups_held, folds_held, models)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(base, moved)
        refit = apply_nested_linear(scores_held, groups_held, folds_held,
                                    fit_nested_linear(scores_fit, reference_fit,
                                                      groups_fit, folds_fit))
        np.testing.assert_array_equal(base, refit)


if __name__ == '__main__':
    unittest.main()
