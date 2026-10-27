"""Unit tests for :mod:`beer_models.q1_decision` on synthetic data.

Run with::

    $env:PYTHONPATH='src'
    .venv/Scripts/python.exe -m unittest tests.test_q1_decision -v
"""
from __future__ import annotations

import unittest

import numpy as np

from beer_models.q1_decision import (
    calibrate_panel_mean,
    fold_style_mean,
    fold_training_mean,
    fold_training_rate,
    high_grade_stats,
    isotonic_calibrate,
    linear_to_grade,
    quantile_calibrate,
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


if __name__ == '__main__':
    unittest.main()
