"""Tests for the batch-independent fixed-threshold Q1 decision layer (T1)."""
import unittest

import numpy as np

from beer_models.q1_decision import (
    enforce_parent_child,
    fold_threshold,
    runtime_threshold_decision,
    threshold_decision,
    training_threshold,
)


class FoldThresholdTest(unittest.TestCase):
    def setUp(self):
        # Two targets x two folds; each fold has three cells per target.
        self.targets = np.array(['a'] * 6 + ['b'] * 6, dtype=object)
        self.folds = np.array([0, 1, 0, 1, 0, 1] * 2, dtype=object)
        self.probs = np.array([.9, .8, .7, .6, .5, .4,
                               .95, .55, .5, .45, .4, .3], dtype=float)
        self.labels = np.array([6, 5, 5, 3, 2, 0,
                                6, 5, 2, 1, 0, 0], dtype=float)

    def test_threshold_uses_other_folds_only(self):
        tau = fold_threshold(self.probs, self.labels, self.targets, self.folds, 5)
        # fold 0 of target a is estimated from fold 1 cells (probs .8/.6/.4,
        # labels 5/3/0 -> r=1/3, n=3, k=1 -> threshold .8).
        self.assertAlmostEqual(float(tau[0]), 0.8)
        self.assertAlmostEqual(float(tau[2]), 0.8)
        # fold 1 of target a is estimated from fold 0 cells (.9/.7/.5, labels 6/5/2
        # -> r=2/3, n=3, k=2 -> threshold .7).
        self.assertAlmostEqual(float(tau[1]), 0.7)

    def test_validation_labels_do_not_change_threshold(self):
        tau = fold_threshold(self.probs, self.labels, self.targets, self.folds, 5)
        shifted = self.labels.copy()
        shifted[self.folds == 0] += 1.0  # perturb fold 0 (validation side)
        shifted = np.clip(shifted, 0, 6)
        tau_perturbed = fold_threshold(self.probs, shifted, self.targets, self.folds, 5)
        # fold 0 thresholds only read fold 1 labels, which were not perturbed.
        self.assertTrue(np.allclose(tau[self.folds == 0], tau_perturbed[self.folds == 0]))

    def test_no_positive_event_falls_back_to_infinity(self):
        labels = self.labels.copy()
        labels[self.folds == 1] = 0.0  # target a fold-1 cells have no >=5 event
        tau = fold_threshold(self.probs, labels, self.targets, self.folds, 5)
        self.assertTrue(np.isinf(tau[0]) and np.isinf(tau[2]))

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            fold_threshold(self.probs, self.labels, self.targets, self.folds[:5], 5)
        with self.assertRaises(ValueError):
            fold_threshold(self.probs + 1.0, self.labels, self.targets, self.folds, 5)
        with self.assertRaises(ValueError):
            fold_threshold(self.probs, self.labels + 7.0, self.targets, self.folds, 5)

    def test_group_without_training_fold_raises(self):
        targets = np.array(['a', 'a', 'b', 'b'], dtype=object)
        folds = np.array([0, 0, 0, 1], dtype=object)
        probs = np.array([.5, .4, .3, .2])
        labels = np.array([5, 2, 1, 0])
        with self.assertRaises(ValueError):
            fold_threshold(probs, labels, targets, folds, 5)


class TrainingThresholdTest(unittest.TestCase):
    def test_rate_support_and_threshold(self):
        probs = np.array([.9, .8, .7, .6, .5, .4])
        labels = np.array([6, 5, 5, 3, 2, 0])
        groups = np.array(['a'] * 4 + ['b'] * 2, dtype=object)
        details = training_threshold(probs, labels, groups, 5)
        self.assertEqual(details['keys'], ['a', 'b'])
        self.assertTrue(np.allclose(details['rate'], [0.75, 0.0]))
        self.assertEqual(details['support'].tolist(), [4, 2])
        self.assertEqual(details['k'].tolist(), [3, 0])
        self.assertAlmostEqual(float(details['tau'][0]), 0.7)
        self.assertTrue(np.isinf(details['tau'][1]))

    def test_no_positive_event(self):
        details = training_threshold(np.array([.2, .1]), np.array([0, 0]), ['a', 'a'], 6)
        self.assertTrue(np.isinf(details['tau'][0]))


class ThresholdDecisionTest(unittest.TestCase):
    def test_decision_rule(self):
        p_ge5 = np.array([.9, .6, .2])
        p6 = np.array([.8, .1, .05])
        base = np.array([3, 5, 2])
        tau5 = np.full(3, .5)
        tau6 = np.full(3, .7)
        groups = np.array(['a', 'a', 'a'], dtype=object)
        grades = threshold_decision(p_ge5, p6, base, tau5, tau6, groups)
        self.assertEqual(grades.tolist(), [6, 5, 2])

    def test_inf_threshold_suppresses_grade(self):
        grades = threshold_decision(np.array([.99]), np.array([.99]), np.array([1]),
                                    np.array([np.inf]), np.array([np.inf]), ['a'])
        self.assertEqual(grades.tolist(), [1])

    def test_mixed_thresholds_in_group_rejected(self):
        with self.assertRaises(ValueError):
            threshold_decision(np.array([.9, .9]), np.array([.9, .9]),
                               np.array([1, 1]), np.array([.5, .4]), np.array([.5, .5]),
                               np.array(['a', 'a'], dtype=object))

    def test_batch_invariance_of_pure_rule(self):
        rng = np.random.default_rng(7)
        n = 400
        p5 = rng.uniform(0, 1, n)
        p6 = rng.uniform(0, p5 + 1e-9, n)
        base = rng.integers(0, 7, n).astype(float)
        groups = np.repeat(['a', 'b'], n // 2).astype(object)
        tau5 = np.where(groups == 'a', .55, .35)
        tau6 = np.where(groups == 'a', .85, .75)
        whole = threshold_decision(p5, p6, base, tau5, tau6, groups)
        single = np.array([threshold_decision(p5[i:i + 1], p6[i:i + 1], base[i:i + 1],
                                              tau5[i:i + 1], tau6[i:i + 1], groups[i:i + 1])[0]
                           for i in range(n)])
        self.assertEqual(whole.tolist(), single.tolist())
        order = rng.permutation(n)
        shuffled = threshold_decision(p5[order], p6[order], base[order],
                                      tau5[order], tau6[order], groups[order])
        self.assertEqual(whole[order].tolist(), shuffled.tolist())


class RuntimeThresholdDecisionTest(unittest.TestCase):
    def test_matrix_batch_invariance_and_parent_cap(self):
        rng = np.random.default_rng(11)
        n_beers, n_targets = 30, 3
        p5 = rng.uniform(0, 1, (n_beers, n_targets))
        p6 = rng.uniform(0, 1, (n_beers, n_targets))
        base = rng.integers(0, 7, (n_beers, n_targets)).astype(float)
        targets = ['a', 'b', 'c']
        tau5 = np.array([.5, .6, .7])
        tau6 = np.array([.8, .85, .9])
        pairs = ([0], [2])  # cap: target c <= target a
        whole = runtime_threshold_decision(p5, p6, base, tau5, tau6, targets, *pairs)
        chunks = np.vstack([runtime_threshold_decision(p5[i:i + 4], p6[i:i + 4], base[i:i + 4],
                                                       tau5, tau6, targets, *pairs)
                            for i in range(0, n_beers, 4)])
        self.assertEqual(whole.tolist(), chunks.tolist())
        self.assertTrue((whole[:, 2] <= whole[:, 0]).all())
        self.assertTrue(((whole >= 0) & (whole <= 6)).all())

    def test_threshold_count_must_match_targets(self):
        with self.assertRaises(ValueError):
            runtime_threshold_decision(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)),
                                       np.array([.5]), np.array([.5]), ['a', 'b'])

    def test_parent_cap_is_deterministic(self):
        grades = np.array([[6, 3, 6], [1, 0, 5]])
        capped = enforce_parent_child(grades, [0], [2])
        self.assertEqual(capped.tolist(), [[6, 3, 6], [1, 0, 1]])


if __name__ == '__main__':
    unittest.main()
