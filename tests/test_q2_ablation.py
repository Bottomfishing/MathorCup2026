"""Unit tests for the Q2 graph-neighbourhood ablation helpers."""
import unittest

import numpy as np

from beer_models.q2_ablation import (ABLATION_WEIGHT, GRAPH_WEIGHT, MAE_GAIN_THRESHOLD,
                                     MIN_NON_WORSE_FRACTION, SPEARMAN_GAIN_THRESHOLD,
                                     STABILITY_JACCARD_DELTA, ablation_verdict, grouped_cv_predictions,
                                     grouped_folds, overlap_row, paired_test, predictive_metrics,
                                     spearman_matrix, stability_row)
from beer_models.q2_top3 import (adjacency_from_corr, bootstrap_top_k, neighbourhood_scores,
                                 preprocess, ridge_coefficients, top_k_fields)


def synthetic(n_samples=40, n_features=8, n_targets=4, seed=7):
    """Small synthetic chemistry/response block that behaves like the real data."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_samples, 3))
    loading = rng.normal(size=(3, n_features))
    X = base @ loading + rng.normal(scale=0.6, size=(n_samples, n_features))
    weights = rng.normal(size=(n_features, n_targets))
    Y = X @ weights + rng.normal(scale=0.5, size=(n_samples, n_targets))
    return X, Y


class ZeroWeightTest(unittest.TestCase):
    def test_zero_weight_scores_are_absolute_ridge_coefficients(self):
        X, Y = synthetic()
        standardised, centred = preprocess(X, Y)
        beta = ridge_coefficients(standardised, centred, 1000.0)
        adjacency, _ = adjacency_from_corr(spearman_matrix(X), 0.5)
        scores = neighbourhood_scores(beta, adjacency, ABLATION_WEIGHT)
        self.assertTrue(np.allclose(scores, np.abs(beta)))
        self.assertTrue(np.allclose(neighbourhood_scores(beta, adjacency, 0.0), np.abs(beta)))

    def test_graph_weight_changes_scores(self):
        X, Y = synthetic()
        standardised, centred = preprocess(X, Y)
        beta = ridge_coefficients(standardised, centred, 1000.0)
        adjacency, _ = adjacency_from_corr(spearman_matrix(X), 0.5)
        graph = neighbourhood_scores(beta, adjacency, GRAPH_WEIGHT)
        plain = neighbourhood_scores(beta, adjacency, ABLATION_WEIGHT)
        self.assertFalse(np.allclose(graph, plain))


class FoldTest(unittest.TestCase):
    def test_folds_cover_every_sample_once(self):
        folds = grouped_folds(250, 5, 20261028)
        self.assertEqual(folds.size, 250)
        self.assertEqual(set(np.unique(folds)), set(range(5)))
        self.assertEqual(min(np.bincount(folds)), 50)

    def test_folds_are_deterministic(self):
        first = grouped_folds(250, 5, 20261028)
        second = grouped_folds(250, 5, 20261028)
        self.assertTrue(np.array_equal(first, second))

    def test_folded_predictions_are_complete(self):
        X, Y = synthetic(n_samples=30, n_features=6, n_targets=3)
        folds = grouped_folds(30, 3, 11)
        predictions, selections = grouped_cv_predictions(X, Y, ABLATION_WEIGHT, folds, alpha=10.0)
        self.assertEqual(predictions.shape, Y.shape)
        self.assertFalse(np.isnan(predictions).any())
        self.assertEqual(len(selections), 3 * 3)
        for key, columns in selections.items():
            self.assertEqual(len(columns), 3)
            self.assertEqual(len(set(columns)), 3)

    def test_predictive_metrics_are_exact_on_perfect_predictions(self):
        _, Y = synthetic(n_samples=20, n_features=4, n_targets=3)
        metrics = predictive_metrics(Y, Y.copy())
        self.assertTrue(np.allclose(metrics["mae"], 0.0, atol=1e-12))
        self.assertTrue(np.allclose(metrics["spearman"], 1.0, atol=1e-12))


class SpearmanMatrixTest(unittest.TestCase):
    def test_diagonal_is_one_and_matrix_is_symmetric(self):
        X, _ = synthetic(n_samples=25, n_features=5)
        matrix = spearman_matrix(X)
        self.assertTrue(np.allclose(np.diag(matrix), 1.0))
        self.assertTrue(np.allclose(matrix, matrix.T))

    def test_constant_column_is_handled(self):
        X = np.column_stack([np.arange(10.0), np.ones(10), np.arange(10.0) ** 2])
        matrix = spearman_matrix(X)
        self.assertFalse(np.isnan(matrix).any())


class OverlapTest(unittest.TestCase):
    def test_identical_sets_give_jaccard_one(self):
        X, Y = synthetic(n_samples=20, n_features=5, n_targets=2)
        standardised, centred = preprocess(X, Y)
        beta = ridge_coefficients(standardised, centred, 1.0)
        adjacency, _ = adjacency_from_corr(spearman_matrix(X), 0.5)
        scores = neighbourhood_scores(beta, adjacency, ABLATION_WEIGHT)
        chosen = top_k_fields(scores, [f"c{i}" for i in range(5)], 3)[0]
        frequency = {field: 0.5 for field in chosen}
        row = overlap_row("t", chosen, chosen, scores[:, 0], scores[:, 0], frequency, frequency,
                          [f"c{i}" for i in range(5)], {}, {})
        self.assertEqual(row["jaccard"], 1.0)
        self.assertEqual(row["overlap_n"], 3)
        self.assertAlmostEqual(row["rank_spearman"], 1.0)

    def test_disjoint_sets_give_jaccard_zero(self):
        scores_a = np.array([[3.0], [2.0], [1.0], [0.0]])
        scores_b = np.array([[0.0], [1.0], [2.0], [3.0]])
        fields = ["a", "b", "c", "d"]
        row = overlap_row("t", ["a", "b", "c"], ["b", "c", "d"], scores_a, scores_b, {}, {},
                          fields, {}, {})
        self.assertEqual(row["overlap_n"], 2)


class StabilityTest(unittest.TestCase):
    def test_stability_summary_counts(self):
        frequencies = [0.6] * 3 + [0.4] * 3
        row = stability_row("graph", frequencies, [0.5, 0.7], n_targets=2)
        self.assertEqual(row["n_top3_rows"], 6)
        self.assertEqual(row["n_slots_freq_ge_0_5"], 3)
        self.assertEqual(row["n_targets_mean_freq_ge_0_5"], 1)
        self.assertAlmostEqual(row["freq_mean"], 0.5)


class VerdictTest(unittest.TestCase):
    def _stats(self, gain, p, fraction, jaccard_gain=0.0, freq_gain=0.0):
        mae = {"gain_mean": gain, "p_value": p, "graph_non_worse_fraction": fraction}
        rho = {"gain_mean": 0.0, "p_value": 1.0, "graph_non_worse_fraction": 0.5}
        stability = {
            "graph_jaccard_mean": 0.5 + jaccard_gain,
            "no_graph_jaccard_mean": 0.5,
            "graph_freq_mean": 0.5 + freq_gain,
            "no_graph_freq_mean": 0.5,
        }
        return mae, rho, stability

    def test_no_reliable_gain(self):
        mae, rho, stability = self._stats(0.001, 0.5, 0.5)
        verdict = ablation_verdict(mae, rho, stability)
        self.assertEqual(verdict["verdict"], "no_reliable_gain")

    def test_reliable_gain_requires_both_axes(self):
        mae, rho, stability = self._stats(MAE_GAIN_THRESHOLD + 0.01, 0.01, 0.8,
                                          jaccard_gain=STABILITY_JACCARD_DELTA + 0.01)
        verdict = ablation_verdict(mae, rho, stability)
        self.assertEqual(verdict["verdict"], "reliable_gain")

    def test_partial_gain_when_only_predictive_reliable(self):
        mae, rho, stability = self._stats(MAE_GAIN_THRESHOLD + 0.01, 0.01, 0.8)
        verdict = ablation_verdict(mae, rho, stability)
        self.assertEqual(verdict["verdict"], "partial_gain")

    def test_large_gain_needs_breadth(self):
        mae, rho, stability = self._stats(MAE_GAIN_THRESHOLD + 0.02, 0.01, 0.3,
                                          jaccard_gain=STABILITY_JACCARD_DELTA + 0.01)
        verdict = ablation_verdict(mae, rho, stability)
        self.assertFalse(verdict["predictive_reliable"])

    def test_spearman_threshold_is_used(self):
        mae = {"gain_mean": 0.0, "p_value": 1.0, "graph_non_worse_fraction": 0.5}
        rho = {"gain_mean": SPEARMAN_GAIN_THRESHOLD + 0.01, "p_value": 0.01,
               "graph_non_worse_fraction": MIN_NON_WORSE_FRACTION + 0.1}
        stability = {"graph_jaccard_mean": 0.5, "no_graph_jaccard_mean": 0.5,
                     "graph_freq_mean": 0.5, "no_graph_freq_mean": 0.5}
        verdict = ablation_verdict(mae, rho, stability)
        self.assertTrue(verdict["criteria"]["spearman_reliable"])


class PairedTestTest(unittest.TestCase):
    def test_identical_vectors_have_zero_difference(self):
        values = np.linspace(0.0, 1.0, 10)
        result = paired_test(values, values)
        self.assertEqual(result["mean_difference"], 0.0)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["n_ties"], 10)


class BootstrapDeterminismTest(unittest.TestCase):
    def test_same_seed_gives_identical_replicates(self):
        X, Y = synthetic(n_samples=30, n_features=6, n_targets=3)
        adjacency, _ = adjacency_from_corr(spearman_matrix(X), 0.5)
        fields = [f"c{i}" for i in range(6)]
        first = bootstrap_top_k(X, Y, fields, adjacency, alpha=10.0, n_bootstrap=5, seed=99,
                                weight=ABLATION_WEIGHT)
        second = bootstrap_top_k(X, Y, fields, adjacency, alpha=10.0, n_bootstrap=5, seed=99,
                                 weight=ABLATION_WEIGHT)
        self.assertEqual([list(entry) for entry in first], [list(entry) for entry in second])


if __name__ == "__main__":
    unittest.main()
