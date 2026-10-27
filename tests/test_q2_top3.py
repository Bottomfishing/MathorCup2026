"""Unit tests for the Q2 Top-3 library (pure functions, synthetic data only)."""
import unittest

import numpy as np

from beer_models.q2_top3 import (IDENTITY_MARKERS, DEFAULT_EXCLUDED, adjacency_from_corr,
                                 adjacency_subset, bootstrap_indices, bootstrap_jaccard,
                                 bootstrap_top_k, canon_key, candidate_features, classify_compound,
                                 evidence_grade, extra_trees_importance, jaccard, louvain_labels,
                                 marginal_spearman, method_jaccard_row, neighbourhood_scores,
                                 preprocess, ridge_coefficients, selection_frequencies,
                                 target_families, top_k_fields)


class Q2Top3Test(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.X = self.rng.normal(size=(60, 6))
        self.X[:, 3] = np.nan
        self.X[0, 3] = -1.5
        self.Y = self.rng.normal(size=(60, 2))
        self.fields = [f'f{i}' for i in range(6)]

    def test_canon_key_folds_greek_letters(self):
        self.assertEqual(canon_key('α-Terpineol'), 'alphaterpineol')
        self.assertEqual(canon_key('4-vinylguaiacol'), '4vinylguaiacol')
        self.assertEqual(canon_key('X2.3.butanedione'), canon_key('X2.3.butanedione'))

    def test_candidate_features_excludes_whole_beer_parameters(self):
        fields = list(DEFAULT_EXCLUDED) + ['DMS', 'X4VG']
        self.assertEqual(candidate_features(fields), ['DMS', 'X4VG'])
        self.assertEqual(len(DEFAULT_EXCLUDED), 12)

    def test_adjacency_from_corr_threshold_and_diagonal(self):
        rho = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, -0.7], [0.2, -0.7, 1.0]])
        adjacency, edges = adjacency_from_corr(rho, threshold=0.5)
        self.assertEqual(adjacency.shape, (3, 3))
        self.assertEqual(adjacency.sum(), 4.0)
        self.assertTrue(np.all(np.diag(adjacency) == 0.0))
        self.assertEqual(sorted((e['source'], e['target']) for e in edges), [(0, 1), (1, 2)])
        self.assertAlmostEqual(max(e['weight'] for e in edges), 0.7)

    def test_adjacency_subset_keeps_requested_nodes(self):
        rho = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.8], [0.1, 0.8, 1.0]])
        adjacency, _ = adjacency_from_corr(rho, threshold=0.5)
        subset = adjacency_subset(adjacency, [0, 2])
        self.assertEqual(subset.shape, (2, 2))
        self.assertEqual(subset.sum(), 0.0)

    def test_louvain_labels_deterministic_and_relabelled(self):
        adjacency = np.zeros((6, 6))
        for a, b in ((0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)):
            adjacency[a, b] = adjacency[b, a] = 1.0
        labels, sizes = louvain_labels(adjacency, seed=42)
        again, _ = louvain_labels(adjacency, seed=42)
        np.testing.assert_array_equal(labels, again)
        self.assertEqual(len(labels), 6)
        self.assertEqual(sorted(sizes, reverse=True), sizes)
        self.assertEqual(max(labels) + 1, 2)
        self.assertEqual(sizes[0], 3)

    def test_preprocess_imputes_and_standardises(self):
        standardised, centred = preprocess(self.X, self.Y)
        self.assertTrue(np.isfinite(standardised).all())
        np.testing.assert_allclose(standardised.mean(axis=0), 0.0, atol=1e-10)
        scales = standardised.std(axis=0)
        np.testing.assert_allclose(scales[scales > 0.5], 1.0, atol=1e-10)
        np.testing.assert_allclose(centred.mean(axis=0), 0.0, atol=1e-10)

    def test_ridge_coefficients_shape(self):
        standardised, centred = preprocess(self.X, self.Y)
        beta = ridge_coefficients(standardised, centred, alpha=10.0)
        self.assertEqual(beta.shape, (6, 2))
        self.assertTrue(np.isfinite(beta).all())

    def test_neighbourhood_scores_formula(self):
        beta = np.array([[1.0], [0.0]])
        adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
        scores = neighbourhood_scores(beta, adjacency, weight=0.5)
        self.assertAlmostEqual(scores[0, 0], 1.0)
        self.assertAlmostEqual(scores[1, 0], 0.5)

    def test_top_k_fields_returns_three_unique_names_per_target(self):
        scores = np.array([[1.0, 0.1], [0.2, 2.0], [3.0, 0.3], [0.4, 4.0], [0.5, 0.6]])
        selections = top_k_fields(scores, self.fields)
        self.assertEqual(len(selections), 2)
        for selection in selections:
            self.assertEqual(len(selection), 3)
            self.assertEqual(len(set(selection)), 3)
        self.assertEqual(selections[0][0], 'f2')
        self.assertEqual(selections[1][0], 'f3')

    def test_bootstrap_indices_reproducible(self):
        first = bootstrap_indices(60, 5, seed=11)
        second = bootstrap_indices(60, 5, seed=11)
        other = bootstrap_indices(60, 5, seed=12)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(first[0], other[0]))

    def test_bootstrap_top_k_shapes(self):
        adjacency = np.eye(6)
        finite = self.rng.normal(size=(60, 6))
        replicates = bootstrap_top_k(finite, self.Y, self.fields, adjacency,
                                     alpha=10.0, n_bootstrap=5, seed=3)
        self.assertEqual(len(replicates), 2)
        for target_replicates in replicates:
            self.assertEqual(len(target_replicates), 5)
            for replicate in target_replicates:
                self.assertEqual(len(replicate), 3)
                self.assertEqual(len(set(replicate)), 3)
                self.assertTrue(set(replicate) <= set(self.fields))

    def test_selection_frequencies_and_jaccard(self):
        replicates = [('a', 'b', 'c'), ('a', 'b', 'd'), ('a', 'b', 'c'), ('x', 'y', 'z')]
        frequencies = selection_frequencies(replicates, ('a', 'b', 'c'), n_bootstrap=4)
        self.assertAlmostEqual(frequencies['a'], 0.75)
        self.assertAlmostEqual(frequencies['c'], 0.5)
        self.assertAlmostEqual(frequencies['b'], 0.75)
        self.assertAlmostEqual(jaccard(['a', 'b'], ['b', 'c']), 1 / 3)
        self.assertEqual(jaccard(['a', 'b'], ['a', 'b']), 1.0)
        stats = bootstrap_jaccard(replicates, ('a', 'b', 'c'))
        self.assertLessEqual(stats['min'], stats['mean'])
        self.assertLessEqual(stats['mean'], 1.0)

    def test_method_jaccard_row(self):
        sets_by_method = {
            'graphridge': {'t': ['a', 'b', 'c']},
            'ridge': {'t': ['a', 'b', 'c']},
            'extratrees': {'t': ['a', 'x', 'y']},
            'marginal_spearman': {'t': ['m', 'n', 'o']},
        }
        pairs = [('graphridge', 'ridge'), ('graphridge', 'extratrees')]
        row = method_jaccard_row('t', sets_by_method, pairs)
        self.assertAlmostEqual(row['jaccard_graphridge_vs_ridge'], 1.0)
        self.assertEqual(row['overlap_n_graphridge_vs_extratrees'], 1)
        self.assertAlmostEqual(row['jaccard_mean'], (1.0 + 1 / 5) / 2)
        self.assertEqual(row['n_pairs_ge_0.5'], 1)

    def test_classify_compound_mechanism_classes(self):
        self.assertEqual(classify_compound('4-vinylguaiacol')[0], 'phenol')
        self.assertEqual(classify_compound('Linalool')[0], 'terpene')
        self.assertEqual(classify_compound('Furfural')[0], 'maillard')
        self.assertEqual(classify_compound('2,3-butanedione')[0], 'diacetyl')
        self.assertEqual(classify_compound('not-a-real-compound')[0], 'other')

    def test_target_families_prefix_and_overrides(self):
        families = target_families(['A_hops_citrus', 'F_malt_bread', 'diacetyl', 'bitternes', 'body'])
        self.assertEqual(families['A_hops_citrus'], {'hops'})
        self.assertEqual(families['F_malt_bread'], {'malt'})
        self.assertEqual(families['diacetyl'], {'diacetyl'})
        self.assertEqual(families['bitternes'], {'bitter'})
        self.assertTrue({'body', 'protein', 'sugar'} <= families['body'])

    def test_evidence_grade_identity_marker_is_high_confidence(self):
        entry = evidence_grade('X4vg', 'X4VG', coefficient=0.4, cross_corr=0.47, frequency=1.0,
                               identity_markers=IDENTITY_MARKERS, display_name='4-vinylguaiacol')
        self.assertEqual(entry['chemical_evidence'], 'A')
        self.assertEqual(entry['confidence'], 'high')
        self.assertEqual(entry['class_key'], 'phenol')
        self.assertTrue(entry['target_semantic_match'])

    def test_evidence_grade_unsupported_entry_is_exploratory(self):
        entry = evidence_grade('A_malt_bread', 'unknown_field', coefficient=0.2, cross_corr=np.nan,
                               frequency=0.1, identity_markers=IDENTITY_MARKERS,
                               display_name='not-a-real-compound')
        self.assertEqual(entry['chemical_evidence'], 'C')
        self.assertEqual(entry['confidence'], 'low')
        self.assertFalse(entry['target_semantic_match'])

    def test_alternative_importance_matrices(self):
        standardised, _ = preprocess(self.X, self.Y)
        importance = extra_trees_importance(standardised, self.Y,
                                            params={'n_estimators': 10, 'random_state': 1, 'n_jobs': 1})
        self.assertEqual(importance.shape, (6, 2))
        self.assertTrue((importance >= 0).all())
        marginal = marginal_spearman(np.nan_to_num(self.X, nan=0.0), self.Y)
        self.assertEqual(marginal.shape, (6, 2))
        self.assertTrue(np.isfinite(marginal).all())


if __name__ == '__main__':
    unittest.main()
