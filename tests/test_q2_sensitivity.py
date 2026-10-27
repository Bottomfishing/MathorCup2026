"""Unit tests for the Q2 parameter-sensitivity helpers."""
import unittest

import numpy as np

from beer_models.q2_sensitivity import (BASELINE_VARIANT, BOUNDARY_THRESHOLDS, COMMUNITY_RANGE,
                                        MAX_LARGEST_SHARE, RESOLUTION_GRID, STABLE_MIN_JACCARD,
                                        STABLE_MIN_TARGETS, THRESHOLD_GRID, community_partition,
                                        evidence_tier_counts, frequency_table, overlap_metrics,
                                        resolution_verdict, threshold_verdict, weighted_adjacency)
from beer_models.q2_top3 import louvain_labels


def small_weighted():
    """6-node weighted graph with two dense triangles joined by one weak edge."""
    weights = np.zeros((6, 6))
    for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]:
        weights[i, j] = weights[j, i] = 0.8
    weights[2, 3] = weights[3, 2] = 0.3
    return weights


class WeightedAdjacencyTest(unittest.TestCase):
    def test_threshold_keeps_only_strong_edges_and_zeroes_diagonal(self):
        rho = np.array([[1.0, 0.6, 0.4], [0.6, 1.0, 0.55], [0.4, 0.55, 1.0]])
        weights = weighted_adjacency(rho, 0.5)
        self.assertEqual(weights[0, 1], 0.6)
        self.assertEqual(weights[1, 2], 0.55)
        self.assertEqual(weights[0, 2], 0.0)
        self.assertTrue(np.all(np.diag(weights) == 0.0))
        self.assertTrue(np.allclose(weights, weights.T))

    def test_threshold_is_inclusive(self):
        rho = np.array([[1.0, 0.5], [0.5, 1.0]])
        weights = weighted_adjacency(rho, 0.5)
        self.assertEqual(weights[0, 1], 0.5)


class CommunityPartitionTest(unittest.TestCase):
    def test_default_resolution_matches_delivered_louvain(self):
        weights = small_weighted()
        labels, sizes = community_partition(weights, resolution=1.0, seed=42)
        expected_labels, expected_sizes = louvain_labels(weights, seed=42)
        self.assertEqual(list(labels), list(expected_labels))
        self.assertEqual(sizes, expected_sizes)

    def test_partition_is_deterministic_and_labelled_by_size(self):
        weights = small_weighted()
        first, sizes_first = community_partition(weights, resolution=1.0, seed=42)
        second, sizes_second = community_partition(weights, resolution=1.0, seed=42)
        self.assertEqual(list(first), list(second))
        self.assertEqual(sizes_first, sizes_second)
        self.assertEqual(sizes_first, sorted(sizes_first, reverse=True))

    def test_larger_resolution_gives_at_least_as_many_communities(self):
        weights = small_weighted()
        _, coarse = community_partition(weights, resolution=0.8, seed=42)
        _, fine = community_partition(weights, resolution=1.2, seed=42)
        self.assertGreaterEqual(len(fine), len(coarse))


class OverlapMetricsTest(unittest.TestCase):
    def test_identical_and_disjoint_sets(self):
        baseline = {'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']}
        same = {'a': ['z', 'y', 'x'], 'b': ['p', 'q', 'r']}
        rows, aggregate = overlap_metrics(baseline, same)
        self.assertEqual([row['jaccard'] for row in rows], [1.0, 1.0])
        self.assertEqual(aggregate['n_targets_identical'], 2)
        other = {'a': ['u', 'v', 'w'], 'b': ['p', 'q', 'r']}
        _, aggregate = overlap_metrics(baseline, other)
        self.assertEqual(aggregate['n_targets_overlap_ge_2'], 1)
        self.assertAlmostEqual(aggregate['jaccard_mean'], 0.5)

    def test_partial_overlap_counts(self):
        rows, aggregate = overlap_metrics({'a': ['x', 'y', 'z']}, {'a': ['x', 'y', 'w']})
        self.assertEqual(rows[0]['overlap'], 2)
        self.assertAlmostEqual(rows[0]['jaccard'], 2 / 4)
        self.assertEqual(aggregate['n_targets_jaccard_ge_0_5'], 1)


class ThresholdVerdictTest(unittest.TestCase):
    def make_rows(self, jaccard, overlap_two_thirds):
        rows = []
        for threshold in THRESHOLD_GRID:
            rows.append({'threshold': threshold, 'jaccard_mean': jaccard,
                         'jaccard_median': jaccard,
                         'n_targets_overlap_ge_2': overlap_two_thirds,
                         'n_targets_jaccard_ge_0_5': overlap_two_thirds,
                         'n_targets': 50})
        return rows

    def test_passes_when_criterion_met(self):
        verdict = threshold_verdict(self.make_rows(0.6, STABLE_MIN_TARGETS))
        self.assertTrue(verdict['passed'])
        self.assertEqual(sorted(verdict['details']), ['0.45', '0.50', '0.55'])

    def test_fails_when_jaccard_too_low(self):
        verdict = threshold_verdict(self.make_rows(0.4, 50))
        self.assertFalse(verdict['passed'])

    def test_fails_when_too_few_dimensions_overlap(self):
        verdict = threshold_verdict(self.make_rows(0.9, STABLE_MIN_TARGETS - 1))
        self.assertFalse(verdict['passed'])

    def test_boundary_variants_are_the_preregistered_ones(self):
        self.assertEqual(BOUNDARY_THRESHOLDS, (0.45, 0.50, 0.55))
        self.assertIn(0.50, THRESHOLD_GRID)
        self.assertEqual(BASELINE_VARIANT['threshold'], 0.5)


class ResolutionVerdictTest(unittest.TestCase):
    def make_rows(self, n_communities, largest_share, n_nodes=231):
        return [{'resolution': resolution, 'n_communities': n_communities,
                 'largest_community': round(largest_share * n_nodes),
                 'largest_share': largest_share} for resolution in RESOLUTION_GRID]

    def test_passes_inside_community_range_and_share(self):
        verdict = resolution_verdict(self.make_rows(60, 0.12))
        self.assertTrue(verdict['passed'])
        self.assertEqual(verdict['stable_resolution_range'], [min(RESOLUTION_GRID), max(RESOLUTION_GRID)])

    def test_fails_when_too_few_communities(self):
        self.assertFalse(resolution_verdict(self.make_rows(COMMUNITY_RANGE[0] - 1, 0.30))['passed'])

    def test_fails_when_largest_community_too_big(self):
        self.assertFalse(resolution_verdict(self.make_rows(60, MAX_LARGEST_SHARE + 0.01))['passed'])

    def test_required_resolution_grid(self):
        self.assertEqual(RESOLUTION_GRID, (0.8, 1.0, 1.2))
        self.assertEqual(STABLE_MIN_JACCARD, 0.5)


class FrequencyAndTierTest(unittest.TestCase):
    def test_frequency_table_covers_all_candidates(self):
        replicates = [('a', 'b', 'c'), ('a', 'b', 'd'), ('a', 'b', 'c')]
        table = frequency_table(replicates, ['a', 'b', 'c', 'd'], 3)
        self.assertEqual(table, {'a': 1.0, 'b': 1.0, 'c': 2 / 3, 'd': 1 / 3})

    def test_evidence_tier_counts(self):
        rows = [{'chemical_evidence': 'A', 'confidence': 'high'},
                {'chemical_evidence': 'B', 'confidence': 'medium'},
                {'chemical_evidence': 'C', 'confidence': 'low'}]
        counts = evidence_tier_counts(rows)
        self.assertEqual(counts['tiers'], {'A': 1, 'B': 1, 'C': 1})
        self.assertEqual(counts['confidence'], {'high': 1, 'medium': 1, 'low': 1})


if __name__ == '__main__':
    unittest.main()
