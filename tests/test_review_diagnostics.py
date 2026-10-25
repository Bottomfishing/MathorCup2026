import unittest
import numpy as np
from beer_models.review_diagnostics import change_summary, set_overlap, pareto_mask


class ReviewDiagnosticsTests(unittest.TestCase):
    def test_actual_direction_not_interval_label(self):
        result = change_summary([[-2, 1], [0, 2], [1, 3], [-1, 4]], 0)
        np.testing.assert_allclose(result['pass_fraction'], [.75, 0])
        np.testing.assert_allclose(result['median'], [-.5, 2.5])

    def test_reject_invalid(self):
        for x in [[], [[np.nan]], [1, 2]]:
            with self.assertRaises(ValueError): change_summary(x)
        with self.assertRaises(ValueError): change_summary([[0]], -1)

    def test_overlap_sets(self):
        self.assertEqual(set_overlap(['a', 'b', 'c'], ['b', 'c', 'd']), .5)
        self.assertEqual(set_overlap(['a', 'a'], ['a']), 1)
        with self.assertRaises(ValueError): set_overlap([], ['a'])

    def test_pareto_matches_direct_dominance_with_ties(self):
        points=np.array([[1,3],[1,2],[1,2],[2,2],[2,1],[3,3],[0,4]])
        direct=[not any(np.all(q<=p) and np.any(q<p) for q in points) for p in points]
        np.testing.assert_array_equal(pareto_mask(points),direct)
        self.assertEqual(len(pareto_mask(np.empty((0,2)))),0)
