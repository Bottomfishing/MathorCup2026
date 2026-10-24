import unittest
import numpy as np
from beer_models.style_diagnostics import nearest_centroid, nearest_neighbor_radius, select_transfer


class StyleDiagnosticsTests(unittest.TestCase):
    def test_nearest_centroid_uses_all_coordinates_and_fixed_order(self):
        centers = {'b': np.array([2., 2.]), 'a': np.array([0., 0.])}
        np.testing.assert_array_equal(nearest_centroid(np.array([[.1, .1], [1.9, 1.9], [1., 1.]]), centers), ['a', 'b', 'a'])

    def test_nn_radius_excludes_self_match(self):
        radius = nearest_neighbor_radius(np.array([[0., 0.], [3., 4.]]))
        self.assertAlmostEqual(radius, 5 / np.sqrt(2))
        with self.assertRaises(ValueError):
            nearest_neighbor_radius(np.array([[0., 0.]]))

    def test_target_style_preferred_only_among_feasible_candidates(self):
        selected = select_transfer([.1, .5, .3], [.1, .4, .2], [False, True, True], [True, True, False])
        self.assertEqual(selected, 1)

    def test_weight_changes_tradeoff_without_changing_constraints(self):
        self.assertEqual(select_transfer([.1, .4], [.9, .1], [True, True], [False, False], .9), 0)
        self.assertEqual(select_transfer([.1, .4], [.9, .1], [True, True], [False, False], .1), 1)
        self.assertIsNone(select_transfer([.1], [.1], [False], [True]))


if __name__ == '__main__':
    unittest.main()
