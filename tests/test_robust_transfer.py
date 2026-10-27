import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from beer_models.robust_transfer import sparse_decode, refit_metrics
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from q3_robust_transfer import chemical_limits
from q3_risk_aware_pruning import parameter_draws


class RobustTransferTests(unittest.TestCase):
    def test_zero_reference_must_not_borrow_another_beers_positive_mapping(self):
        data = dict(features=['X4VG'], x=np.array([[-.5], [-.5], [-2.]]), ids=pd.DataFrame({'beer': [1, 2, 3]}))
        reference = pd.DataFrame({'X4VG': [0., 10**(-.5), .01]}, index=[1, 2, 3])
        source = np.array([-.5]); p = dict(reducible_fields=['X4VG'])
        censored, _ = chemical_limits(source, data, reference, p, source_beer=1)
        positive, _ = chemical_limits(source, data, reference, p, source_beer=2)
        np.testing.assert_array_equal(censored, source)
        self.assertLess(positive[0], source[0])
        with self.assertRaises(ValueError): chemical_limits(source, data, reference, p)

    def test_coordinate_stress_preserves_locked_fields_and_chemical_direction(self):
        source = np.array([1., -2., .4])
        selected = np.array([2., -2., .1])
        jitter = parameter_draws(source, selected, 32, 7)
        np.testing.assert_array_equal(jitter[:, 1], -2.)
        self.assertTrue((jitter[:, 2] >= .1-1e-12).all() and (jitter[:, 2] <= .4+1e-12).all())
        self.assertTrue((jitter[:, 0] >= 1-1e-12).all() and (jitter[:, 0] <= 2+1e-12).all())

    def test_decoder_keeps_unselected_and_censored_fields_while_reducing_supported_fields(self):
        source = np.array([1., 2., -2., .3, .01, 4.])
        neighbors = np.array([[3., 7., 5., 1., 1., 9.], [5., 6., 5., 1., 1., 10.]])
        u = np.array([[1., 0., 1., .5, 0., .5, 1.]])
        x = sparse_decode(u, source, neighbors, [0, 1], [0, 1, 0, 0, 0, 0], [3, 4, 5], [.1, .005, 3.], n_groups=2)
        np.testing.assert_allclose(x, [[3., 4.5, -2., .3, .0075, 3.]])
        self.assertEqual(int((abs(x-source) > 1e-10).sum()), 4)

    def test_zero_weight_endpoint_and_zero_progress_reproduce_source_except_explicit_reduction(self):
        source = np.array([1., .2])
        u = np.zeros((1, 4))
        x = sparse_decode(u, source, np.array([[2., .4], [4., .6]]), [0], [0, 0], [1], [.1], n_groups=1)
        np.testing.assert_array_equal(x[0], source)
        with self.assertRaises(ValueError):
            sparse_decode(u-1, source, np.ones((2, 2)), [0], [0, 0], [1], [.1], n_groups=1)

    def test_joint_event_requires_style_and_risk_in_same_refit(self):
        grade = np.zeros((4, 1, 5), int)
        delta = np.full((4, 1, 5), -.01)
        delta[2:, 0, 0] = .01
        style = np.array([[False], [False], [True], [True]])
        margins = np.ones((4, 1, 2))*.1
        req = dict(probability_delta_quantile=.8, maximum_four_nonzero_delta_quantile=0., minimum_joint_refit_fraction=.5)
        result = refit_metrics(grade, delta, margins, style, req)
        self.assertEqual(float(result['joint_fraction'][0]), 0.)
        self.assertFalse(result['qualified'][0])

    def test_margin_applies_to_each_supported_target_and_does_not_certify_stale(self):
        grades = np.zeros((8, 2, 5), int)
        delta = np.full((8, 2, 5), -.01)
        delta[:, 0, 4] = .2
        delta[:, 1, 2] = -.001
        req = dict(probability_delta_quantile=.8, maximum_four_nonzero_delta_quantile=-.002,
            minimum_joint_refit_fraction=.75, minimum_style_margin_quantile=0.)
        result = refit_metrics(grades, delta, np.ones((8, 2, 2))*.1, np.ones((8, 2), bool), req)
        self.assertTrue(result['qualified'][0])
        self.assertEqual(result['five_nonzero_fraction'][0], 0.)
        self.assertFalse(result['qualified'][1])


if __name__ == '__main__':
    unittest.main()
