import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from beer_models.local_transfer import decode, guard_metrics, rank_candidates, defect_predictions
from beer_models.tail_risk import SelectedRiskModel


class Identity:
    def transform(self, x):
        return x


class FakeForest:
    classes_ = [np.arange(7)]*5
    def predict_proba(self, x):
        p = np.array([.5, .2, .1, .1, .05, .03, .02])
        return [np.tile(p, (len(x), 1)) for _ in range(5)]


class LocalTransferTests(unittest.TestCase):
    def test_local_simplex_preserves_locked_fields_and_bounds(self):
        source = np.array([1., 2., 3.])
        neighbors = np.array([[4., 9., 6.], [7., 8., 9.]])
        x, progress, weights = decode(np.array([[1, 1, 0], [0, 0, 0], [.5, .2, .8]]), source, neighbors, [0, 2])
        np.testing.assert_array_equal(x[:, 1], 2.)
        np.testing.assert_allclose(weights.sum(1), 1.)
        np.testing.assert_allclose(x[0], [4, 2, 6])
        self.assertTrue((x >= source).all())
        self.assertTrue((progress >= .1).all() and (progress <= 1).all())
        with self.assertRaises(ValueError):
            decode(np.array([[1, -1, 0]]), source, neighbors, [0])

    def test_stale_screen_is_distinct_from_four_supported_guards(self):
        p = np.zeros((1, 5, 7)); p[:, :, 0] = 1
        before = dict(grade=np.zeros((1, 5), int), uncalibrated=p, calibrated=p)
        afterp = p.copy(); afterp[:, 4, 0] = .9; afterp[:, 4, 1] = .1
        after = dict(grade=np.zeros((1, 5), int), uncalibrated=afterp, calibrated=afterp)
        gates, _ = guard_metrics(before, after)
        self.assertTrue(gates['supported_nonzero_pass'].item())
        self.assertFalse(gates['five_nonzero_pass'].item())
        self.assertFalse(gates['thirty_guard_pass'].item())

    def test_probability6_increase_is_not_the_same_as_nonzero_increase(self):
        a = np.zeros((1, 5, 7)); a[:, :, 0] = .8; a[:, :, 1] = .2
        b = a.copy(); b[:, 1, 1] = .19; b[:, 1, 6] = .01
        before = dict(grade=np.zeros((1, 5), int), calibrated=a, uncalibrated=a)
        after = dict(grade=np.zeros((1, 5), int), calibrated=b, uncalibrated=b)
        gates, _ = guard_metrics(before, after)
        self.assertTrue(gates['supported_nonzero_pass'].item())
        self.assertFalse(gates['thirty_guard_pass'].item())

    def test_targeted_prediction_matches_original_risk_projection(self):
        raw = SimpleNamespace(imputer=Identity(), model=FakeForest())
        risk = SelectedRiskModel({'extra_trees': raw}, ['extra_trees']*5,
            [[.8, .7, .6, .3, .2, .1]]*5, np.zeros((5, 7)), {'calibration_epsilon': 1e-6})
        targets = ['a', 'b', 'c', 'd', 'e']
        bundle = dict(targets=targets, risk_panel_model=risk, raw_panel_model=SimpleNamespace(penalties=None))
        x = np.zeros((3, 1))
        fast = defect_predictions(bundle, x, targets)
        np.testing.assert_allclose(fast['calibrated'], risk.predict_distribution(x), atol=1e-12)
        np.testing.assert_allclose(fast['uncalibrated'], risk.predict_uncalibrated_distribution(x), atol=1e-12)

    def test_search_ranks_feasibility_before_objective(self):
        t = pd.DataFrame(dict(objective=[0., 2., 1.], supported_positive_delta=[0., 0., 0.],
            sensory_violation_count=[0, 0, 1], structural_violation_count=[1, 0, 0]))
        self.assertEqual(rank_candidates(t).tolist(), [1, 2, 0])


if __name__ == '__main__':
    unittest.main()
