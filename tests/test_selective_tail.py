import unittest
import numpy as np
from beer_models.selective_tail import selective_decision, select_policy, candidates


class SelectiveTailTests(unittest.TestCase):
    def test_exact6_reward_differs_from_tail_weighted_absolute_median(self):
        p = np.array([[[.6, 0, 0, 0, 0, .3, .1]]])
        prior = np.array([.8, 0, 0, 0, .15, .04, .01])
        self.assertEqual(selective_decision(p, prior, .25, 0, []).item(), 6)
        self.assertEqual(selective_decision(p, prior, 8, 0, []).item(), 0)
        self.assertEqual(selective_decision(p, prior, .25, 10, []).item(), 0)

    def test_unobserved_training_grade_is_not_invented_and_parent_cap_is_kept(self):
        p = np.array([[[1, 0, 0, 0, 0, 0, 0], [.1, 0, 0, 0, 0, .3, .6]]])
        prior = np.array([.8, 0, 0, 0, .15, .04, .01])
        np.testing.assert_array_equal(selective_decision(p, prior, .25, 0, [(0, 1)]), [[0, 0]])
        prior[6] = 0
        self.assertEqual(selective_decision(p[:, 1:], prior, .25, 0, []).item(), 5)

    def test_failed_tail_gates_force_explicit_baseline_fallback(self):
        p = dict(mae_penalties=[1], false_positive_penalties=[0], point_limits={
            'maximum_raw_mae_increase': .02, 'minimum_each_raw_grade5_grade6_recall': .05,
            'minimum_each_panel_median5_median6_recall': .05,
            'maximum_raw_high_false_positive_rate': .02, 'maximum_panel_high_false_positive_rate': .02})
        q = np.zeros((2, 1, 7)); q[:, 0, 0] = .8; q[:, 0, 6] = .2
        baseline = np.zeros((2, 1), int)
        selected, rows = select_policy({0: baseline}, q, baseline, p)
        self.assertIsNone(selected)
        self.assertFalse(rows[0]['eligible'])
        self.assertEqual(candidates(p), [(1, 0)])


if __name__ == '__main__':
    unittest.main()
