import unittest
import numpy as np
from beer_models.reliability import permute_within_groups, monte_carlo_upper_p, benjamini_yekutieli


class ReliabilityTests(unittest.TestCase):
    def test_group_permutation_preserves_membership_and_bijection(self):
        groups = np.array(['a', 'b', 'a', 'c', 'b'])
        result = permute_within_groups(groups, np.random.default_rng(2))
        np.testing.assert_array_equal(groups[result], groups)
        np.testing.assert_array_equal(np.sort(result), np.arange(len(groups)))
        self.assertEqual(result[3], 3)

    def test_monte_carlo_correction_never_returns_zero(self):
        self.assertEqual(monte_carlo_upper_p(10., np.array([1., 2., 3.])), .25)
        self.assertEqual(monte_carlo_upper_p(0., np.array([1., 2., 3.])), 1)

    def test_BY_adjustment_preserves_order_and_accounts_for_dependence(self):
        p = np.array([.01, .04, .03])
        q = benjamini_yekutieli(p)
        np.testing.assert_allclose(q, [.055, .07333333333333333, .07333333333333333])
        self.assertTrue((q >= p).all())
        with self.assertRaises(ValueError):
            benjamini_yekutieli([np.nan])


if __name__ == '__main__':
    unittest.main()
