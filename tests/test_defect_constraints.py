import unittest
import numpy as np
from beer_models.defect_constraints import project_defects, defect_aware_box, reference_zero_bound


class DefectConstraintTests(unittest.TestCase):
    def test_lower_defect_does_not_break_target_support(self):
        # A safer DMS value below the observed target minimum must be allowed.
        candidates = np.array([[.8, 2.], [1.3, 2.], [.8, .5], [3., 2.]])
        np.testing.assert_array_equal(defect_aware_box(candidates, [1., 1.], [2., 3.], [0]),
                                      [True, True, False, False])

    def test_both_projection_policies_never_increase_a_defect(self):
        source = np.array([.8, 2., -2.])
        raw = np.array([[1.4, 3., -1.], [.5, 4., -3.]])
        for policy in ['preserve', 'reduce']:
            result = project_defects(raw, source, [0, 2], policy)
            self.assertTrue((result[:, [0, 2]] <= source[[0, 2]]).all())
            np.testing.assert_array_equal(result[:, 1], raw[:, 1])
        np.testing.assert_array_equal(raw, [[1.4, 3., -1.], [.5, 4., -3.]])

    def test_reference_zero_is_not_general_imputation(self):
        self.assertEqual(reference_zero_bound(np.nan, 'X2.3.butanedione', 0.)[0], 0.)
        self.assertEqual(reference_zero_bound(.2, 'X2.3.butanedione', 0.)[0], .2)
        for field, ref in [('DMS', 0.), ('X2.3.butanedione', np.nan), ('X2.3.butanedione', .1)]:
            with self.assertRaises(ValueError):
                reference_zero_bound(np.nan, field, ref)
        with self.assertRaises(ValueError):
            project_defects([[1., 2.]], [np.nan, 2.], [0], 'preserve')

    def test_zero_cap_and_nonincrease_imply_threshold_severity_nonincrease(self):
        # Covers unknown nonnegative sources, arbitrary positive thresholds,
        # and unchanged/decreased observed concentrations.
        for before in [0., .001, .1, 1., 10.]:
            for after in [0., before / 2, before]:
                for threshold in [.0001, .09, .2, 19.]:
                    self.assertLessEqual(max(after / threshold - 1, 0), max(before / threshold - 1, 0))


if __name__ == '__main__':
    unittest.main()
