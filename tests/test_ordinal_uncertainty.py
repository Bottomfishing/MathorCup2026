import unittest
import numpy as np
from beer_models.ordinal import ranked_probability_score,central_grade_interval


class OrdinalUncertaintyTest(unittest.TestCase):
    def test_perfect_and_opposite_predictions(self):
        p=np.zeros((2,1,7));p[0,0,0]=1;p[1,0,6]=1
        np.testing.assert_allclose(ranked_probability_score(p,[[0],[0]]),[[0],[1]])

    def test_uniform_and_point_intervals(self):
        p=np.ones((1,1,7))/7
        np.testing.assert_allclose(ranked_probability_score(p,[[3]]),[[28/294]])
        lo,hi=central_grade_interval(p);self.assertEqual((lo.item(),hi.item()),(0,6))
        p=np.zeros((1,1,7));p[0,0,3]=1
        lo,hi=central_grade_interval(p);self.assertEqual((lo.item(),hi.item()),(3,3))

    def test_invalid_grade_and_interval_rejected(self):
        p=np.ones((1,1,7))/7
        with self.assertRaises(ValueError):ranked_probability_score(p,[[3.5]])
        with self.assertRaises(ValueError):central_grade_interval(p,1)

    def test_high_level_interval_with_numerical_mass_deficit(self):
        p=np.zeros((1,1,7));p[0,0,6]=1-1e-8
        lo,hi=central_grade_interval(p,1-1e-10)
        self.assertEqual((lo.item(),hi.item()),(6,6))


if __name__=='__main__':unittest.main()
