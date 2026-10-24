import unittest
import numpy as np
from beer_models.ordinal import grade_decisions, seven_level_probabilities


class OrdinalTests(unittest.TestCase):
    def test_median_and_mode_optimize_different_losses(self):
        p = np.array([[[.4, 0, 0, .3, 0, 0, .3]]])
        result = grade_decisions(p)
        self.assertEqual(result['conditional_mode'][0, 0], 0)
        self.assertEqual(result['conditional_median'][0, 0], 3)
        expected_loss = np.array([(abs(np.arange(7) - action) * p[0, 0]).sum() for action in range(7)])
        self.assertEqual(int(expected_loss.argmin()), 3)

    def test_invalid_probability_distribution_rejected(self):
        with self.assertRaises(ValueError):
            grade_decisions(np.zeros((1, 1, 7)))
        with self.assertRaises(ValueError):
            grade_decisions(np.ones((1, 1, 6)) / 6)

    def test_absent_training_classes_remain_zero(self):
        class Predictor:
            classes_ = [np.array([0, 3, 6]), np.array([2])]
            def predict_proba(self, x):
                return [np.tile([.2, .3, .5], (len(x), 1)), np.ones((len(x), 1))]
        p = seven_level_probabilities(Predictor(), np.zeros((2, 1)))
        np.testing.assert_allclose(p[:, 0, :], [[.2, 0, 0, .3, 0, 0, .5]] * 2)
        np.testing.assert_allclose(p[:, 1, 2], 1)


if __name__ == '__main__':
    unittest.main()
