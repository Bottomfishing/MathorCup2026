import unittest
import numpy as np
from beer_models.final_model import relative_grades, projected_paths, estimator


class FinalModelTests(unittest.TestCase):
    def test_relative_grades_ties_bounds_and_monotonicity(self):
        reference = np.array([[0., 2.], [1., 2.], [2., 2.]])
        actual = relative_grades(reference, np.array([[-10., 2.], [1., 2.], [10., 2.]]))
        np.testing.assert_array_equal(actual, [[0, 3], [3, 3], [6, 3]])

    def test_projected_path_preserves_defects_at_every_step(self):
        source = np.array([-2., 1., 5.])
        anchors = np.array([[10., 9., 8.], [-8., -3., 1.]])
        values, fractions, parents = projected_paths(source, anchors, [0, 2], 4)
        np.testing.assert_array_equal(values[:, [0, 2]], np.tile(source[[0, 2]], (8, 1)))
        np.testing.assert_allclose(values[:, 1], source[1] + fractions * (anchors[parents, 1] - source[1]))

    def test_projection_does_not_hide_missing_source(self):
        with self.assertRaises(ValueError):
            projected_paths([np.nan, 1], [[2, 3]], [0])

    def test_target_scaler_fits_only_training_labels(self):
        config = {'ridge_alpha': 1000., 'extra_trees': {'n_estimators': 5, 'min_samples_leaf': 2, 'max_features': .7}}
        x = np.arange(40).reshape(20, 2)
        y = np.column_stack([np.arange(20), 100 * np.arange(20)])
        model = estimator('extra_trees_equal_targets', config, 1).fit(x[:15], y[:15])
        np.testing.assert_allclose(model.transformer_.mean_, y[:15].mean(axis=0))
        self.assertEqual(model.predict(x[15:]).shape, (5, 2))


if __name__ == '__main__':
    unittest.main()
