import unittest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from beer_models.ordered_models import (cumulative_probabilities, CumulativeOrdinalClassifier,
                                       training_choices, SelectedGradeModel)
from beer_models.ordinal import seven_level_probabilities


class OrderedModelsTests(unittest.TestCase):
    def test_crossing_thresholds_are_projected_not_sorted(self):
        p = cumulative_probabilities([[[.8, .3, .6, .2, .1, .05]]])
        np.testing.assert_allclose(p, [[[.2, .35, 0, .25, .1, .05, .05]]])
        self.assertTrue((p >= 0).all())
        np.testing.assert_allclose(p.sum(axis=2), 1)

    def test_constant_and_missing_grade_levels(self):
        x = np.arange(20.).reshape(10, 2)
        y = np.column_stack([np.full(10, 3), np.where(x[:, 0] < 10, 0, 6)])
        model = CumulativeOrdinalClassifier(C=.1).fit(x, y)
        p = seven_level_probabilities(model, x)
        np.testing.assert_allclose(p[:, 0, 3], 1)
        np.testing.assert_allclose(p.sum(axis=2), 1)
        self.assertTrue((p >= 0).all())
        self.assertTrue(model.predict(x).min() >= 0)
        self.assertTrue(model.predict(x).max() <= 6)

    def test_imputation_is_kept_from_training(self):
        x = np.array([[1., 2.], [3., 4.], [np.nan, 6.], [5., 8.]])
        model = Pipeline([('impute', SimpleImputer()), ('model', CumulativeOrdinalClassifier())])
        model.fit(x, np.array([[0], [2], [4], [6]]))
        before = model.named_steps['impute'].statistics_.copy()
        model.predict([[np.nan, 100.]])
        np.testing.assert_array_equal(model.named_steps['impute'].statistics_, before)

    def test_targetwise_training_selection_and_ties(self):
        truth = np.array([[0, 6], [1, 5]])
        predictions = np.array([[[0, 2], [1, 2]], [[0, 6], [1, 5]]])
        choices, errors = training_choices(predictions, truth)
        np.testing.assert_array_equal(choices, [0, 1])
        np.testing.assert_array_equal(errors[:, 0], [0, 0])
        x = np.arange(4.).reshape(2, 2)
        a = CumulativeOrdinalClassifier().fit(x, np.array([[0, 2], [0, 2]]))
        b = CumulativeOrdinalClassifier().fit(x, np.array([[1, 6], [1, 6]]))
        selected = SelectedGradeModel([a, b], choices)
        np.testing.assert_array_equal(selected.predict(x), [[0, 6], [0, 6]])

    def test_invalid_targets_are_rejected(self):
        for target in [[[7]], [[.5]], [[np.nan]]]:
            with self.assertRaises(ValueError):
                CumulativeOrdinalClassifier().fit([[1]], target)
