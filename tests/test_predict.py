import unittest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from beer_models.predict import predict_frame


class PredictTests(unittest.TestCase):
    def setUp(self):
        x = np.array([[1., 2.], [2., 4.], [4., 3.], [3., 1.]])
        y = np.column_stack([x[:, 0] + 2 * x[:, 1], 3 * x[:, 0] - x[:, 1]])
        self.bundle = {'features': ['a', 'b'], 'targets': ['t1', 't2'], 'primary': np.array([True, False]),
                       'reference_y': y, 'model': make_pipeline(SimpleImputer(strategy='median'), Ridge()).fit(x, y)}

    def test_input_order_and_string_ids_preserved(self):
        frame = pd.DataFrame({'beer': ['001', '002'], 'b': [2., 1.], 'extra': ['unused', 'unused'], 'a': [1., 3.]})
        long, wide, samples = predict_frame(frame, self.bundle)
        expected = self.bundle['model'].predict([[1., 2.], [3., 1.]])
        np.testing.assert_allclose(long.standardized_prediction.to_numpy().reshape(2, 2), expected)
        self.assertEqual(wide.beer.tolist(), ['001', '002'])
        self.assertEqual(wide.columns.tolist(), ['beer', 't1'])
        np.testing.assert_array_equal(samples.observed_feature_count, [2, 2])

    def test_missing_columns_use_fitted_imputer(self):
        long, _, samples = predict_frame(pd.DataFrame({'beer': ['x'], 'b': [4.]}), self.bundle)
        expected = self.bundle['model'].predict([[np.nan, 4.]])
        np.testing.assert_allclose(long.standardized_prediction, expected[0])
        self.assertEqual(samples.missing_feature_fraction.iloc[0], .5)

    def test_no_observed_features_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No observed'):
            predict_frame(pd.DataFrame({'beer': ['x'], 'a': [np.nan]}), self.bundle)

    def test_bad_values_and_duplicate_ids_rejected(self):
        for frame in [pd.DataFrame({'a': ['bad'], 'b': [1]}), pd.DataFrame({'a': [np.inf], 'b': [1]}),
                      pd.DataFrame({'beer': ['x', 'x'], 'a': [1, 2]}), pd.DataFrame({'beer': [' '], 'a': [1]})]:
            with self.assertRaises(ValueError):
                predict_frame(frame, self.bundle)

    def test_ids_generated_when_absent(self):
        long, wide, _ = predict_frame(pd.DataFrame({'a': [1.], 'b': [2.]}), self.bundle)
        self.assertEqual(wide.beer.tolist(), ['row_000001'])
        self.assertTrue(long.relative_grade_0_6.between(0, 6).all())

    def test_direct_grade_model_preserves_continuous_predictions(self):
        grade_model = make_pipeline(SimpleImputer(strategy='median'), ExtraTreesClassifier(n_estimators=5, random_state=1))
        x = np.array([[1., 2.], [2., 4.], [4., 3.], [3., 1.]])
        grade_model.fit(x, np.array([[0, 6], [0, 6], [6, 0], [6, 0]]))
        # Prediction expects named stages, as in the saved deployment bundle.
        from sklearn.pipeline import Pipeline
        named = Pipeline([('impute', grade_model.steps[0][1]), ('model', grade_model.steps[1][1])])
        bundle = dict(self.bundle, grade_model=named, grade_method='direct_relative_grade_conditional_median')
        frame = pd.DataFrame({'a': [1.], 'b': [2.]})
        old, _, _ = predict_frame(frame, self.bundle)
        new, _, _ = predict_frame(frame, bundle)
        np.testing.assert_allclose(old.standardized_prediction, new.standardized_prediction)
        np.testing.assert_array_equal(new.relative_grade_0_6, [0, 6])
        self.assertTrue((new.relative_grade_forecast_lower90<=new.relative_grade_0_6).all())
        self.assertTrue((new.relative_grade_forecast_upper90>=new.relative_grade_0_6).all())
        self.assertTrue(new.relative_grade_forecast_lower90.between(0,6).all())
        self.assertTrue(new.relative_grade_forecast_upper90.between(0,6).all())
        self.assertTrue(new.grade_max_class_probability.between(0,1).all())
        self.assertFalse(new.grade_probability_calibrated.any())
        self.assertNotIn('relative_grade_forecast_lower90',old)

    def test_explicit_targetwise_candidate_keeps_continuous_model(self):
        from sklearn.pipeline import Pipeline
        from beer_models.ordered_models import CumulativeOrdinalClassifier, SelectedGradeModel
        x = np.array([[1., 2.], [2., 4.], [4., 3.], [3., 1.]])
        models = []
        for grades in [[0, 6], [6, 0]]:
            models.append(Pipeline([('impute', SimpleImputer()),
                ('model', CumulativeOrdinalClassifier())]).fit(x, np.tile(grades, (len(x), 1))))
        bundle = dict(self.bundle, selected_grade_model=SelectedGradeModel(models, [1, 0]),
            grade_method='explicit_task2_candidate')
        frame = pd.DataFrame({'a': [1.], 'b': [2.]})
        old, _, _ = predict_frame(frame, self.bundle)
        new, _, _ = predict_frame(frame, bundle)
        np.testing.assert_allclose(old.standardized_prediction, new.standardized_prediction)
        np.testing.assert_array_equal(new.relative_grade_0_6, [6, 6])
        self.assertIn('relative_grade_forecast_lower90', new)


if __name__ == '__main__':
    unittest.main()
