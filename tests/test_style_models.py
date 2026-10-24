import unittest
import numpy as np
from beer_models.style_models import make_style_model, make_stale_model


class StyleModelTests(unittest.TestCase):
    def setUp(self):
        self.config = {'logistic_C': 1., 'logistic_max_iter': 1000, 'n_estimators': 5, 'min_samples_leaf': 2, 'max_features': .7}

    def test_style_imputation_is_training_only_and_handles_missing_inputs(self):
        x = np.array([[1., 2.], [2., np.nan], [3., 4.], [8., 9.], [9., 10.], [10., 11.]])
        y = np.array(['a'] * 3 + ['b'] * 3)
        for name in ['logistic', 'extra_trees']:
            model = make_style_model(name, self.config, 1).fit(x, y)
            np.testing.assert_allclose(model.named_steps['impute'].statistics_, np.nanmedian(x, axis=0))
            self.assertEqual(model.predict([[np.nan, 9.]]).shape, (1,))
            np.testing.assert_allclose(model.predict_proba([[np.nan, 9.]]).sum(axis=1), 1)

    def test_focused_regressor_is_scalar_and_finite(self):
        x = np.arange(20, dtype=float).reshape(10, 2)
        model = make_stale_model(self.config, 1).fit(x, np.arange(10, dtype=float))
        self.assertEqual(model.predict([[np.nan, 3.]]).shape, (1,))
        self.assertTrue(np.isfinite(model.predict([[np.nan, 3.]])).all())


if __name__ == '__main__':
    unittest.main()
