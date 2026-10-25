import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from beer_models.predict import predict_frame
from beer_models.scorecard import below_threshold_grade, predict_scorecard_frame


class ScorecardTests(unittest.TestCase):
    def setUp(self):
        x = np.array([[0., 1.], [1., 2.], [1.2, 4.], [.5, 3.]])
        y = np.column_stack([x[:, 0]+x[:, 1], x[:, 1]-x[:, 0]])
        self.bundle = {'features': ['acetaldehyde', 'b'], 'targets': ['p1', 'p2'],
            'primary': np.array([True, False]), 'reference_y': y,
            'model': Pipeline([('impute', SimpleImputer(strategy='median')), ('model', Ridge())]).fit(x, y),
            'grade_model': Pipeline([('impute', SimpleImputer(strategy='median')),
                ('model', ExtraTreesClassifier(n_estimators=5, random_state=1))]).fit(x, [[0, 6], [2, 4], [6, 0], [4, 2]]),
            'grade_method': 'direct_relative_grade_conditional_median',
            'scorecard_scope': [{'target_id': t, 'acceptance_scope': s} for t, s in
                [('p1', 'main_content'), ('p2', 'supplementary'), ('A_acetaldehyde', 'main_content'), ('F_acetaldehyde', 'main_content')]],
            'scorecard_protocol': {'chemical_field': 'acetaldehyde', 'reference_threshold_mg_L': 50.,
                                  'extra_targets': ['A_acetaldehyde', 'F_acetaldehyde']}}

    def test_threshold_is_not_a_seven_level_calibration(self):
        g = below_threshold_grade(np.log10([1., 49.9, 50., 1000.]), 50.)
        np.testing.assert_array_equal(g[:2], [0, 0])
        self.assertTrue(np.isnan(g[2:]).all())
        for threshold in [0., -1., np.nan]:
            with self.assertRaises(ValueError):
                below_threshold_grade([1.], threshold)

    def test_panel_predictions_preserved_and_proxy_labels_not_fabricated(self):
        f = pd.DataFrame({'beer': ['001', '002'], 'b': [2., 3.], 'acetaldehyde': [1., .5]})
        old, _, _ = predict_frame(f, self.bundle)
        long, primary, all_items, _ = predict_scorecard_frame(f, self.bundle)
        panel = long.loc[long.panel_target_available]
        np.testing.assert_array_equal(panel.relative_grade_0_6, old.relative_grade_0_6)
        np.testing.assert_allclose(panel.standardized_prediction, old.standardized_prediction)
        extra = long.loc[~long.panel_target_available]
        self.assertTrue(extra.estimated_grade_0_6.eq(0).all())
        self.assertTrue(extra.standardized_prediction.isna().all())
        self.assertTrue(extra.relative_grade_0_6.isna().all())
        self.assertTrue(extra.relative_grade_forecast_lower90.isna().all())
        self.assertFalse(long.original_card_score_recovered.any())
        self.assertEqual(primary.columns.tolist(), ['beer', 'p1', 'A_acetaldehyde', 'F_acetaldehyde'])
        self.assertEqual(all_items.shape, (2, 5))
        self.assertEqual(primary.beer.tolist(), ['001', '002'])

    def test_missing_chemical_uses_fitted_median_and_discloses_it(self):
        f = pd.DataFrame({'beer': ['x'], 'b': [2.]})
        long, _, _, _ = predict_scorecard_frame(f, self.bundle)
        extra = long.loc[~long.panel_target_available]
        self.assertTrue(extra.chemical_value_imputed.all())
        np.testing.assert_allclose(extra.chemical_log10_coordinate, .75)

    def test_above_threshold_returns_explicitly_unidentified_point_grade(self):
        f = pd.DataFrame({'beer': ['x'], 'acetaldehyde': [2.], 'b': [2.]})
        long, primary, _, _ = predict_scorecard_frame(f, self.bundle)
        extra = long.loc[~long.panel_target_available]
        self.assertTrue(extra.estimated_grade_0_6.isna().all())
        self.assertTrue(extra.score_status.eq('at_or_above_threshold_intensity_unidentified').all())
        self.assertTrue(primary[['A_acetaldehyde', 'F_acetaldehyde']].isna().all().all())


if __name__ == '__main__':
    unittest.main()
