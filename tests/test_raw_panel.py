"""Raw-grade weighting, decision loss and beer-level fit boundaries."""
import unittest
import numpy as np
import pandas as pd

from beer_models.raw_panel import (rating_weights, rating_distribution, median_grades,
    rating_loss, round_grade, enforce_hierarchy, RawPanelModel, predict_raw_frame)


class FixedRawModel:
    def predict(self, x):
        values = np.tile([2, 1], (len(x), 1))
        return dict(grade=values, unconstrained_grade=values, raw_median_estimate=values.astype(float))


class RawPanelTests(unittest.TestCase):
    def test_repeated_taster_does_not_dominate_a_beer(self):
        raw = pd.DataFrame({'beer': [10, 10, 10, 20], 'taster': [1, 1, 2, 3]})
        weights = rating_weights(raw)
        np.testing.assert_allclose(weights, [.25, .25, .5, 1])
        q = rating_distribution([0, 0, 0, 1], np.array([[0], [0], [6], [3]]), weights, 2)
        np.testing.assert_allclose(q[0, 0], [.5, 0, 0, 0, 0, 0, .5])
        self.assertEqual(median_grades(q).tolist(), [[0], [3]])

    def test_median_action_minimizes_weighted_original_absolute_error(self):
        q = np.array([[[.1, 0, .5, 0, 0, .4, 0], [.5, 0, 0, 0, 0, 0, .5]]])
        action = median_grades(q)
        self.assertEqual(action.tolist(), [[2, 0]])
        loss = rating_loss(q, action)
        for grade in range(7):
            self.assertTrue((loss <= rating_loss(q, np.full((1, 2), grade)) + 1e-12).all())
        self.assertEqual(round_grade([-.1, .5, 1.5, 5.8, 6.2]).tolist(), [0, 0, 1, 6, 6])

    def test_invalid_weight_totals_and_scores_fail(self):
        for scores, weights in [([[1], [2]], [.5, .5]), ([[1.2], [2]], [1, 1])]:
            with self.assertRaises(ValueError):
                rating_distribution([0, 1], np.array(scores), weights, 2)

    def test_imputer_is_fitted_on_unique_beers_not_rating_replicates(self):
        x = np.array([[0.], [100.], [200.]])
        q = np.zeros((3, 2, 7)); q[:, :, 0] = 1
        model = RawPanelModel('training_median', {}, 1, []).fit(x, [0, 0, 1, 2],
            np.zeros((4, 2), int), [.5, .5, 1, 1], q)
        self.assertEqual(model.imputer.statistics_.tolist(), [100.])

    def test_leaf_minimum_is_beer_weight_despite_many_repeats(self):
        x = np.arange(12.)[:, None]
        positions = np.r_[np.zeros(20, int), np.arange(1, 12)]
        weights = np.r_[np.full(20, 1/20), np.ones(11)]
        scores = np.column_stack([positions % 7, positions % 7])
        q = rating_distribution(positions, scores, weights, 12)
        p = {'minimum_leaf_beer_weight': 3., 'parallel_jobs': 1,
             'extra_trees': {'n_estimators': 5, 'max_features': 1.}}
        model = RawPanelModel('extra_trees', p, 1, [(0, 1)]).fit(x, positions, scores, weights, q)
        for tree in model.model.estimators_:
            leaf = tree.tree_.children_left == -1
            self.assertTrue((tree.tree_.weighted_n_node_samples[leaf] >= 3 - 1e-12).all())

    def test_hierarchy_caps_children_without_changing_parents(self):
        original = np.array([[2, 5], [0, 1]])
        np.testing.assert_array_equal(enforce_hierarchy(original, [(0, 1)]), [[2, 2], [0, 0]])
        np.testing.assert_array_equal(original, [[2, 5], [0, 1]])

    def test_raw_prediction_identifies_unlabelled_items_and_rejects_empty_chemistry(self):
        scope = [{'target_id': t, 'acceptance_scope': 'main_content'} for t in ['a', 'b', 'u', 'v']]
        bundle = dict(features=['x'], targets=['a', 'b'], raw_panel_model=FixedRawModel(),
            scorecard_scope=scope, protocol={'unlabelled_card_targets': ['u', 'v']})
        long, _, _, _ = predict_raw_frame(pd.DataFrame({'beer': ['one'], 'x': [1]}), bundle)
        self.assertEqual(len(long), 4)
        self.assertTrue(long.loc[~long.panel_target_available, 'estimated_grade_0_6'].isna().all())
        self.assertNotIn('relative_grade_0_6', long)
        with self.assertRaises(ValueError):
            predict_raw_frame(pd.DataFrame({'beer': ['one']}), bundle)


if __name__ == '__main__':
    unittest.main()
