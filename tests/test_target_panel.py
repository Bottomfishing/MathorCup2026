"""Tests for group-level probability learning and hierarchy-aware inner decisions."""
import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from beer_models.target_panel import (TargetDistributionForest, SelectedTargetPanel,
                                      select_targets, adoption_decision)


class TargetPanelTests(unittest.TestCase):
    def test_auc_does_not_rank_machine_epsilon_changes_as_predictive_information(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        from q1_target_experiment import probability_tables
        q = np.zeros((6, 1, 7)); q[:, 0, 6] = [0, .1, .2, .4, .5, .8]
        q[:, 0, 0] = 1-q[:, 0, 6]
        p = np.zeros_like(q); p[:, 0, 6] = np.arange(6)*1e-16
        p[:, 0, 0] = 1-p[:, 0, 6]
        result = probability_tables({'x': p}, q, ['a'], 0)
        np.testing.assert_allclose(result.weighted_auc, .5)

    def test_end_to_end_metric_summary_keeps_personal_and_panel_metrics(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        from q1_target_experiment import derive
        q = np.full((6, 2, 7), 1/7)
        data = dict(distribution=q, primary=np.array([True, False]), targets=['a', 'b'],
            ids=pd.DataFrame(dict(beer=range(6), tasting_category_fine=['a']*3+['b']*3)))
        p = dict(repeat_outer_seeds=[7], bootstrap_repeats=10, bootstrap_seed=7)
        probs = {k: q[None].copy() for k in ['shared_et', 'inner_selected']}
        grades = {k: np.full((1, 6, 2), 3) for k in probs}
        tables = derive(data, p, probs, grades)
        result = tables['metrics_by_repeat.csv']
        np.testing.assert_allclose(result.raw_rating_mae, 12/7)
        np.testing.assert_allclose(result.panel_median_mae, 0)
        np.testing.assert_allclose(tables['paired_intervals.csv'].mae_gain, 0)
        self.assertEqual(len(tables['confusion.csv']), 196)

    def test_unique_beer_probability_fit_preserves_mass_and_training_imputation(self):
        x = np.arange(12., dtype=float)[:, None]
        x[0] = np.nan
        q = np.zeros((12, 1, 7))
        q[:6, 0, 0] = .8; q[:6, 0, 6] = .2
        q[6:, 0, 0] = .2; q[6:, 0, 6] = .8
        p = dict(target_forest=dict(n_estimators=8, max_features=1.), parallel_jobs=1)
        m = TargetDistributionForest(3, p, 7).fit(x, q)
        np.testing.assert_allclose(m.imputer.statistics_, [6.])
        probability = m.predict_distribution(np.array([[0.], [11.], [np.nan]]))
        np.testing.assert_allclose(probability.sum(2), 1.)
        self.assertTrue((probability >= 0).all())
        self.assertGreater(probability[1, 0, 6], probability[0, 0, 6])
        np.testing.assert_array_equal(m.imputer.statistics_, [6.])
        for tree in m.models[0].estimators_:
            leaf = tree.tree_.children_left == -1
            self.assertTrue((tree.tree_.n_node_samples[leaf] >= 3).all())

    def test_child_selection_uses_chosen_parent_and_ties_prefer_current_model(self):
        q = np.zeros((2, 2, 7)); q[:, 0, 1] = 1; q[:, 1, 1] = 1
        shared = q.copy(); shared[:, 1] = 0; shared[:, 1, 0] = 1
        high = q.copy(); high[:, 1] = 0; high[:, 1, 6] = 1
        probs = dict(shared_et=shared, training_prior=shared, target_leaf3=high, target_leaf1=high)
        choices, _ = select_targets(probs, q, [(0, 1)])
        self.assertEqual(choices, ['shared_et', 'target_leaf3'])
        model = SelectedTargetPanel({}, q.mean(0), ['training_prior']*2, [(0, 1)])
        self.assertEqual(model.predict(np.zeros((2, 1)))['grade'].tolist(), [[1, 1], [1, 1]])

    def test_better_mae_does_not_claim_tail_solution_and_bad_repeat_blocks_adoption(self):
        p = dict(adoption=dict(minimum_mean_mae_gain=.005, minimum_lower95_gain=0.,
            maximum_raw_high_fpr=.02, maximum_panel_high_fpr=.02, require_each_repeat_gain_positive=True),
            tail_solution=dict(minimum_each_raw_recall5_recall6=.05,
                minimum_each_panel_recall5_recall6=.05, maximum_high_fpr=.02))
        frame = pd.DataFrame([dict(method=m, repeat=r, raw_rating_mae=v, raw_high_fpr=0.,
            panel_high_fpr=0., raw_recall_5=0., raw_recall_6=0., panel_recall_5=0., panel_recall_6=0.)
            for r in range(3) for m, v in [('shared_et', .8), ('inner_selected', .78)]])
        self.assertEqual(adoption_decision(frame, .02, .01, p), (True, False))
        frame.loc[(frame.method == 'inner_selected')&(frame.repeat == 1), 'raw_rating_mae'] = .81
        self.assertFalse(adoption_decision(frame, .01, .001, p)[0])


if __name__ == '__main__':
    unittest.main()
