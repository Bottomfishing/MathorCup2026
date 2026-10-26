import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1]/'src'), str(HERE.parents[1]/'scripts')]
from two_stage import BinaryStage, TwoStageHighModel, JointTailForest, decide, apply_gates
from confidence import fit_gates
from beer_models.raw_panel import median_grades, enforce_hierarchy
from beer_models.selective_tail import selective_decision
from run import summarize, success_check, probability_metrics


class TwoStageTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
        self.p['model']['n_estimators'] = 5
        self.p['jobs'] = 1

    def test_balancing_is_undone_in_probability_predictions(self):
        x = np.zeros((20, 1))
        model = BinaryStage(self.p['model'], 7).fit(x, np.full(20, .9), np.full(20, .1))
        self.assertIsNotNone(model.model)
        np.testing.assert_allclose(model.predict(x), .1, atol=1e-12)

    def test_absent_six_has_no_artificial_auc_from_cumulative_roundoff(self):
        q = np.zeros((5, 1, 7)); q[:, 0, 0] = .1; q[:, 0, 1] = .9
        table = probability_metrics({'test': q}, q, ['target'], 0)
        six = table.loc[table.rating_ge.eq(6)].iloc[0]
        self.assertEqual(six.observed_prevalence, 0)
        self.assertTrue(np.isnan(six.weighted_auc) and np.isnan(six.weighted_ap))

    def test_conditional_stage_weights_and_low_distribution_are_preserved(self):
        x = np.zeros((20, 1)); x[0] = np.nan
        q = np.zeros((20, 1, 7)); q[:, :, 0] = .7; q[:, :, 5] = .2; q[:, :, 6] = .1
        model = TwoStageHighModel(self.p, 7).fit(x, q)
        base = np.zeros_like(q); base[:, :, 0] = .6; base[:, :, 1] = .2; base[:, :, 5] = .2
        p = model.predict_distribution(x, base)
        np.testing.assert_allclose(p[:, :, 5:], q[:, :, 5:], atol=1e-12)
        np.testing.assert_allclose(p[:, :, 0]/p[:, :, 1], 3.)
        np.testing.assert_allclose(p.sum(2), 1.)
        np.testing.assert_allclose(model.stages[0][1].prior, 1/3)
        self.assertEqual(model.imputer.statistics_.tolist(), [0.])

    def test_unobserved_high_stage_does_not_create_fake_six_labels(self):
        q = np.zeros((10, 1, 7)); q[:, :, 0] = 1
        model = TwoStageHighModel(self.p, 7).fit(np.zeros((10, 1)), q)
        h, conditional = model.predict_stages(np.ones((3, 1)))
        np.testing.assert_array_equal(h, 0)
        np.testing.assert_array_equal(conditional, 0)
        self.assertEqual(model.stages[0][1].n_exposed_beers, 0)

    def test_old_probability_control_has_exact_same_decisions(self):
        p = np.random.default_rng(7).dirichlet(np.ones(7), size=(30, 3))
        parent = [(0, 1)]
        base = enforce_hierarchy(median_grades(p), parent)
        for lam, mu in [(1, 2), (.25, 0), (8, 4)]:
            np.testing.assert_array_equal(decide(p, base, p.mean((0, 1)), lam, mu, parent),
                selective_decision(p, p.mean((0, 1)), lam, mu, parent))

    def test_confidence_support_and_upward_parent_rule(self):
        q = np.zeros((4, 2, 7)); q[:, :, 0] = .25; q[:, :, 6] = .75
        proposed = np.tile([0, 6], (4, 1))
        gates = fit_gates(q, proposed)
        self.assertFalse(gates['high'][0]); self.assertTrue(gates['high'][1])
        result = apply_gates(np.zeros((4, 2), int), proposed, gates, [(0, 1)])
        np.testing.assert_array_equal(result, 6)
        sparse = fit_gates(q[:2], proposed[:2])
        self.assertFalse(sparse['six'].any())

    def test_joint_forest_undoes_response_scale_and_does_not_invent_six(self):
        p = dict(self.p, model=dict(n_estimators=10, min_samples_leaf=3, max_features=1., bootstrap=True))
        q = np.zeros((20, 1, 7)); q[:, :, 0] = .8; q[:, :, 5] = .2
        model = JointTailForest(p, 7).fit(np.arange(20.)[:, None], q)
        result = model.predict_distribution(np.arange(20.)[:, None], q)
        np.testing.assert_allclose(result, q, atol=1e-12)
        for tree in model.models[0].estimators_:
            leaf = tree.tree_.children_left == -1
            self.assertTrue((tree.tree_.n_node_samples[leaf] >= 3).all())

    def test_full_summary_keeps_original_and_tail_metrics(self):
        q = np.full((6, 2, 7), 1/7)
        d = dict(distribution=q, primary=np.array([True, False]), targets=['a', 'b'],
            ids=pd.DataFrame(dict(beer=range(6), tasting_category_fine=['a']*3+['b']*3)))
        p = dict(self.p, bootstrap_repeats=10)
        grades = {name: np.full((3, 6, 2), 3) for name in ['original_median', 'old_same_selector', 'new_selected']}
        probs = {'original_et': np.broadcast_to(q, (3, *q.shape)), 'two_stage': np.broadcast_to(q, (3, *q.shape))}
        with tempfile.TemporaryDirectory() as temp:
            tables = summarize(d, p, grades, probs, Path(temp))
        np.testing.assert_allclose(tables['metrics.csv'].raw_rating_mae, 12/7)
        self.assertFalse(success_check(tables, p)[0])


if __name__ == '__main__':
    unittest.main()
