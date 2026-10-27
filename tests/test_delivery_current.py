"""Contract tests for the 20261028 delivery layer (no packaging side effects)."""
import json
import unittest
import pandas as pd
from beer_models import delivery_current as d
from beer_models.current_runtime import read_json


class DeliveryCurrentTests(unittest.TestCase):
    def test_new_and_history_table_groups(self):
        groups = d.TABLES
        for group in ['q1_decision', 'q2_top3', 'q3_generation']:
            self.assertIn(group, groups)
        self.assertTrue(any(g.startswith('history_') for g in groups))
        for group, (folder, names) in groups.items():
            self.assertTrue(names, group)
            for name in names:
                self.assertTrue((folder/name).exists(), '%s/%s' % (group, name))

    def test_conclusions_read_back_formal_metrics(self):
        table = d.conclusions().set_index('claim')
        metrics = pd.read_csv(d.Q1D/'q1__metrics_summary.csv').set_index('method')
        self.assertAlmostEqual(float(table.loc['决策层 top-k：面板中位 MAE', 'value']),
                               float(metrics.loc['q1_topk_nonforced', 'panel_median_mae']), places=9)
        self.assertAlmostEqual(float(table.loc['ET 整数中位：面板中位 MAE', 'value']),
                               float(metrics.loc['q1_et_integer_median', 'panel_median_mae']), places=9)
        self.assertEqual(int(table.loc['预测 5 分格数/精确/召回', 'value']), 204)
        self.assertEqual(int(table.loc['预测 6 分格数/精确/召回', 'value']), 16)

    def test_q2_rows_match_summary(self):
        self.assertEqual(len(pd.read_csv(d.Q2/'q2__top3_all50_150.csv')), 150)
        summary = read_json(d.Q2/'q2__summary.json')
        self.assertEqual(summary['n_targets'], 50)
        communities = pd.read_csv(d.Q2/'q2__network_communities.csv')
        self.assertEqual(communities.community_id.nunique(), summary['graph']['communities'])
        self.assertEqual(int(communities.community_size.max()), summary['graph']['largest_community'])

    def test_q3_gate_and_integrity_separated(self):
        state = d._gate_state()
        self.assertTrue(state['integrity_passed'])
        state_json = read_json(d.Q3G/'verification.json')
        self.assertEqual(state['gate_passed'], state_json['gate_passed'])
        self.assertEqual(state['replacement_gate_passed'], state_json['replacement_gate_passed'])
        safety = pd.read_csv(d.Q3G/'defect_safety.csv')
        self.assertTrue(safety.passed.all())

    def test_unresolved_keeps_known_limits(self):
        issues = set(d.unresolved()['issue'])
        for expected in ['个人 5/6 分精确识别', '生成替换门槛', 'Stale hops 缺陷', '酿造工艺可行性']:
            self.assertIn(expected, issues)

    def test_generated_source_respects_gate(self):
        from beer_models.current_runtime import generated_source
        source = generated_source()
        data = read_json(source/'verification.json')
        if 'replacement_gate_passed' in data:
            self.assertTrue(data['integrity_passed'] and data['replacement_gate_passed'])
        else:
            self.assertEqual(data['status'], 'passed')

    def test_q1_panelmean_reported_not_adopted(self):
        if not (d.Q1P/'summary.json').exists():
            self.skipTest('panel-mean output absent')
        self.assertIn('q1_panelmean', d.dynamic_tables())
        info = read_json(d.Q1P/'summary.json')
        self.assertTrue(str(info['adoption_recommendation']).startswith('do_not_adopt'))
        table = d.conclusions().set_index('claim')
        self.assertIn('面板均值直接回归 MAE', table.index)
        metrics = pd.read_csv(d.Q1P/'q1__panel_mean_metrics.csv')
        expected = metrics[(metrics.method == 'extra_trees_direct') &
                           (metrics.scope == 'pooled_3repeats')].panel_mean_mae.iloc[0]
        self.assertAlmostEqual(float(table.loc['面板均值直接回归 MAE', 'value']),
                               float(expected), places=9)
        self.assertLess(float(expected), 0.4622338)
        issues = set(d.unresolved()['issue'])
        self.assertIn('面板均值直接回归', issues)

    def test_q2_sensitivity_is_reported(self):
        if not (d.Q2S/'summary.json').exists():
            self.skipTest('sensitivity output absent')
        self.assertIn('q2_sensitivity', d.dynamic_tables())
        verdict = read_json(d.Q2S/'summary.json')['verdict']
        self.assertTrue(verdict['threshold_stable'])
        self.assertTrue(verdict['resolution_stable'])
        table = d.conclusions().set_index('claim')
        self.assertEqual(str(table.loc['参数敏感性', 'value']), '阈值不敏感、分辨率稳健')

    def test_q3_generation_v1_is_reported_not_adopted(self):
        v1 = read_json(d.Q3G/'verification.json')
        self.assertTrue(v1['integrity_passed'])
        self.assertFalse(v1['gate_passed'])
        table = d.conclusions().set_index('claim')
        self.assertIn('生成器 v2 门槛', table.index)

    def test_q3_generation_v3_is_adopted_after_gate(self):
        if not (d.Q3G3/'generated_profiles.csv').exists():
            self.skipTest('v3 output absent')
        v3 = read_json(d.Q3G3/'verification.json')
        self.assertTrue(v3['integrity_passed'])
        self.assertTrue(v3['replacement_gate_passed'])
        groups = d.dynamic_tables()
        self.assertEqual(groups['q3_generation'][0], d.Q3G3)
        from beer_models.current_runtime import generated_source
        self.assertIn('q3_copula_v3', str(generated_source()))
        table = d.conclusions().set_index('claim')
        self.assertIn('生成器 v3 门槛', table.index)
        gates = d._gate_state(d.Q3G3)
        expected = '通过 %d/%d' % (gates['gate_checks']-len(gates['failures']),
                                   gates['gate_checks'])
        self.assertEqual(str(table.loc['替换门槛状态', 'value']), expected)


if __name__ == '__main__':
    unittest.main()
