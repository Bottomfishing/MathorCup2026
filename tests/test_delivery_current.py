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

    def test_q3_generation_v2_is_reported_not_adopted(self):
        if not (d.Q3G2/'generated_profiles.csv').exists():
            self.skipTest('v2 output absent')
        self.assertIn('q3_generation_v2', d.dynamic_tables())
        v2 = read_json(d.Q3G2/'verification.json')
        self.assertTrue(v2['integrity_passed'])
        self.assertFalse(v2['gate_passed'])
        from beer_models.current_runtime import generated_source
        self.assertIn('task4_raw_q3', str(generated_source()))
        table = d.conclusions().set_index('claim')
        self.assertIn('生成器 v2 门槛', table.index)


if __name__ == '__main__':
    unittest.main()
