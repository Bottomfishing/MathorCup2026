"""End-to-end test: the runtime bundle carries and applies the official Q1 rule.

The synthetic bundle replaces the fitted ExtraTrees only; the decision code under
test is the production implementation (``beer_models.q1_decision.runtime_decision``).
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from beer_models.current_runtime import ROOT, q1_decision_bundle
from beer_models.inference import run
from beer_models.q1_decision import runtime_decision, runtime_threshold_decision

TARGETS = ['A_a', 'A_b', 'F_c']


class _FakeEstimator:
    def __init__(self, probabilities):
        self._probabilities = probabilities
        self.classes_ = [np.arange(7) for _ in TARGETS]

    def predict_proba(self, values):
        return [np.resize(part, (len(values), 7)) for part in self._probabilities]


class _FakeImputer:
    def transform(self, values):
        return values


class _FakeRawPanel:
    def __init__(self, probabilities):
        self.imputer = _FakeImputer()
        self.model = _FakeEstimator(probabilities)
        self._probabilities = probabilities

    def predict(self, values):
        grid = np.argmax(np.stack([np.resize(part, (len(values), 7))
                                   for part in self._probabilities], axis=1), axis=2)
        return {'grade': grid, 'raw_median_estimate': grid.astype(float)}


class _FakePointModel:
    def __init__(self, raw_panel):
        self.model = raw_panel

    def predict(self, values):
        return self.model.predict(values)


def _probabilities():
    low = np.array([[0.05, 0.7, 0.2, 0.05, 0.0, 0.0, 0.0],
                    [0.05, 0.7, 0.2, 0.05, 0.0, 0.0, 0.0]])
    high = np.array([[0.0, 0.05, 0.05, 0.1, 0.1, 0.3, 0.4],
                     [0.0, 0.05, 0.05, 0.1, 0.1, 0.5, 0.2]])
    mid = np.array([[0.0, 0.0, 0.1, 0.2, 0.3, 0.2, 0.2],
                    [0.4, 0.3, 0.2, 0.05, 0.05, 0.0, 0.0]])
    return [low, high, mid]


def _bundle():
    raw = _FakeRawPanel(_probabilities())
    panel_median = np.array([[0, 5, 6], [4, 5, 2]])
    decision = q1_decision_bundle(panel_median, TARGETS, [[1, 2]])
    scope = [{'target_id': target, 'acceptance_scope': 'main_content'} for target in TARGETS]
    return dict(bundle_kind='r2_raw_panel', selected_model='synthetic_test_bundle',
                raw_panel_model=_FakePointModel(raw), features=['c1', 'c2'],
                targets=TARGETS, protocol={'unlabelled_card_targets': []},
                scorecard_scope=scope, score_status='synthetic_test_bundle',
                point_semantics='integer_conditional_median_then_non_forced_high_grade_topk_with_parent_cap',
                grade_method=decision['rule'], q1_decision=decision,
                raw_source_sha256='synthetic', qualified_on_historical_grouped_folds=True,
                tail_point_qualified_on_historical_folds=False,
                point_deployment_status='synthetic', risk_provenance='synthetic')


class RuntimeDecisionBundleTest(unittest.TestCase):
    def test_bundle_rates_match_observed_panel(self):
        decision = q1_decision_bundle(np.array([[0, 5, 6], [4, 5, 2]]), TARGETS, [[1, 2]])
        np.testing.assert_allclose(decision['rate_ge5'], [0.0, 1.0, 0.5])
        np.testing.assert_allclose(decision['rate_6'], [0.0, 0.0, 0.5])
        self.assertEqual(decision['parent_child_pairs'], [[1, 2]])
        self.assertEqual(decision['implementation'], 'beer_models.q1_decision.runtime_decision')
        expected = hashlib.sha256(
            (ROOT / 'src/beer_models/q1_decision.py').read_bytes()).hexdigest()
        self.assertEqual(decision['implementation_sha256'], expected)

    def test_predict_output_is_the_frozen_official_rule(self):
        bundle = _bundle()
        frame = pd.DataFrame({'beer': ['b1', 'b2'], 'c1': [0.1, 0.2], 'c2': [0.3, 0.4]})
        expected = runtime_decision(
            np.stack([p[:, 5:].sum(axis=1) for p in _probabilities()], axis=1),
            np.stack([p[:, 6] for p in _probabilities()], axis=1),
            np.argmax(np.stack(_probabilities(), axis=1), axis=2).astype(float),
            np.asarray(bundle['q1_decision']['rate_ge5']),
            np.asarray(bundle['q1_decision']['rate_6']), TARGETS, [1], [2])
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model_path = folder / 'q1.joblib'
            joblib.dump(bundle, model_path)
            input_path = folder / 'input.csv'
            frame.to_csv(input_path, index=False)
            output = folder / 'predict'
            run(model_path, input_path, output, legacy_decision=True)
            long = pd.read_csv(output / 'scores_all52_long.csv')
            wide = pd.read_csv(output / 'estimated_grades_all52.csv')
            info = json.loads((output / 'prediction_info.json').read_text(encoding='utf-8'))
        grid = long.pivot(index='beer', columns='target',
                          values='estimated_grade_0_6').loc[['b1', 'b2'], TARGETS]
        np.testing.assert_array_equal(grid.to_numpy(), expected)
        self.assertEqual(wide[['beer', *TARGETS]].shape, (2, 4))
        legacy = long.pivot(index='beer', columns='target',
                            values='grade_legacy_integer_median').loc[['b1', 'b2'], TARGETS]
        np.testing.assert_array_equal(legacy.to_numpy(),
                                      np.argmax(np.stack(_probabilities(), axis=1), axis=2))
        self.assertEqual(info['grade_method'], bundle['q1_decision']['rule'])
        self.assertEqual(info['decision_rule']['implementation_sha256'],
                         bundle['q1_decision']['implementation_sha256'])
        self.assertEqual(info['official_grade_column'], 'estimated_grade_0_6')
        self.assertEqual(info['legacy_grade_column'], 'grade_legacy_integer_median')
        self.assertIn('probability_rating_ge5', long.columns)
        self.assertIn('probability_rating_6', long.columns)
        parent, child = bundle['q1_decision']['parent_child_pairs'][0]
        self.assertTrue((grid.iloc[:, child].to_numpy() <= grid.iloc[:, parent].to_numpy()).all())

    def test_fixed_threshold_bundle_runs_without_legacy_flag(self):
        bundle = _bundle()
        decision = bundle['q1_decision']
        decision = dict(decision, rule='fixed_threshold_quantile',
                        tau_ge5=[0.5, 0.0, 1.0], tau_6=[1.0, 0.0, 0.05],
                        implementation='beer_models.q1_decision.runtime_threshold_decision')
        bundle['q1_decision'] = decision
        bundle['grade_method'] = decision['rule']
        frame = pd.DataFrame({'beer': ['b1', 'b2'], 'c1': [0.1, 0.2], 'c2': [0.3, 0.4]})
        expected = runtime_threshold_decision(
            np.stack([p[:, 5:].sum(axis=1) for p in _probabilities()], axis=1),
            np.stack([p[:, 6] for p in _probabilities()], axis=1),
            np.argmax(np.stack(_probabilities(), axis=1), axis=2).astype(float),
            np.asarray(decision['tau_ge5']), np.asarray(decision['tau_6']),
            TARGETS, [1], [2])
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model_path = folder / 'q1.joblib'
            joblib.dump(bundle, model_path)
            input_path = folder / 'input.csv'
            frame.to_csv(input_path, index=False)
            run(model_path, input_path, folder / 'predict')
            long = pd.read_csv(folder / 'predict' / 'scores_all52_long.csv')
        grid = long.pivot(index='beer', columns='target',
                          values='estimated_grade_0_6').loc[['b1', 'b2'], TARGETS]
        np.testing.assert_array_equal(grid.to_numpy(), expected)

    def test_bundle_without_rule_is_refused_unless_legacy_requested(self):
        bundle = _bundle()
        bundle.pop('q1_decision')
        bundle['grade_method'] = 'r2_raw_panel_median'
        frame = pd.DataFrame({'beer': ['b1'], 'c1': [0.1], 'c2': [0.2]})
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            model_path = folder / 'q1.joblib'
            joblib.dump(bundle, model_path)
            input_path = folder / 'input.csv'
            frame.to_csv(input_path, index=False)
            with self.assertRaises(ValueError):
                run(model_path, input_path, folder / 'refused')
            self.assertFalse((folder / 'refused').exists())
            run(model_path, input_path, folder / 'predict', legacy_decision=True)
            long = pd.read_csv(folder / 'predict' / 'scores_all52_long.csv')
            self.assertIn('scores_all52_long.csv',
                          [path.name for path in (folder / 'predict').iterdir()])
        self.assertNotIn('grade_legacy_integer_median', long.columns)
        self.assertNotIn('probability_rating_ge5', long.columns)


if __name__ == '__main__':
    unittest.main()
