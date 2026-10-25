"""Verify the 52-item Q1 view without treating unlabelled threshold proxies as truth."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from beer_audit.audit import sha256, write_json
from complete_modeling import load_q1_artifacts


def verify(source, q1, ordinal, output, smoke=None):
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    for name, digest in summary['source_sha256'].items():
        assert sha256(source / name) == digest
    assert sha256(Path('configs/q1_task2_protocol.json')) == summary['config_sha256']
    assert sha256(Path(summary['config']['scorecard_scope'])) == summary['scope_sha256']
    assert sha256(q1 / 'model.joblib') == summary['parent_q1_model_sha256']
    assert sha256(ordinal / 'model.joblib') == summary['parent_ordinal_model_sha256']
    assert sha256(ordinal / 'oof_predictions.csv') == summary['parent_ordinal_oof_sha256']
    assert sha256(output / 'model.joblib') == summary['model_sha256']
    ids, xf, yf, base, splits = load_q1_artifacts(source, Path('outputs/p0_audit_20261023_final'), q1)
    beers = ids.beer.astype(str).to_numpy(); targets = list(yf); y = yf.to_numpy(float)
    scope = pd.read_csv(output / 'score_semantics.csv').fillna('')
    target_order = scope.target_id.tolist(); primary = scope.loc[scope.acceptance_scope.eq('main_content'), 'target_id'].tolist()
    assert len(target_order) == 52 and len(primary) == 47
    long = pd.read_csv(output / 'scores_all52_long.csv', dtype={'beer': str})
    assert len(long) == 13000 and not long.duplicated(['beer', 'target']).any()
    assert set(long.beer) == set(beers) and set(long.target) == set(target_order)
    assert long.estimated_grade_0_6.between(0, 6).all() and (long.estimated_grade_0_6 % 1 == 0).all()
    assert not long.original_card_score_recovered.any()
    old = pd.read_csv(ordinal / 'oof_predictions.csv', dtype={'beer': str})
    def old_matrix(field):
        return old.pivot(index='beer', columns='target', values=field).loc[beers, targets].to_numpy()
    truth = np.empty((len(beers), 50), int); naive = np.empty_like(truth)
    for train, test in splits:
        # Independent direct counting, rather than calling the grading implementation.
        for j in range(50):
            ref = y[train, j]
            def grades(values):
                u = ((ref[:, None] < values).sum(axis=0)+.5*(ref[:, None] == values).sum(axis=0))/len(ref)
                return np.minimum(6, np.floor(7*u)).astype(int)
            truth[test, j] = grades(y[test, j])
            naive[test, j] = np.quantile(grades(ref), .5, method='inverted_cdf')
    np.testing.assert_array_equal(truth, old_matrix('observed_relative_grade'))
    panel = long.loc[long.panel_target_available]
    def new_matrix(field):
        return panel.pivot(index='beer', columns='target', values=field).loc[beers, targets].to_numpy()
    median = old_matrix('median_relative_grade').astype(int)
    np.testing.assert_array_equal(new_matrix('estimated_grade_0_6'), median)
    np.testing.assert_array_equal(new_matrix('observed_relative_grade'), truth)
    probs = np.stack([old_matrix(f'probability_grade_{k}') for k in range(7)], axis=2)
    for action in range(7):
        expected_selected_loss = np.sum(probs*abs(np.arange(7)-median[:, :, None]), axis=2)
        expected_other_loss = np.sum(probs*abs(np.arange(7)-action), axis=2)
        assert (expected_selected_loss <= expected_other_loss+1e-12).all()
    lower, upper = new_matrix('relative_grade_forecast_lower90'), new_matrix('relative_grade_forecast_upper90')
    assert ((lower <= median) & (median <= upper)).all()
    for i in range(250):
        for j in range(50):
            assert probs[i, j, int(lower[i, j]):int(upper[i, j])+1].sum() >= .9-1e-12
    continuous = pd.read_csv(q1 / 'oof_predictions.csv', dtype={'beer': str})
    expected = continuous.loc[continuous.model.eq('nested_selected')].pivot(index='beer', columns='target', values='prediction').loc[beers, targets]
    np.testing.assert_allclose(new_matrix('standardized_prediction'), expected, atol=1e-12)
    extras = long.loc[~long.panel_target_available]
    assert len(extras) == 500 and set(extras.target) == {'A_acetaldehyde', 'F_acetaldehyde'}
    assert extras.observed_relative_grade.isna().all() and extras.standardized_prediction.isna().all()
    assert extras.relative_grade_0_6.isna().all() and extras.relative_grade_forecast_lower90.isna().all()
    threshold_rows = pd.read_excel(source / '8.xlsx', sheet_name='Sensory thresholds')
    ref = threshold_rows.loc[threshold_rows.field_name_R.str.strip().eq('acetaldehyde')].iloc[0]
    assert ref['unit'] == 'mg/L' and ref.threshold_median == summary['reference_threshold_mg_L'] == 50.
    concentration = 10**xf.acetaldehyde.to_numpy(float)
    assert np.isfinite(concentration).all() and (concentration < 50).all()
    assert extras.estimated_grade_0_6.eq(0).all()
    for target in ['A_acetaldehyde', 'F_acetaldehyde']:
        a = extras.loc[extras.target.eq(target)].set_index('beer').loc[beers]
        np.testing.assert_allclose(a.chemical_log10_coordinate, xf.acetaldehyde, atol=1e-12)
        assert a.score_status.eq('below_reference_threshold_assumed_zero').all()
    for name, selected_targets in [('estimated_grades_all52', target_order), ('estimated_grades_primary47', primary)]:
        actual = pd.read_csv(output / (name+'.csv'), dtype={'beer': str}).set_index('beer').loc[beers, selected_targets]
        expected = long.pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[beers, selected_targets]
        np.testing.assert_array_equal(actual, expected)
    links = []
    for row in scope.loc[scope.parent_formula_cell.ne('')].itertuples():
        parent = scope.loc[scope.template_score_formula_cell.eq(row.parent_formula_cell), 'target_id'].iloc[0]
        links.append((targets.index(parent), targets.index(row.target_id)))
    clipped = median.copy()
    for parent, child in links:
        clipped[:, child] = np.minimum(clipped[:, child], clipped[:, parent])
    actions = {'training_median': naive, 'continuous_then_grade': old_matrix('baseline_relative_grade'),
        'conditional_median': median, 'conditional_mode': old_matrix('mode_relative_grade'),
        'conditional_median_child_clipped_diagnostic': clipped}
    comparisons = pd.read_csv(output / 'model_comparison.csv').set_index(['method', 'scope'])
    diagnostics = pd.read_csv(output / 'hierarchy_diagnostics.csv').set_index('method')
    for name, values in actions.items():
        for label, mask in [('panel_primary45', base['primary']), ('panel_all50', np.ones(50, bool))]:
            error = abs(values[:, mask]-truth[:, mask])
            np.testing.assert_allclose(comparisons.loc[(name, label), ['grade_mae', 'exact_accuracy', 'within_one_accuracy']].to_numpy(float),
                [error.mean(), (error == 0).mean(), (error <= 1).mean()], atol=1e-12)
    for name, values in [('constructed_observed_relative_grades', truth), *actions.items()]:
        violations = np.column_stack([values[:, c] > values[:, p] for p, c in links])
        assert diagnostics.loc[name, 'child_above_parent_count'] == violations.sum()
    sensitivity = pd.read_csv(output / 'threshold_sensitivity.csv')
    for r in sensitivity.itertuples():
        assert r.below_threshold_zero_count == (concentration < r.threshold_mg_L).sum()
        assert r.at_or_above_threshold_intensity_unidentified_count == (concentration >= r.threshold_mg_L).sum()
    old_missing = pd.read_csv(ordinal / 'missingness.csv').query("method == 'conditional_median'").reset_index(drop=True)
    selected_missing = pd.read_csv(output / 'selected_missingness.csv')
    pd.testing.assert_frame_equal(selected_missing[old_missing.columns], old_missing, check_exact=False, atol=1e-12, rtol=0)
    stress = pd.read_csv(output / 'acetaldehyde_missingness.csv')
    old_config = json.loads((ordinal / 'summary.json').read_text(encoding='utf-8'))['config']
    chemical_index = list(xf).index('acetaldehyde')
    for r in stress.itertuples():
        imputed = 0
        for fold, (_, test) in enumerate(splits):
            random = np.random.default_rng(old_config['seed']+1000+fold*100+r.repeat).random((len(test), len(xf.columns)))
            imputed += int((random[:, chemical_index] < r.additional_missing_rate).sum())
        assert r.chemical_inputs_imputed_training_fold_only == imputed and r.proxy_rule_changed_count == 0
    bundle = joblib.load(output / 'model.joblib'); original_bundle = joblib.load(ordinal / 'model.joblib')
    x = xf.to_numpy(float)
    np.testing.assert_allclose(bundle['model'].predict(x), original_bundle['model'].predict(x), atol=1e-12)
    np.testing.assert_allclose(bundle['grade_model'].named_steps['impute'].statistics_, np.nanmedian(x, axis=0))
    cli_rows = None
    if smoke:
        cli = pd.read_csv(smoke / 'scores_all52_long.csv', dtype={'beer': str})
        assert len(cli) == 13000
        filled = bundle['grade_model'].named_steps['impute'].transform(x)
        distributions = bundle['grade_model'].named_steps['model'].predict_proba(filled)
        # Model-generated probabilities have full precision; no CSV median tie here.
        expected = np.column_stack([(np.cumsum(p, axis=1) >= .5).argmax(axis=1) for p in distributions])
        cp = cli.loc[cli.panel_target_available].pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[beers, targets]
        np.testing.assert_array_equal(cp, expected)
        assert cli.loc[~cli.panel_target_available, 'estimated_grade_0_6'].eq(0).all()
        info = json.loads((smoke / 'prediction_info.json').read_text(encoding='utf-8'))
        assert info['targets'] == 52 and info['labelled_panel_targets'] == 50 and info['complete_scorecard_view']
        cli_rows = len(cli)
    result = {'status': 'passed', 'score_rows': 13000, 'primary_items': 47, 'supplementary_items': 5,
        'panel_OOF_rows_verified': 12500, 'proxy_rows_verified_under_reference_threshold_assumption': 500,
        'panel_accuracy_evaluation_items': 45, 'cli_rows_verified': cli_rows,
        'original_absolute_card_scale_identified': False, 'extra_targets_have_panel_labels': False,
        'checks': ['source, parent, scope and model hashes', 'independent fold-reference grade counting',
            'all seven actions expected absolute loss', '50-target OOF decisions and continuous predictions',
            '52/47-item wide tables', '500 threshold scenarios without invented expert labels or probabilities',
            'hierarchy and clipped-decision diagnostics', 'threshold sensitivity',
            'nine retained panel missingness scenarios and nine separate proxy consistency scenarios',
            'unchanged underlying full-data predictor and training-median imputation', 'real default CLI output' if smoke else 'CLI not requested'],
        'interpretation': 'Checks establish implementation and provenance. They do not calibrate original card intensity or validate the threshold assumption against missing panel labels.'}
    write_json(output / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--q1', type=Path, default=Path('outputs/final_20261024/q1'))
    p.add_argument('--ordinal', type=Path, default=Path('outputs/q1_ordinal_delivery_20261024'))
    p.add_argument('--output', type=Path, default=Path('outputs/q1_task2_20261025'))
    p.add_argument('--smoke', type=Path)
    args = p.parse_args()
    verify(args.source, args.q1, args.ordinal, args.output, args.smoke)
