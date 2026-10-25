"""Recount task2 outcomes, reconstruct training-only selection and replay outer fits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256, write_json
from beer_models.baseline import load_data, make_splits
from beer_models.explain import stratified_draw
from beer_models.ordinal import seven_level_probabilities
from beer_models.ordered_models import make_ordered_candidate


def count_grades(reference, values):
    result = np.empty(values.shape, int)
    for j in range(values.shape[1]):
        u = np.array([(np.sum(reference[:, j] < v) + .5 * np.sum(reference[:, j] == v)) / len(reference)
                      for v in values[:, j]])
        result[:, j] = np.minimum(6, np.floor(7 * u)).astype(int)
    return result


def verify(source, output, smoke=None):
    p = json.loads(Path('configs/task2_ordinal_protocol_20261025.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(json.dumps(p, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    summary = json.loads((output / 'summary.json').read_text())
    assert digest == summary['protocol_content_sha256']
    assert json.loads((output / 'protocol.json').read_text()) == p
    for name, expected in summary['source_sha256'].items():
        assert sha256(source / name) == expected
    ids, xf, yf, primary = load_data(source, Path('outputs/p0_audit_20261023_final'))
    excluded = json.loads(Path('configs/final_protocol.json').read_text())['excluded_derived_features']
    features = [f for f in xf if f not in excluded]
    x, y = xf[features].to_numpy(float), yf.to_numpy(float)
    target_mask = np.array([t in primary for t in yf])
    methods = p['candidates'] + ['inner_selected']
    data = pd.read_csv(output / 'oof_predictions.csv.gz', dtype={'beer': str}, float_precision='round_trip')
    inner = pd.read_csv(output / 'inner_predictions.csv.gz', dtype={'beer': str})
    selections = pd.read_csv(output / 'inner_selection.csv')
    inner_splits = pd.read_csv(output / 'inner_splits.csv', dtype={'beer': str})
    outer_splits = pd.read_csv(output / 'outer_splits.csv', dtype={'beer': str}).set_index('beer')
    assert len(data) == 4 * 250 * 50 and not data.duplicated(['method', 'beer', 'target']).any()
    assert set(data.method) == set(methods)
    probs = data[[f'p{k}' for k in range(7)]].to_numpy()
    assert np.isfinite(probs).all() and (probs >= 0).all()
    np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-12)
    np.testing.assert_array_equal(data.predicted, (probs.cumsum(axis=1) >= .5).argmax(axis=1))
    assert np.array_equal(data.primary, data.target.isin(primary))
    splits = make_splits(ids.tasting_category_fine, p['outer_folds'], p['outer_split_seed'])
    truths = np.empty_like(y, int)
    target_choices = {}
    # Each inner label reference is reconstructed from only that fold's fit rows.
    for context, train, seed in [(f'outer{f}', tr, p['seed'] + f * 100) for f, (tr, _) in enumerate(splits)] + [
            ('full_training', np.arange(len(y)), p['seed'] + 1000)]:
        for fold, (it, iv) in enumerate(make_splits(ids.tasting_category_fine.iloc[train], p['inner_folds'], seed)):
            fitting, validation = train[it], train[iv]
            saved = inner_splits[inner_splits.context.eq(context) & inner_splits.fold.eq(fold)]
            assert saved[saved.role.eq('train')].beer.tolist() == ids.beer.iloc[fitting].astype(str).tolist()
            assert saved[saved.role.eq('validation')].beer.tolist() == ids.beer.iloc[validation].astype(str).tolist()
            assert set(fitting).isdisjoint(validation)
            expected = count_grades(y[fitting], y[validation])
            for name in p['candidates']:
                part = inner[inner.context.eq(context) & inner.fold.eq(fold) & inner.method.eq(name)]
                actual = part.pivot(index='beer', columns='target', values='observed').loc[ids.beer.iloc[validation], yf.columns]
                np.testing.assert_array_equal(actual, expected)
        errors = []
        part = inner[inner.context.eq(context)]
        assert len(part) == len(train) * 50 * 3 and not part.duplicated(['method', 'beer', 'target']).any()
        for name in p['candidates']:
            g = part[part.method.eq(name)]
            errors.append((g.predicted - g.observed).abs().groupby(g.target).mean().reindex(yf.columns).to_numpy())
        errors = np.array(errors)
        choices = errors.argmin(axis=0)
        target_choices[context] = choices
        for m, name in enumerate(p['candidates']):
            saved = selections[selections.context.eq(context) & selections.method.eq(name)].set_index('target').loc[yf.columns]
            np.testing.assert_allclose(saved.inner_mae, errors[m], atol=1e-12)
            np.testing.assert_array_equal(saved.selected, choices == m)
    replayed = 0
    for fold, (train, test) in enumerate(splits):
        np.testing.assert_array_equal(outer_splits.loc[ids.beer.iloc[test], 'fold'], fold)
        truth = count_grades(y[train], y[test])
        truths[test] = truth
        fitting_grades = count_grades(y[train], y[train])
        for name in p['candidates']:
            # Actual refitting of all nine outer models, not just saved summaries.
            model = make_ordered_candidate(name, p, p['current_tree_seed'] + fold).fit(x[train], fitting_grades)
            probability = seven_level_probabilities(model, x[test])
            np.testing.assert_allclose(model.named_steps['impute'].statistics_, np.nanmedian(x[train], axis=0))
            part = data[data.method.eq(name) & data.fold.eq(fold)]
            for k in range(7):
                actual = part.pivot(index='beer', columns='target', values=f'p{k}').loc[ids.beer.iloc[test], yf.columns]
                np.testing.assert_allclose(actual, probability[:, :, k], atol=1e-12, rtol=0)
            replayed += 1
        for j, target in enumerate(yf):
            chosen = p['candidates'][target_choices[f'outer{fold}'][j]]
            current = data[data.fold.eq(fold) & data.target.eq(target)]
            a = current[current.method.eq('inner_selected')].set_index('beer').loc[ids.beer.iloc[test]]
            b = current[current.method.eq(chosen)].set_index('beer').loc[ids.beer.iloc[test]]
            np.testing.assert_array_equal(a.predicted, b.predicted)
            np.testing.assert_allclose(a[[f'p{k}' for k in range(7)]], b[[f'p{k}' for k in range(7)]], atol=0, rtol=0)
    frames = {}
    for name in methods:
        g = data[data.method.eq(name)]
        observed = g.pivot(index='beer', columns='target', values='observed').loc[ids.beer.astype(str), yf.columns]
        np.testing.assert_array_equal(observed, truths)
        frames[name] = g.pivot(index='beer', columns='target', values='predicted').loc[ids.beer.astype(str), yf.columns].to_numpy(int)
    historical = pd.read_csv('outputs/q1_task2_20261025/scores_all52_long.csv', dtype={'beer': str})
    historical = historical[historical.panel_target_available].pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[ids.beer.astype(str), yf.columns]
    np.testing.assert_array_equal(frames[methods[0]], historical)

    def verify_metric(row, predicted, observed):
        error = abs(predicted - observed)
        assert row.n == error.size
        np.testing.assert_allclose([row.grade_mae, row.exact_accuracy, row.within_one_accuracy],
            [error.mean(), (error == 0).mean(), (error <= 1).mean()], atol=1e-12)
    for row in pd.read_csv(output / 'model_comparison.csv').itertuples():
        mask = target_mask if row.scope == 'primary45' else np.ones(50, bool)
        verify_metric(row, frames[row.method][:, mask], truths[:, mask])
    for row in pd.read_csv(output / 'metrics_by_target.csv').itertuples():
        j = yf.columns.get_loc(row.target)
        assert row.primary == target_mask[j]
        verify_metric(row, frames[row.method][:, j], truths[:, j])
    grade_rows = pd.read_csv(output / 'metrics_by_observed_grade.csv')
    for row in grade_rows.itertuples():
        mask = truths[:, target_mask] == row.observed_grade
        verify_metric(row, frames[row.method][:, target_mask][mask], truths[:, target_mask][mask])
    for row in pd.read_csv(output / 'confusion_primary45.csv').itertuples():
        expected = ((truths[:, target_mask] == row.observed) & (frames[row.method][:, target_mask] == row.predicted)).sum()
        assert row.count == expected
        np.testing.assert_allclose(row.row_fraction, expected / (truths[:, target_mask] == row.observed).sum())
    for row in pd.read_csv(output / 'grade_distribution_primary45.csv').itertuples():
        assert row.observed_count == (truths[:, target_mask] == row.grade).sum()
        assert row.predicted_count == (frames[row.method][:, target_mask] == row.grade).sum()
    for row in pd.read_csv(output / 'fold_metrics.csv').itertuples():
        test = splits[row.fold][1]
        verify_metric(row, frames[row.method][test][:, target_mask], truths[test][:, target_mask])
    base_loss = abs(frames[methods[0]][:, target_mask] - truths[:, target_mask]).mean(axis=1)
    difference = np.stack([base_loss - abs(frames[name][:, target_mask] - truths[:, target_mask]).mean(axis=1)
                           for name in methods[1:]], axis=1)
    rng = np.random.default_rng(p['seed'])
    boot = np.array([difference[stratified_draw(ids.tasting_category_fine, rng)].mean(axis=0)
                     for _ in range(p['uncertainty_repeats'])])
    intervals = pd.read_csv(output / 'paired_uncertainty.csv').set_index('method')
    for j, name in enumerate(methods[1:]):
        row = intervals.loc[name]
        np.testing.assert_allclose([row.mae_improvement, row.fixed_oof_p025, row.fixed_oof_p975],
            [difference[:, j].mean(), np.quantile(boot[:, j], .025), np.quantile(boot[:, j], .975)], atol=1e-12)
    grade_rows = grade_rows.set_index(['method', 'observed_grade'])
    tails = all(grade_rows.loc[('inner_selected', g), 'grade_mae'] <= grade_rows.loc[(methods[0], g), 'grade_mae']
        and grade_rows.loc[('inner_selected', g), 'exact_accuracy'] >= grade_rows.loc[(methods[0], g), 'exact_accuracy'] for g in [0, 6])
    assert summary['tail_nonworsening_checks'] == tails
    assert summary['selected_workflow_qualified_on_historical_partition'] == bool(tails and intervals.loc['inner_selected', 'fixed_oof_p025'] > 0)
    assert sha256(output / 'model.joblib') == summary['model_sha256']
    bundle = joblib.load(output / 'model.joblib')
    selected = bundle['selected_grade_model']
    np.testing.assert_array_equal(selected.choices, target_choices['full_training'])
    for model in selected.models:
        np.testing.assert_allclose(model.named_steps['impute'].statistics_, np.nanmedian(x, axis=0))
    assert bundle['task2_official_default'] is False
    smoke_rows = 0
    if smoke is not None:
        info = json.loads((smoke / 'prediction_info.json').read_text())
        assert info['task2_official_default'] is False
        scores = pd.read_csv(smoke / 'scores_all52_long.csv', dtype={'beer': str})
        panel = scores[scores.panel_target_available]
        actual = panel.pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[ids.beer.astype(str), yf.columns]
        q = seven_level_probabilities(selected, x)
        np.testing.assert_array_equal(actual, (q.cumsum(axis=2) >= .5).argmax(axis=2))
        expected_continuous = bundle['model'].predict(x)
        actual_continuous = panel.pivot(index='beer', columns='target', values='standardized_prediction').loc[ids.beer.astype(str), yf.columns]
        np.testing.assert_allclose(actual_continuous, expected_continuous, atol=1e-12)
        assert len(scores) == 13000 and scores.target.nunique() == 52
        assert scores[~scores.panel_target_available].grade_kind.eq('chemical_threshold_proxy_no_panel_label').all()
        assert scores[~scores.panel_target_available].estimated_grade_0_6.eq(0).all()
        smoke_rows = len(scores)
    report = dict(status='passed',outer_models_refitted=replayed, OOF_rows_checked=len(data),
        inner_rows_checked=len(inner), inner_contexts_checked=4, CLI_rows_checked=smoke_rows,
        baseline_OOF_reproduced=12500, original_absolute_scale_recovered=False,
        checks=['training-only split memberships and independent grade counting',
            'per-target choices from inner data only', 'all nine outer models refitted',
            'seven-level probabilities and selected column assembly', 'all metrics/confusion/distributions',
            '2000 paired fixed-OOF resamples and qualification', 'full-data choices and explicit CLI'],
        limitation='Outer fits replayed; inner stored predictions and full-data coefficients were not all independently refitted. No later task5 repeated validation.')
    write_json(output / 'verification.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        verify(args.source, args.output, args.smoke)
