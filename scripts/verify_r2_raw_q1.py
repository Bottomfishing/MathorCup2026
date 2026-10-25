"""Independently recount raw-rating risks and refit the twelve outer models."""
import argparse
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256, write_json
from beer_models.baseline import make_splits, mask_observed
from beer_models.raw_panel import RawPanelModel, predict_raw_frame
from r2_raw_q1 import (PROTOCOL, RAW_MANIFEST, content_hash, load_inputs, metric_tables,
                       supplement_tables, summary_metrics)


def verify(args):
    output = args.output
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['protocol_content_sha256'] == content_hash(protocol)
    assert json.loads((output / 'protocol.json').read_text(encoding='utf-8')) == protocol
    for name, digest in summary['source_sha256'].items():
        assert sha256(args.source / name) == digest
    assert sha256(output / 'model.joblib') == summary['model_sha256']
    data = load_inputs(args.source, args.raw, protocol)
    x, raw, targets, ids = data['x'], data['raw'], data['targets'], data['ids']
    assert summary['raw_source_sha256'] == sha256(args.raw)
    # Recount the label distribution directly by beer -> taster -> presentation.
    q = np.zeros((len(ids), len(targets), 7))
    independent_weights = np.zeros(len(raw))
    for i, beer in enumerate(ids.beer):
        part = raw.loc[raw.beer.eq(beer)]
        groups = list(part.groupby('taster'))
        for _, ratings in groups:
            weight = 1 / (len(groups) * len(ratings))
            independent_weights[ratings.index] = weight
            values = ratings[targets].to_numpy(int)
            q[i] += np.equal(values[:, :, None], np.arange(7)).sum(axis=0) * weight
    np.testing.assert_allclose(q, data['distribution'], rtol=0, atol=1e-14)
    np.testing.assert_allclose(independent_weights, data['weights'], rtol=0, atol=1e-14)
    risks = np.stack([(q * abs(np.arange(7)-action)).sum(axis=2) for action in range(7)], axis=2)
    observed_median = (risks <= risks.min(axis=2, keepdims=True) + 1e-12).argmax(axis=2)

    def read(name):
        return pd.read_csv(output / name, float_precision='round_trip')

    def compare(name, expected):
        # Compare the CSV representation, including nullable blanks, not pandas-only axis metadata.
        expected = pd.read_csv(io.StringIO(pd.DataFrame(expected).to_csv(index=False)), float_precision='round_trip')
        actual = read(name)
        if 'beer' in expected:
            actual['beer'] = actual.beer.astype(str)
            expected['beer'] = expected.beer.astype(str)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False,
                                      check_exact=False, rtol=0, atol=1e-12, obj=name)

    def matrix(frame, field, beers=ids.beer):
        return frame.pivot(index='beer', columns='target', values=field).loc[beers, targets].to_numpy()

    methods = protocol['candidates']
    oof = read('oof_predictions.csv.gz')
    inner = read('inner_predictions.csv.gz')
    assert len(oof) == 62500 and not oof.duplicated(['method', 'beer', 'target']).any()
    assert len(inner) == 150000 and not inner.duplicated(['context', 'method', 'beer', 'target']).any()
    predictions = {}
    for method in methods + ['inner_selected']:
        frame = oof.loc[oof.method.eq(method)]
        values = matrix(frame, 'predicted_grade')
        assert np.isfinite(values).all() and np.equal(values, np.floor(values)).all()
        assert ((0 <= values) & (values <= 6)).all()
        np.testing.assert_array_equal(matrix(frame, 'observed_panel_median'), observed_median)
        direct = np.sum(abs(data['scores'] - values[data['positions']]) * independent_weights[:, None], axis=0) / len(ids)
        weighted = (q * abs(np.arange(7) - values[:, :, None])).sum(axis=2)
        np.testing.assert_allclose(direct, weighted.mean(axis=0), rtol=0, atol=1e-13)
        np.testing.assert_allclose(matrix(frame, 'weighted_raw_absolute_loss'), weighted, rtol=0, atol=1e-12)
        predictions[method] = dict(grade=values.astype(int),
            unconstrained_grade=matrix(frame, 'unconstrained_grade').astype(int),
            raw_median_estimate=matrix(frame, 'raw_median_estimate'))
        for parent, child in data['parents']:
            assert (values[:, child] <= values[:, parent]).all()
    for name, table in metric_tables(data, predictions).items():
        compare(name, table)
    hierarchy = []
    for method, result in predictions.items():
        for stage, key in [('before_parent_cap', 'unconstrained_grade'), ('after_parent_cap', 'grade')]:
            count = sum(int((result[key][:, child] > result[key][:, parent]).sum()) for parent, child in data['parents'])
            hierarchy.append(dict(method=method, stage=stage, checked_pairs=len(ids)*len(data['parents']), violations=count))
    compare('hierarchy.csv', hierarchy)
    compare('panel_disagreement.csv', [dict(scope=scope,
        empirical_panel_median_raw_absolute_deviation=float(risks[:, mask].min(axis=2).mean()),
        interpretation='Observed-panel best constant action per beer; a data diagnostic, not an achievable forecast or population noise floor')
        for scope, mask in [('all50', np.ones(len(targets), bool)), ('primary45', data['primary'])]])

    truth = read('observed_panel_distribution.csv')
    np.testing.assert_array_equal(matrix(truth, 'panel_weighted_median'), observed_median)
    np.testing.assert_allclose(matrix(truth, 'panel_weighted_mean'), (q*np.arange(7)).sum(axis=2), atol=1e-12, rtol=0)
    for k in range(7):
        np.testing.assert_allclose(matrix(truth, f'probability_grade_{k}'), q[:, :, k], atol=1e-14, rtol=0)

    styles = ids.tasting_category_fine.to_numpy()
    splits = make_splits(styles, protocol['outer_folds'], protocol['outer_seed'])
    saved_folds = read('outer_folds.csv')
    np.testing.assert_array_equal(saved_folds.beer, ids.beer)
    choices = read('inner_selection.csv')
    split_log = read('inner_splits.csv')
    chosen = {}
    contexts = [(f'outer_{fold}', train, fold) for fold, (train, _) in enumerate(splits)]
    contexts.append(('full_training', np.arange(len(ids)), len(splits)))
    for context, train, number in contexts:
        expected_splits = make_splits(styles[train], protocol['inner_folds'], protocol['inner_seed'] + number)
        for fold, (it, iv) in enumerate(expected_splits):
            records = split_log.loc[split_log.context.eq(context) & split_log.inner_fold.eq(fold)]
            for role, positions in [('train', train[it]), ('validation', train[iv])]:
                np.testing.assert_array_equal(records.loc[records.role.eq(role), 'beer'], ids.beer.iloc[positions])
            for method in methods:
                recorded = inner.loc[inner.context.eq(context) & inner.inner_fold.eq(fold) & inner.method.eq(method)]
                assert set(recorded.beer) == set(ids.beer.iloc[train[iv]])
        losses = []
        for method in methods:
            frame = inner.loc[inner.context.eq(context) & inner.method.eq(method)]
            grades = matrix(frame, 'predicted_grade', ids.beer.iloc[train])
            loss = float((q[train] * abs(np.arange(7) - grades[:, :, None])).sum(axis=2).mean())
            saved = choices.loc[choices.context.eq(context) & choices.method.eq(method)].iloc[0]
            assert abs(saved.inner_raw_rating_mae - loss) <= 1e-12
            losses.append(loss)
        selected = next(m for m, loss in zip(methods, losses) if loss <= min(losses) + 1e-12)
        assert choices.loc[choices.context.eq(context) & choices.selected, 'method'].tolist() == [selected]
        chosen[context] = selected

    replayed = 0
    stress = read('missingness_by_fold.csv')
    for fold, (train, test) in enumerate(splits):
        assert (saved_folds.outer_fold.iloc[test] == fold).all()
        assert set(ids.beer.iloc[train]).isdisjoint(ids.beer.iloc[test])
        selected = chosen[f'outer_{fold}']
        np.testing.assert_array_equal(predictions['inner_selected']['grade'][test], predictions[selected]['grade'][test])
        for method in methods:
            saved = joblib.load(output / 'models' / f'{method}_fold{fold}.joblib')
            np.testing.assert_allclose(saved.imputer.statistics_, np.nanmedian(x[train], axis=0), rtol=0, atol=0)
            lookup = np.full(len(x), -1, int); lookup[train] = np.arange(len(train))
            rows = lookup[data['positions']] >= 0
            print(f'Refit outer {fold} {method}', flush=True)
            fresh = RawPanelModel(method, protocol, protocol['model_seed']+fold*100+99, data['parents']).fit(
                x[train], lookup[data['positions'][rows]], data['scores'][rows], independent_weights[rows], q[train])
            expected = fresh.predict(x[test]); replay = saved.predict(x[test])
            for key in expected:
                np.testing.assert_allclose(replay[key], expected[key], rtol=0, atol=1e-12)
                np.testing.assert_allclose(replay[key], predictions[method][key][test], rtol=0, atol=1e-12)
            replayed += 1
            if method == selected:
                for rate in protocol['additional_missing_rates']:
                    for repeat in range(protocol['missing_repeats']):
                        masked, _ = mask_observed(x[test], rate, protocol['model_seed']+1000+fold*100+repeat)
                        after = saved.predict(masked)['grade']
                        expected_loss = (q[test] * abs(np.arange(7)-after[:, :, None])).sum(axis=2).mean()
                        row = stress.loc[stress.outer_fold.eq(fold) & stress.additional_missing_rate.eq(rate) & stress['repeat'].eq(repeat)].iloc[0]
                        assert abs(row.raw_rating_mae - expected_loss) <= 1e-12
                        assert abs(row.changed_grade_fraction - (after != replay['grade']).mean()) <= 1e-12
    for name, table in supplement_tables(data, predictions, saved_folds.outer_fold.to_numpy(), stress).items():
        compare(name, table)
    weights = read('rating_weights_and_folds.csv')
    np.testing.assert_allclose(weights.weight_within_beer, independent_weights, atol=1e-14, rtol=0)
    assert weights.groupby('beer').outer_fold.nunique().eq(1).all()
    np.testing.assert_array_equal(weights.outer_fold, saved_folds.outer_fold.to_numpy()[data['positions']])
    prep = read('preprocessing_checks.csv')
    assert len(prep) == 45 and prep.unique_beer_imputation_max_difference.eq(0).all()
    assert prep.maximum_per_beer_weight_error.max() < 1e-12

    paired = read('paired_uncertainty.csv')
    base = (q * abs(np.arange(7)-predictions['training_median']['grade'][:, :, None])).sum(axis=2).mean(axis=1)
    for row in paired.itertuples():
        delta = base - (q * abs(np.arange(7)-predictions[row.method]['grade'][:, :, None])).sum(axis=2).mean(axis=1)
        rng = np.random.default_rng(protocol['bootstrap_seed'])
        draws = []
        for _ in range(protocol['bootstrap_repeats']):
            index = np.concatenate([rng.choice(np.flatnonzero(styles == style), size=int((styles == style).sum()), replace=True)
                                    for style in sorted(set(styles))])
            draws.append(delta[index].mean())
        np.testing.assert_allclose([row.mean_improvement, row.lower95, row.upper95],
            [delta.mean(), *np.quantile(draws, [.025, .975])], rtol=0, atol=1e-12)
    selected_ci = paired.loc[paired.method.eq('inner_selected')].iloc[0]
    assert summary['qualified_raw_scale_workflow_on_historical_folds'] == bool(selected_ci.lower95 > 0 and selected_ci.mean_improvement > 0)

    bundle = joblib.load(output / 'model.joblib')
    assert bundle['selected_model'] == chosen['full_training'] == summary['full_training_selected_method']
    frame = pd.read_csv(args.source / '1.csv', dtype=str)
    long, primary, wide, quality = predict_raw_frame(frame, bundle)
    for name, table in [('full_fit_scores_all52.csv', long), ('full_fit_grades_primary47.csv', primary),
                        ('full_fit_grades_all52.csv', wide), ('sample_quality.csv', quality)]:
        compare(name, table)
    main = read('oof_scores_all52.csv')
    assert len(main) == 13000 and not main.duplicated(['beer', 'target']).any()
    no_labels = main.target.isin(protocol['unlabelled_card_targets'])
    assert no_labels.sum() == 500 and main.loc[no_labels, 'estimated_grade_0_6'].isna().all()
    np.testing.assert_array_equal(matrix(main.loc[~no_labels], 'estimated_grade_0_6'), predictions['inner_selected']['grade'])
    main_wide = main.pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[ids.beer, data['scope'].target_id].reset_index()
    compare('estimated_grades_all52.csv', main_wide)
    compare('estimated_grades_primary47.csv', main_wide[['beer', *data['scope'].loc[data['scope'].acceptance_scope.eq('main_content'), 'target_id']]])
    for name, table in [('scores_all52_long.csv', long), ('estimated_grades_primary47.csv', primary),
                        ('estimated_grades_all52.csv', wide), ('sample_quality.csv', quality)]:
        actual = pd.read_csv(args.smoke / name, float_precision='round_trip')
        actual['beer'] = actual.beer.astype(str)
        table = pd.read_csv(io.StringIO(table.to_csv(index=False)), float_precision='round_trip')
        table['beer'] = table.beer.astype(str)
        pd.testing.assert_frame_equal(actual, table, check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
    info = json.loads((args.smoke / 'prediction_info.json').read_text(encoding='utf-8'))
    assert info['score_scale'] == 'original_0_6_individual_rating_scale'
    assert info['model_sha256'] == summary['model_sha256'] and info['input_sha256'] == sha256(args.source / '1.csv')
    result = dict(status='passed', protocol_content_sha256=content_hash(protocol),
        independently_recounted_rating_cells=int(data['scores'].size), oof_rows=len(oof), inner_rows=len(inner),
        outer_models_independently_refitted=replayed, inner_choice_contexts=len(contexts),
        source_beers=len(ids), grouped_rating_records=len(raw), primary_metric_recomputed_from_original_records=True,
        missingness_replays=len(stress), complete_card_rows=len(main), unlabelled_blank_rows=int(no_labels.sum()),
        actual_cli_samples=len(frame), actual_cli_rows=len(long),
        uncertainty_recomputed=True, hierarchy_checked=True,
        scope='Original-scale historical grouped nested comparison; full-data and inner fitted models not all independently refitted.')
    write_json(output / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent / 'Raw sensory data anonymized.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        verify(args)
