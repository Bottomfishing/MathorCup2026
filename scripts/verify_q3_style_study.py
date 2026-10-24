"""Replay Q3 style results and report a stricter three-proxy sensitivity scenario."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, sha256, write_json
from beer_models.baseline import metric_vectors
from beer_models.generation import StyleGenerator
from beer_models.style_models import make_style_model, make_stale_model
from complete_modeling import load_q1_artifacts


def verify(source, q1, q3, output):
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    config_path = Path('configs/q3_style_protocol.json')
    assert sha256(config_path) == summary['config_sha256']
    config = summary['config']
    ids, xf, yf, bundle, splits = load_q1_artifacts(source, Path('outputs/p0_audit_20261023_final'), q1)
    for name, digest in summary['source_sha256'].items():
        assert sha256(source / name) == digest
    x = xf.to_numpy(float); styles = ids.tasting_category_fine.to_numpy()
    beers = ids.beer.astype(str).to_numpy(); target = yf.stale_hops.to_numpy(float)
    stored = pd.read_csv(output / 'style_oof.csv', dtype={'beer': str})
    stored_stale = pd.read_csv(output / 'stale_oof.csv', dtype={'beer': str}).set_index('beer').loc[beers]
    np.testing.assert_allclose(stored_stale.observed, target, atol=1e-12)
    oof = {name: np.empty(len(x), object) for name in ['logistic', 'extra_trees']}
    stale = np.empty(len(x)); mean = np.empty(len(x))
    for fold, (train, test) in enumerate(splits):
        for name in oof:
            model = make_style_model(name, config, config['seed'] + fold).fit(x[train], styles[train])
            np.testing.assert_allclose(model.named_steps['impute'].statistics_, np.nanmedian(x[train], axis=0))
            oof[name][test] = model.predict(x[test])
        focused = make_stale_model(config, config['seed'] + fold).fit(x[train], target[train])
        np.testing.assert_allclose(focused.named_steps['impute'].statistics_, np.nanmedian(x[train], axis=0))
        stale[test] = focused.predict(x[test]); mean[test] = target[train].mean()
        print(f'Independent Q3 fold {fold} refit passed', flush=True)
    np.testing.assert_allclose(stale, stored_stale.focused, rtol=0, atol=1e-10)
    np.testing.assert_allclose(mean, stored_stale['mean'], rtol=0, atol=1e-10)
    old = pd.read_csv(q1 / 'oof_predictions.csv', dtype={'beer': str})
    previous = old.loc[old.model.eq('nested_selected') & old.target.eq('stale_hops')].set_index('beer').loc[beers, 'prediction'].to_numpy()
    np.testing.assert_allclose(previous, stored_stale.previous, rtol=0, atol=1e-10)
    centroid = pd.read_csv('outputs/q3_delivery_20261024/style_centroid_oof.csv', dtype={'beer': str}).set_index('beer')
    oof['nearest_centroid'] = centroid.loc[beers, 'nearest_centroid_style'].to_numpy()
    metrics = pd.read_csv(output / 'style_metrics.csv').set_index('method')
    by_class = pd.read_csv(output / 'style_metrics_by_class.csv').set_index(['method', 'style'])
    for name, pred in oof.items():
        subset = stored.loc[stored.method.eq(name)].set_index('beer').loc[beers]
        np.testing.assert_array_equal(subset['style'], styles)
        np.testing.assert_array_equal(pred, subset.prediction)
        np.testing.assert_allclose([accuracy_score(styles, pred), balanced_accuracy_score(styles, pred)],
                                   metrics.loc[name, ['accuracy', 'macro_recall']].to_numpy(float), atol=1e-12)
        for style in np.unique(styles):
            idx = styles == style
            np.testing.assert_allclose((pred[idx] == style).mean(), by_class.loc[(name, style), 'recall'], atol=1e-12)
    stale_metrics = pd.read_csv(output / 'stale_metrics.csv').set_index('method')
    for name, pred in [('mean', mean), ('previous_multioutput', previous), ('focused_single_target', stale)]:
        for key, val in metric_vectors(target[:, None], pred[:, None]).items():
            np.testing.assert_allclose(val[0], stale_metrics.loc[name, key], atol=1e-12)

    # Paired, style-stratified resampling of FIXED OOF outputs; no retraining uncertainty.
    rng = np.random.default_rng(config['seed']); groups = [np.flatnonzero(styles == s) for s in np.unique(styles)]
    differences = {}
    for left, right in [('logistic', 'nearest_centroid'), ('extra_trees', 'nearest_centroid'), ('extra_trees', 'logistic')]:
        differences[left + '_minus_' + right + '_accuracy'] = (oof[left] == styles).astype(float) - (oof[right] == styles)
    differences['focused_minus_previous_stale_mae'] = abs(stale - target) - abs(previous - target)
    intervals = {name: [] for name in differences}
    for _ in range(2000):
        draw = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for name, delta in differences.items():
            intervals[name].append(delta[draw].mean())
    save_csv(output, 'paired_oof_uncertainty.csv', [
        {'contrast': name, 'estimate': delta.mean(), 'lower_95': np.quantile(intervals[name], .025),
         'upper_95': np.quantile(intervals[name], .975), 'resamples': 2000,
         'scope': 'paired style-stratified fixed OOF; excludes refitting and model-selection uncertainty'}
        for name, delta in differences.items()])

    models = {name: joblib.load(output / (name + '.joblib')) for name in ['logistic', 'extra_trees']}
    focused = joblib.load(output / 'focused_stale.joblib')
    for model in [*models.values(), focused]:
        np.testing.assert_allclose(model.named_steps['impute'].statistics_, np.nanmedian(x, axis=0))
    generated = pd.read_csv(q3 / 'generated_profiles.csv')
    diagnostics = pd.read_csv(output / 'generated_style_diagnostics.csv').set_index('generated_id').loc[generated.generated_id]
    np.testing.assert_array_equal(diagnostics['style'], generated['style'])
    match = np.ones(len(generated), bool)
    for name, model in models.items():
        pred = model.predict(generated[xf.columns].to_numpy(float))
        np.testing.assert_array_equal(pred, diagnostics[name + '_prediction'])
        match &= pred == generated['style'].to_numpy()
    np.testing.assert_array_equal(match, diagnostics.both_match_requested_style)

    generator = StyleGenerator().fit(x, styles)
    zall = generator.robust_.transform(generator.x_)
    tripel = generator.x_[generator.indices_['Tripel']]
    q1config = json.loads((q1 / 'run_manifest.json').read_text(encoding='utf-8'))['config']
    samples, _ = generator.sample('Tripel', 100, 'interpolate', q1config['seed'] + 90000)
    anchors = np.vstack([tripel, tripel.mean(axis=0), samples])
    protected = [xf.columns.get_loc(f) for f in q1config['protected_features']]
    lower, upper = tripel.min(axis=0), tripel.max(axis=0)
    center = zall[generator.indices_['Tripel']].mean(axis=0)
    results = pd.read_csv(output / 'strict_transfer_results.csv', dtype={'beer': str}).set_index('beer')
    profiles = pd.read_csv(output / 'strict_transfer_profiles.csv', dtype={'beer': str}).set_index('beer')
    sensory = pd.read_csv(output / 'strict_transfer_sensory.csv', dtype={'beer': str})
    constraints = pd.read_csv(output / 'strict_transfer_constraints.csv', dtype={'beer': str})
    expected_constraints = []
    consensus = []; consensus_profiles = []; consensus_sensory = []; consensus_constraints = []
    stale_index = yf.columns.get_loc('stale_hops'); selected_count = 0
    pils = np.flatnonzero(styles == 'Pils/Lager')
    assert set(results.index) == set(beers[pils])
    for i in pils:
        beer = beers[i]; record = results.loc[beer]
        source_row = generator.x_[i]
        if not np.isfinite(x[i, protected]).all():
            assert not record.selected and record.status == 'source_defect_missing'
            consensus.append({'beer': beer, 'selected': False, 'status': record.status})
            continue
        outside = (source_row[protected] < lower[protected] - 1e-10) | (source_row[protected] > upper[protected] + 1e-10)
        if outside.any():
            assert not record.selected and record.status == 'fixed_defect_outside_empirical_target_box'
            assert record.blocking_fields == ';'.join(np.array(xf.columns)[np.array(protected)[outside]])
            consensus.append({'beer': beer, 'selected': False, 'status': record.status, 'blocking_fields': record.blocking_fields})
            continue
        # Explicit reconstruction checks the saved selection without calling projected_paths.
        fractions = np.tile(np.linspace(1 / config['transfer_steps'], 1, config['transfer_steps']), len(anchors))
        parents = np.repeat(np.arange(len(anchors)), config['transfer_steps'])
        candidates = (1 - fractions[:, None]) * source_row + fractions[:, None] * anchors[parents]
        candidates[:, protected] = source_row[protected]
        pred = bundle['model'].predict(candidates); before = bundle['model'].predict(source_row[None])[0]
        ridge = bundle['guard'].predict(candidates)[:, stale_index]; ridge_before = bundle['guard'].predict(source_row[None])[0, stale_index]
        focused_pred = focused.predict(candidates); focused_before = focused.predict(source_row[None])[0]
        z = generator.robust_.transform(candidates)
        distance = np.sqrt(np.mean((z - center) ** 2, axis=1))
        change = np.sqrt(np.mean((z - zall[i]) ** 2, axis=1))
        source_distance = np.sqrt(np.mean((zall[i] - center) ** 2))
        good = ((candidates >= lower - 1e-10) & (candidates <= upper + 1e-10)).all(axis=1)
        for model in models.values():
            good &= model.predict(candidates) == 'Tripel'
        good &= (pred[:, stale_index] <= before[stale_index] + 1e-10) & (ridge <= ridge_before + 1e-10)
        good &= distance < source_distance - 1e-8
        objective = config['target_distance_weight'] * distance + (1 - config['target_distance_weight']) * change
        assert record.selected == good.any()
        if good.any():
            k = np.flatnonzero(good)[np.argmin(objective[good])]; selected_count += 1
            assert record.status == 'strict_empirical_model_candidate'
            assert int(record.n_passed) == good.sum() and int(record.anchor) == parents[k]
            assert not record.style_authenticated and not record.physical_constraints_certified
            np.testing.assert_allclose(profiles.loc[beer, xf.columns].to_numpy(float), candidates[k], rtol=0, atol=1e-10)
            np.testing.assert_array_equal(profiles.loc[beer, xf.columns[protected]].to_numpy(float), source_row[protected])
            np.testing.assert_allclose([record.fraction, record.source_target_distance, record.target_distance,
                record.distance_reduction_fraction, record.focused_stale_delta_diagnostic_only],
                [fractions[k], source_distance, distance[k], 1 - distance[k] / source_distance, focused_pred[k] - focused_before], atol=1e-10)
            saved_sensory = sensory.loc[sensory.beer.eq(beer)].set_index('target').loc[yf.columns]
            np.testing.assert_allclose(saved_sensory.before, before, atol=1e-10)
            np.testing.assert_allclose(saved_sensory.after, pred[k], atol=1e-10)
            expected_constraints.extend((beer, xf.columns[j], source_row[j], candidates[k, j]) for j in protected)
            expected_constraints += [(beer, 'previous_stale_model', before[stale_index], pred[k, stale_index]),
                                     (beer, 'ridge_stale_model', ridge_before, ridge[k])]
        stricter = good & (focused_pred <= focused_before + 1e-10)
        if not stricter.any():
            consensus.append({'beer': beer, 'selected': False, 'status': 'no_three_proxy_consensus_in_fixed_grid'})
            continue
        k = np.flatnonzero(stricter)[np.argmin(objective[stricter])]
        consensus.append({'beer': beer, 'selected': True, 'status': 'three_proxy_empirical_candidate',
            'fraction': fractions[k], 'anchor': int(parents[k]), 'n_passed': int(stricter.sum()),
            'source_target_distance': source_distance, 'target_distance': distance[k],
            'distance_reduction_fraction': 1 - distance[k] / source_distance,
            'style_authenticated': False, 'physical_constraints_certified': False})
        consensus_profiles.append(dict(zip(xf.columns, candidates[k]), beer=beer))
        consensus_sensory.extend({'beer': beer, 'target': t, 'before': before[j], 'after': pred[k, j]} for j, t in enumerate(yf))
        for j in protected:
            consensus_constraints.append({'beer': beer, 'field': xf.columns[j], 'kind': 'exact_preservation',
                'before': source_row[j], 'after': candidates[k, j]})
        for name, a, b in [('previous_stale_model', before[stale_index], pred[k, stale_index]),
                           ('ridge_stale_model', ridge_before, ridge[k]), ('focused_stale_model', focused_before, focused_pred[k])]:
            consensus_constraints.append({'beer': beer, 'field': name, 'kind': 'proxy_nonincrease', 'before': a, 'after': b})
        print(f'Q3 source {beer}: {int(good.sum())} two-proxy, {int(stricter.sum())} three-proxy candidates', flush=True)
    actual = constraints.set_index(['beer', 'field'])
    assert len(actual) == len(expected_constraints)
    for beer, field, before, after in expected_constraints:
        np.testing.assert_allclose(actual.loc[(beer, field), ['before', 'after']].to_numpy(float), [before, after], atol=1e-10)
    for name, data in [('consensus_transfer_results', consensus), ('consensus_transfer_profiles', consensus_profiles),
                       ('consensus_transfer_sensory', consensus_sensory), ('consensus_transfer_constraints', consensus_constraints)]:
        save_csv(output, name + '.csv', data)
    result = {'status': 'passed', 'style_fold_refits': 6, 'focused_stale_fold_refits': 3,
        'generated_style_rows_verified': len(generated), 'strict_selected_verified': selected_count,
        'three_proxy_status_counts': pd.DataFrame(consensus).status.value_counts().to_dict(),
        'models_sha256': {name: sha256(output / (name + '.joblib')) for name in [*models, 'focused_stale']},
        'checks': ['source and Q1 model hashes', 'independent fold refits and training-only imputation',
            'OOF labels and metrics', 'paired fixed-OOF uncertainty', 'generated style predictions',
            'explicit path reconstruction and optimal selection within fixed grid',
            'all-feature empirical box, two style classifiers, exact defects, stale proxies, sensory exports'],
        'interpretation': 'Three-proxy scenario is a stricter sensitivity analysis after observing model disagreement, not new validation data. Empirical ranges and model agreement do not certify real flavor, style or brewing feasibility.'}
    write_json(output / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--q1', type=Path, default=Path('outputs/final_20261024/q1'))
    p.add_argument('--q3', type=Path, default=Path('outputs/final_20261024/q3'))
    p.add_argument('--output', type=Path, default=Path('outputs/q3_style_20261024'))
    args = p.parse_args()
    with threadpool_limits(limits=1):
        verify(args.source, args.q1, args.q3, args.output)
