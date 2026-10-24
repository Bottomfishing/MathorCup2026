"""Correct Q3 defect support, resolve zero-reference missingness and check Stale loss."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, sha256, write_json
from beer_models.baseline import metric_vectors
from beer_models.defect_constraints import project_defects, defect_aware_box, reference_zero_bound
from beer_models.generation import StyleGenerator
from complete_modeling import load_q1_artifacts


def stale_loss_check(x, yf, ids, splits, style_dir, config, output):
    y = yf.stale_hops.to_numpy(float)
    saved = pd.read_csv(style_dir / 'stale_oof.csv', dtype={'beer': str}).set_index('beer').loc[ids.beer.astype(str)]
    np.testing.assert_allclose(saved.observed, y, atol=1e-12)
    pred = {name: saved[field].to_numpy() for name, field in
            [('mean', 'mean'), ('previous_multioutput', 'previous'), ('focused_squared_tree', 'focused')]}
    for name in ['median', 'absolute_tree', 'absolute_boosting']:
        pred[name] = np.empty(len(y))
    for fold, (train, test) in enumerate(splits):
        pred['median'][test] = np.median(y[train])
        candidates = {
            'absolute_tree': ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=.7,
                criterion='absolute_error', random_state=config['seed'] + fold, n_jobs=1),
            'absolute_boosting': GradientBoostingRegressor(loss='absolute_error', n_estimators=200,
                max_depth=2, learning_rate=.03, min_samples_leaf=5, random_state=config['seed'] + fold)}
        for name, model in candidates.items():
            pipeline = make_pipeline(SimpleImputer(strategy='median'), model).fit(x[train], y[train])
            np.testing.assert_allclose(pipeline.steps[0][1].statistics_, np.nanmedian(x[train], axis=0))
            pred[name][test] = pipeline.predict(x[test])
        print(f'Stale loss comparison fold {fold} complete', flush=True)
    metrics = []
    for name, values in pred.items():
        metrics.append({'method': name, **{k: float(v[0]) for k, v in metric_vectors(y[:, None], values[:, None]).items()}})
    # Same paired style-stratified FIXED-OOF uncertainty as the previous stage.
    groups = [np.flatnonzero(ids.tasting_category_fine.to_numpy() == s) for s in sorted(set(ids.tasting_category_fine))]
    rng = np.random.default_rng(config['seed']); deltas = {n: abs(v - y) - abs(pred['median'] - y) for n, v in pred.items() if n != 'median'}
    draws = {name: [] for name in deltas}
    for _ in range(2000):
        sample = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for name, delta in deltas.items():
            draws[name].append(delta[sample].mean())
    save_csv(output, 'stale_loss_comparison.csv', metrics)
    save_csv(output, 'stale_loss_oof.csv', dict(beer=ids.beer, observed=y, **pred))
    save_csv(output, 'stale_mae_difference_from_median.csv', [
        {'method': name, 'mae_difference': delta.mean(), 'lower_95': np.quantile(draws[name], .025),
         'upper_95': np.quantile(draws[name], .975), 'resamples': 2000,
         'scope': 'paired style-stratified fixed OOF; no refitting uncertainty'} for name, delta in deltas.items()])
    return metrics


def run(source, q1, style_dir, output):
    config_path = Path('configs/q3_resolution_protocol.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    ids, xf, yf, bundle, splits = load_q1_artifacts(source, Path('outputs/p0_audit_20261023_final'), q1)
    prior = json.loads((style_dir / 'summary.json').read_text(encoding='utf-8'))
    assert prior['source_sha256'] == {n: sha256(source / n) for n in ['1.csv', '4.csv', '8.xlsx']}
    output.mkdir(parents=True, exist_ok=False)
    x = xf.to_numpy(float); styles = ids.tasting_category_fine.to_numpy()
    calibration = pd.read_excel(source / '8.xlsx', sheet_name='Calibrated chemical dataset')
    assert not calibration.beer.duplicated().any()
    calibration['excel_row'] = np.arange(len(calibration)) + 2
    calibration = calibration.set_index('beer'); calibration.index = calibration.index.astype(str)
    protected = [xf.columns.get_loc(f) for f in config['protected_features']]
    # Report disagreements; do not overwrite observed coordinates with reference values.
    scale_rows = []; conflict_rows = []
    for field, transform in [('DMS', 'log10_positive'), ('X4VG', 'log10_positive'),
                              ('X2.3.butanedione', 'identity'), ('X.E..2.nonenal', 'identity')]:
        ref = calibration.loc[ids.beer.astype(str), field].to_numpy(float)
        observed = xf[field].to_numpy(float)
        comparable = np.isfinite(ref) & np.isfinite(observed)
        transformed = ref.copy()
        if transform == 'log10_positive':
            comparable &= ref > 0
            transformed = np.log10(np.where(ref > 0, ref, np.nan))
        equal = abs(observed - transformed) <= 1e-6
        scale_rows.append({'field': field, 'transform': transform, 'n_comparable': int(comparable.sum()),
                           'n_agree_1e6': int((comparable & equal).sum()), 'n_zero_references': int((ref == 0).sum())})
        for i in np.flatnonzero(comparable & ~equal):
            conflict_rows.append({'beer': ids.beer.iloc[i], 'field': field, 'csv_coordinate': observed[i],
                                  'reference_value': ref[i], 'action': 'retain_observed_coordinate'})
    save_csv(output, 'coordinate_reference_audit.csv', scale_rows)
    save_csv(output, 'coordinate_reference_conflicts.csv', conflict_rows)
    loss_metrics = stale_loss_check(x, yf, ids, splits, style_dir, config, output)
    generator = StyleGenerator().fit(x, styles)
    tripel = generator.x_[generator.indices_['Tripel']]
    lower, upper = tripel.min(axis=0), tripel.max(axis=0)
    # Match the previous generation/anchor protocol; do not search more seeds.
    anchor_seed = json.loads((q1 / 'run_manifest.json').read_text(encoding='utf-8'))['config']['seed'] + 90000
    samples, _ = generator.sample('Tripel', 100, 'interpolate', anchor_seed)
    anchors = np.vstack([tripel, tripel.mean(axis=0), samples])
    save_csv(output, 'anchors.csv', pd.DataFrame(anchors, columns=xf.columns).assign(anchor=np.arange(len(anchors))))
    center = generator.robust_.transform(tripel).mean(axis=0)
    classifiers = {n: joblib.load(style_dir / (n + '.joblib')) for n in ['logistic', 'extra_trees']}
    focused = joblib.load(style_dir / 'focused_stale.joblib')
    models = {'previous_multioutput': bundle['model'], 'ridge': bundle['guard'], 'focused': focused}
    stale_index = yf.columns.get_loc('stale_hops')
    results = []; profiles = []; sensory = []; checks = []; evidence = []; proxy_rows = []
    for i in np.flatnonzero(styles == 'Pils/Lager'):
        beer = str(ids.beer.iloc[i]); source_row = generator.x_[i].copy()
        for j in protected:
            field = xf.columns[j]; reference = float(calibration.loc[beer, field])
            value, basis = reference_zero_bound(x[i, j], field, reference)
            source_row[j] = value
            evidence.append({'beer': beer, 'field': field, 'csv_coordinate': x[i, j],
                'attachment8_calibrated_value': reference, 'constraint_cap_coordinate': value, 'basis': basis,
                'reference_sheet': 'Calibrated chemical dataset', 'reference_excel_row': int(calibration.loc[beer, 'excel_row']),
                'raw_measurement_recovered': False})
        fractions = np.tile(np.linspace(1 / config['transfer_steps'], 1, config['transfer_steps']), len(anchors))
        parents = np.repeat(np.arange(len(anchors)), config['transfer_steps'])
        raw = source_row + fractions[:, None] * (anchors[parents] - source_row)
        candidates = np.vstack([project_defects(raw, source_row, protected, p) for p in config['defect_policies']])
        policy = np.repeat(config['defect_policies'], len(raw))
        fractions = np.tile(fractions, len(config['defect_policies'])); parents = np.tile(parents, len(config['defect_policies']))
        good = defect_aware_box(candidates, lower, upper, protected)
        n_box = int(good.sum()); style_predictions = {n: m.predict(candidates) for n, m in classifiers.items()}
        for prediction in style_predictions.values():
            good &= prediction == 'Tripel'
        n_style = int(good.sum())
        full_prediction = models['previous_multioutput'].predict(candidates)
        full_before = models['previous_multioutput'].predict(source_row[None])[0]
        proxy = {'previous_multioutput': full_prediction[:, stale_index]}
        proxy_before = {'previous_multioutput': full_before[stale_index]}
        for name in ['ridge', 'focused']:
            a = models[name].predict(candidates); b = models[name].predict(source_row[None])[0]
            proxy[name] = a if name == 'focused' else a[:, stale_index]
            proxy_before[name] = b if name == 'focused' else b[stale_index]
        for name in proxy:
            good &= proxy[name] <= proxy_before[name] + 1e-10
        z = generator.robust_.transform(candidates); zsource = generator.robust_.transform(source_row[None])[0]
        distance = np.sqrt(np.mean((z - center) ** 2, axis=1)); change = np.sqrt(np.mean((z - zsource) ** 2, axis=1))
        source_distance = np.sqrt(np.mean((zsource - center) ** 2))
        good &= distance < source_distance - 1e-8
        if not good.any():
            results.append({'beer': beer, 'selected': False, 'status': 'no_model_candidate_in_fixed_grid'})
            continue
        w = config['target_distance_weight']; eligible = np.flatnonzero(good)
        k = eligible[np.argmin(w * distance[eligible] + (1 - w) * change[eligible])]
        results.append({'beer': beer, 'selected': True, 'status': 'defect_aware_model_candidate',
            'policy': policy[k], 'anchor': int(parents[k]), 'fraction': fractions[k],
            'source_target_distance': source_distance, 'target_distance': distance[k], 'change_distance': change[k],
            'distance_reduction_fraction': 1 - distance[k] / source_distance, 'n_candidates': len(candidates),
            'n_range_passed': n_box, 'n_range_style_passed': n_style, 'n_all_passed': int(good.sum()),
            'below_target_minimum_defect_fields': ';'.join(xf.columns[j] for j in protected if candidates[k, j] < lower[j] - 1e-10),
            'uses_reference_zero_cap': bool(np.isnan(x[i, protected]).any()),
            'style_authenticated': False, 'stale_observed_nonworsening_verified': False,
            'physical_constraints_certified': False})
        profiles.append(dict(zip(xf.columns, candidates[k]), beer=beer))
        for j, t in enumerate(yf.columns):
            sensory.append({'beer': beer, 'target': t, 'source_observed': yf.iloc[i, j],
                'source_model': full_before[j], 'after_model': full_prediction[k, j],
                'model_change': full_prediction[k, j] - full_before[j],
                'residual_anchored_after': yf.iloc[i, j] + full_prediction[k, j] - full_before[j]})
        for j in protected:
            checks.append({'beer': beer, 'field': xf.columns[j], 'kind': 'coordinate_or_zero_cap_nonincrease',
                'before_bound': source_row[j], 'after': candidates[k, j], 'passed': bool(candidates[k, j] <= source_row[j] + 1e-10)})
        for name in proxy:
            delta = proxy[name][k] - proxy_before[name]
            proxy_rows.append({'beer': beer, 'model': name, 'source_prediction': proxy_before[name],
                'after_prediction': proxy[name][k], 'predicted_change': delta,
                'source_observed_stale': yf.stale_hops.iloc[i],
                'residual_anchored_after': yf.stale_hops.iloc[i] + delta,
                'passed_proxy_nonincrease': bool(delta <= 1e-10), 'is_observed_after': False})
        print(f'Q3 beer {beer}: {int(good.sum())} candidates, selected {policy[k]}, distance reduction {1-distance[k]/source_distance:.2%}', flush=True)
    for name, data in [('transfer_results', results), ('transfer_profiles', profiles), ('transfer_sensory', sensory),
                       ('defect_checks', checks), ('source_defect_evidence', evidence), ('stale_proxy_checks', proxy_rows)]:
        save_csv(output, name + '.csv', data)
    summary = {'status': 'completed', 'config': config, 'config_sha256': sha256(config_path),
        'source_sha256': {n: sha256(source / n) for n in ['1.csv', '4.csv', '8.xlsx']},
        'style_models_sha256': {n: sha256(style_dir / (n + '.joblib')) for n in [*classifiers, 'focused_stale']},
        'style_model_directory': str(style_dir), 'anchor_seed': anchor_seed,
        'selected': int(sum(r['selected'] for r in results)), 'n_sources': len(results),
        'reference_zero_capped_beers': sorted({e['beer'] for e in evidence if e['basis'] != 'observed_coordinate'}),
        'stale_loss_metrics': loss_metrics,
        'stale_predictor_reliable_improvement_established': False,
        'interpretation': 'All design checks are explicit model assumptions. Safer defect lower values are permitted; zero caps do not recover missing raw values. No reliable Stale predictor or observed post-transfer outcome has been established. Residual-anchored scores assume source model bias remains unchanged.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--q1', type=Path, default=Path('outputs/final_20261024/q1'))
    parser.add_argument('--style-models', type=Path, default=Path('outputs/q3_style_20261024'))
    parser.add_argument('--output', type=Path, default=Path('outputs/q3_resolution_20261024'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.q1, args.style_models, args.output)
