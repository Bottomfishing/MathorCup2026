"""Reproducible Q1/Q3 continuation; paths are provided by the caller.

Run from a fresh output directory. Q2 evidence and paper are built separately.
AI-assisted, 2026-10-24. No upstream trained models are loaded.
"""
import argparse
import json
import platform
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256, save_csv, write_json
from beer_models.baseline import load_data, make_splits, metric_vectors, mask_observed
from beer_models.explain import stratified_draw
from beer_models.final_model import estimator, relative_grades, projected_paths
from beer_models.generation import StyleGenerator


def choose_model(x, y, styles, primary, config, seed):
    rows = []
    splits = make_splits(styles, config['inner_folds'], seed)
    for name in config['models']:
        pred = np.empty_like(y)
        for fold, (train, test) in enumerate(splits):
            m = estimator(name, config, seed + fold).fit(x[train], y[train])
            pred[test] = m.predict(x[test])
        rows.append({'model': name, 'inner_primary_mae': float(abs(pred[:, primary] - y[:, primary]).mean())})
    # Stable ordering resolves exact ties in favor of the simpler earlier model.
    best = min(rows, key=lambda r: r['inner_primary_mae'])['model']
    return best, rows


def prediction_table(ids, targets, values, **extra):
    return pd.DataFrame({'beer': np.repeat(ids.beer.to_numpy(), len(targets)),
                         'style': np.repeat(ids.tasting_category_fine.to_numpy(), len(targets)),
                         'target': np.tile(targets, len(ids)), 'prediction': values.ravel(), **extra})


def run_q1(source, audit, config, out):
    out.mkdir(parents=True, exist_ok=False)
    ids, xf, yf, primary_names = load_data(source, audit)
    features = [c for c in xf if c not in config['excluded_derived_features']]
    x, y = xf[features].to_numpy(float), yf.to_numpy(float)
    primary = np.array([c in primary_names for c in yf])
    styles = ids.tasting_category_fine.to_numpy()
    splits = make_splits(styles, config['outer_folds'], 20261023)  # match historical folds
    oof = {n: np.empty_like(y) for n in config['models'] + ['nested_selected']}
    grades = np.empty_like(y, int); truth_grades = np.empty_like(y, int)
    folds = np.empty(len(y), int); selections = []; missing = []; calibrations = []
    by_missing = {(r, k): np.empty_like(y) for r in config['missing_rates'] for k in range(config['missing_repeats'])}
    for fold, (train, test) in enumerate(splits):
        folds[test] = fold
        selected, inner = choose_model(x[train], y[train], styles[train], primary, config, config['seed'] + fold)
        selections.extend(dict(r, fold=fold, selected=r['model'] == selected) for r in inner)
        fitted = {}
        for name in config['models']:
            m = estimator(name, config, config['seed'] + fold).fit(x[train], y[train])
            fitted[name] = m; oof[name][test] = m.predict(x[test])
        oof['nested_selected'][test] = oof[selected][test]
        grades[test] = relative_grades(y[train], oof[selected][test])
        truth_grades[test] = relative_grades(y[train], y[test])
        calibrations.append({'fold': fold, 'training_beers': ids.beer.iloc[train].astype(str).tolist(),
                             'sorted_reference': np.sort(y[train], axis=0).tolist()})
        joblib.dump(fitted[selected], out / f'fold{fold}.joblib', compress=3)
        for rate, repeat in by_missing:
            masked, _ = mask_observed(x[test], rate, config['seed'] + 1000 + fold * 100 + repeat)
            by_missing[rate, repeat][test] = fitted[selected].predict(masked)
        print(f'Q1 fold {fold}: selected {selected}', flush=True)
    rows = []; per_target = []
    for name, pred in oof.items():
        metrics = metric_vectors(y, pred)
        rows.append({'model': name, **{'macro_' + k: float(v[primary].mean()) for k, v in metrics.items()}})
        per_target.extend({'model': name, 'target': t, 'primary': bool(primary[j]),
                           **{k: v[j] for k, v in metrics.items()}} for j, t in enumerate(yf))
    save_csv(out, 'model_comparison.csv', rows); save_csv(out, 'metrics_by_target.csv', per_target)
    records = []
    for name, pred in oof.items():
        records.append(prediction_table(ids, yf.columns, pred, model=name, observed=y.ravel(), fold=np.repeat(folds, len(yf.columns))))
    save_csv(out, 'oof_predictions.csv', pd.concat(records, ignore_index=True))
    long = prediction_table(ids, yf.columns, oof['nested_selected'], relative_grade_0_6=grades.ravel(),
                            observed_relative_grade=truth_grades.ravel(), fold=np.repeat(folds, len(yf.columns)),
                            grade_semantics='training_reference_relative_rank_not_raw_panel_score')
    save_csv(out, 'q1_scores_all50_long.csv', long)
    save_csv(out, 'q1_scores_primary45.csv', pd.DataFrame(grades[:, primary], columns=np.array(yf.columns)[primary]).assign(beer=ids.beer.to_numpy(), style=styles))
    grade_metrics = []
    for j, t in enumerate(yf):
        error = abs(grades[:, j] - truth_grades[:, j])
        grade_metrics.append({'target': t, 'primary': bool(primary[j]), 'mae_levels': error.mean(),
                              'exact_accuracy': (error == 0).mean(), 'within_one_accuracy': (error <= 1).mean()})
    save_csv(out, 'grade_metrics.csv', grade_metrics)
    for (rate, repeat), pred in by_missing.items():
        g = np.empty_like(grades)
        for train, test in splits:
            g[test] = relative_grades(y[train], pred[test])
        metric = metric_vectors(y, pred)
        missing.append({'additional_missing_rate': rate, 'repeat': repeat,
                        'macro_mae': metric['mae'][primary].mean(), 'macro_r2': metric['r2'][primary].mean(),
                        'relative_grade_mae': abs(g[:, primary] - truth_grades[:, primary]).mean(),
                        'grade_change_fraction': (g[:, primary] != grades[:, primary]).mean(),
                        'within_one_accuracy': (abs(g[:, primary] - truth_grades[:, primary]) <= 1).mean()})
    save_csv(out, 'missingness.csv', missing)
    # Paired, style-stratified sample bootstrap; fixed OOF models, not retraining uncertainty.
    rng = np.random.default_rng(config['seed'])
    loss = abs(oof['nested_selected'][:, primary] - y[:, primary]).mean(axis=1)
    base_loss = abs(oof['mean'][:, primary] - y[:, primary]).mean(axis=1)
    boot = np.array([[loss[d].mean(), (base_loss[d] - loss[d]).mean()]
                     for d in (stratified_draw(styles, rng) for _ in range(config['bootstrap_repeats']))])
    uncertainty = {'primary_mae': float(loss.mean()), 'mae_interval_95': np.quantile(boot[:, 0], [.025, .975]).tolist(),
                   'improvement_over_mean': float((base_loss - loss).mean()),
                   'improvement_interval_95': np.quantile(boot[:, 1], [.025, .975]).tolist(),
                   'scope': 'fixed_OOF_predictions_style_stratified_sample_bootstrap_not_selection_or_refit_uncertainty'}
    write_json(out / 'uncertainty.json', uncertainty)
    selected, inner = choose_model(x, y, styles, primary, config, config['seed'] + 100)
    save_csv(out, 'full_data_selection.csv', inner)
    final = estimator(selected, config, config['seed']).fit(x, y)
    guard = estimator('ridge_asinh', config, config['seed']).fit(x, y)
    bundle = {'model': final, 'guard': guard, 'features': features, 'targets': yf.columns.tolist(),
              'reference_y': y, 'selected_model': selected, 'primary': primary}
    joblib.dump(bundle, out / 'model.joblib', compress=3)
    save_csv(out, 'splits.csv', ids.assign(outer_test_fold=folds))
    write_json(out / 'calibration.json', calibrations)
    write_json(out / 'run_manifest.json', {'source_sha256': {n: sha256(source / n) for n in ['1.csv', '4.csv', '8.xlsx']},
               'selected_model': selected, 'features': features, 'targets': yf.columns.tolist(), 'config': config,
               'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__,
               'model_sha256': sha256(out / 'model.joblib'), 'original_scores_recovered': False})
    return ids, xf[features], yf, bundle, splits


def joint_metrics(real, hold, generated):
    """Multivariate energy V-statistic and joint rank-correlation discrepancy."""
    p = real.shape[1]
    distance = (2 * cdist(hold, generated).mean() - cdist(hold, hold).mean() - cdist(generated, generated).mean()) / np.sqrt(p)
    # Spearman across generated profiles versus training profiles; constant coordinates excluded.
    keep = (np.std(real, axis=0) > 1e-10) & (np.std(generated, axis=0) > 1e-10)
    a = pd.DataFrame(real[:, keep]).corr(method='spearman').to_numpy()
    b = pd.DataFrame(generated[:, keep]).corr(method='spearman').to_numpy()
    upper = np.triu_indices(len(a), 1)
    corr = float(abs(a[upper] - b[upper]).mean()) if len(upper[0]) else np.nan
    return float(distance), corr


def run_q3(ids, xf, yf, bundle, splits, config, out):
    out.mkdir(parents=True, exist_ok=False)
    x = xf.to_numpy(float); styles = ids.tasting_category_fine.to_numpy(); validation = []
    methods = config['generation_methods']; p = x.shape[1]
    for fold, (train, test) in enumerate(splits):
        generator = StyleGenerator().fit(x[train], styles[train])
        for si, style in enumerate(sorted(set(styles))):
            idx = test[styles[test] == style]
            hold = generator.robust_.transform(generator.impute_.transform(x[idx]))
            real = generator.robust_.transform(generator.x_[generator.indices_[style]])
            for mi, method in enumerate(methods):
                values, meta = generator.sample(style, config['generation_count'], method, config['seed'] + fold * 10000 + si * 100 + mi)
                z = generator.robust_.transform(values)
                energy, correlation = joint_metrics(real, hold, z)
                validation.append({'fold': fold, 'style': style, 'method': method, 'n_holdout': len(idx),
                    'marginal_wasserstein': np.mean([wasserstein_distance(hold[:, j], z[:, j]) for j in range(p)]),
                    'joint_energy': energy, 'training_rank_correlation_mae': correlation,
                    'variance_ratio': np.var(z, axis=0).sum() / max(np.var(real, axis=0).sum(), 1e-12),
                    'copy_fraction': (cdist(z, real).min(axis=1) < 1e-10).mean(),
                    'clipped_fraction': meta['clipped_fraction'].mean()})
        print(f'Q3 joint validation fold {fold}', flush=True)
    validation = pd.DataFrame(validation); save_csv(out, 'validation_by_style_fold.csv', validation)
    save_csv(out, 'validation_summary.csv', validation.groupby('method').mean(numeric_only=True).reset_index())
    generator = StyleGenerator().fit(x, styles)
    joblib.dump(generator, out / 'generator.joblib', compress=3)
    generated = []; metadata = []; sensory = []
    # Interpolation is chosen for novel, bounded designs without the severe PCA variance collapse.
    for si, style in enumerate(config['generation_styles']):
        values, meta = generator.sample(style, config['generation_count'], 'interpolate', config['seed'] + 80000 + si)
        predicted = bundle['model'].predict(values); grades = relative_grades(bundle['reference_y'], predicted)
        for i, row in enumerate(values):
            key = f'{style}_{i:03d}'
            generated.append(dict(zip(xf.columns, row), generated_id=key, style=style))
            metadata.append({'generated_id': key, 'parent_1': ids.beer.iloc[meta['parents'][i, 0]],
                              'parent_2': ids.beer.iloc[meta['parents'][i, 1]], 'weight': meta['weights'][i],
                              'coordinate_system': 'supplied_1csv', 'physical_brewing_validated': False})
            sensory.extend({'generated_id': key, 'style': style, 'target': target, 'prediction': predicted[i, j],
                            'relative_grade_0_6': grades[i, j]} for j, target in enumerate(yf))
    save_csv(out, 'generated_profiles.csv', generated); save_csv(out, 'generated_metadata.csv', metadata)
    save_csv(out, 'generated_sensory.csv', sensory)
    protect = [xf.columns.get_loc(c) for c in config['protected_features']]
    stale = yf.columns.get_loc('stale_hops')
    tripel = generator.x_[generator.indices_['Tripel']]
    interpolated, _ = generator.sample('Tripel', 100, 'interpolate', config['seed'] + 90000)
    anchors = np.vstack([tripel, tripel.mean(axis=0), interpolated])
    ztripel = generator.robust_.transform(tripel); center = ztripel.mean(axis=0)
    centers = {s: generator.robust_.transform(generator.x_[ii]).mean(axis=0) for s, ii in generator.indices_.items()}
    style_order = sorted(centers)
    tripel_radius = float(np.quantile(np.sqrt(np.mean((ztripel - center) ** 2, axis=1)), .95))
    transfers = []; constraints = []; transfer_profiles = []; transfer_sensory = []
    for i in np.flatnonzero(styles == 'Pils/Lager'):
        beer = ids.beer.iloc[i]
        if not np.isfinite(x[i, protect]).all():
            transfers.append({'beer': beer, 'status': 'source_defect_missing', 'selected': False})
            continue
        source = generator.impute_.transform(x[[i]])[0]
        candidates, fractions, parents = projected_paths(source, anchors, protect, config['transfer_steps'])
        z = generator.robust_.transform(candidates); zs = generator.robust_.transform(source[None])[0]
        before_distance = np.sqrt(np.mean((zs - center) ** 2))
        distances = np.sqrt(np.mean((z - center) ** 2, axis=1))
        change = np.sqrt(np.mean((z - zs) ** 2, axis=1))
        prediction = bundle['model'].predict(candidates); before = bundle['model'].predict(source[None])[0]
        guard = bundle['guard'].predict(candidates); guard_before = bundle['guard'].predict(source[None])[0]
        good = (prediction[:, stale] <= before[stale] + 1e-10) & (guard[:, stale] <= guard_before[stale] + 1e-10)
        good &= distances < before_distance - 1e-8
        nearest_style = np.array(style_order)[np.argmin(np.stack([np.mean((z - centers[s]) ** 2, axis=1) for s in style_order]), axis=0)]
        target_like = good & (nearest_style == 'Tripel') & (distances <= tripel_radius)
        pool = np.flatnonzero(target_like if target_like.any() else good)
        if not len(pool):
            transfers.append({'beer': beer, 'status': 'no_feasible_grid_candidate', 'selected': False,
                              'source_target_distance': before_distance, 'n_candidates': len(candidates)})
            continue
        # Keep historical 0.7/0.3 tradeoff, not tuned on successful transfers.
        chosen = pool[np.argmin(.7 * distances[pool] + .3 * change[pool])]
        transfers.append({'beer': beer, 'status': 'target_style_model_design' if target_like[chosen] else 'partial_toward_target',
            'selected': True, 'source_target_distance': before_distance, 'target_distance': distances[chosen],
            'relative_distance_reduction': 1 - distances[chosen] / before_distance,
            'change_distance': change[chosen], 'fraction': fractions[chosen], 'anchor': parents[chosen],
            'nearest_centroid_style': nearest_style[chosen], 'within_target_radius': distances[chosen] <= tripel_radius,
            'n_candidates': len(candidates), 'n_feasible': int(good.sum()), 'n_target_like': int(target_like.sum()),
            'physical_brewing_validated': False})
        transfer_profiles.append(dict(zip(xf.columns, candidates[chosen]), beer=beer))
        for j, field in zip(protect, config['protected_features']):
            constraints.append({'beer': beer, 'field': field, 'kind': 'exact_coordinate_preservation',
                                'before': source[j], 'after': candidates[chosen, j], 'passed': source[j] == candidates[chosen, j]})
        for name, a, b in [('selected_model_stale_hops', before[stale], prediction[chosen, stale]),
                           ('ridge_stale_hops', guard_before[stale], guard[chosen, stale])]:
            constraints.append({'beer': beer, 'field': name, 'kind': 'sensory_model_proxy', 'before': a, 'after': b, 'passed': b <= a + 1e-10})
        for j, target in enumerate(yf):
            transfer_sensory.append({'beer': beer, 'target': target, 'before': before[j], 'after': prediction[chosen, j]})
        print(f'Q3 transfer {beer}: {transfers[-1]["status"]}', flush=True)
    save_csv(out, 'transfer_results.csv', transfers); save_csv(out, 'transfer_constraints.csv', constraints)
    save_csv(out, 'transfer_profiles.csv', transfer_profiles); save_csv(out, 'transfer_sensory.csv', transfer_sensory)
    write_json(out / 'assumptions.json', {'profile_features': xf.columns.tolist(), 'excluded_derived_features': config['excluded_derived_features'],
        'selected_generator': 'interpolate', 'tripel_distance_radius_p95': tripel_radius,
        'chemical_guarantee': 'Exact supplied-coordinate equality implies equal concentration and threshold exceedance under the same deterministic inverse map; unknown map values are not estimated.',
        'stale_hops_guarantee': 'Nonincrease in two fitted sensory proxies only; not measured stale-hop compounds or panel scores.',
        'scope': '224-dimensional model designs. No mass-balance, fermentation kinetics or experimental brewing validation.'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--audit', type=Path, default=Path('outputs/p0_audit_20261023_final'))
    p.add_argument('--output', type=Path, default=Path('outputs/final_20261024'))
    p.add_argument('--config', type=Path, default=Path('configs/final_protocol.json'))
    p.add_argument('--stage', choices=['q1', 'all'], default='all', help='Train Q1 only or run Q1 and Q3')
    args = p.parse_args(); config = json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=1):
        data = run_q1(args.source, args.audit, config, args.output / 'q1')
        if args.stage == 'all':
            run_q3(*data, config, args.output / 'q3')
    write_json(args.output / 'run_manifest.json', {'config_sha256': sha256(args.config),
        'source': str(args.source.resolve()), 'status': 'completed', 'stage': args.stage,
        'code_sha256': {str(f): sha256(f) for f in [Path(__file__), Path('src/beer_models/final_model.py')]}})


if __name__ == '__main__':
    main()
