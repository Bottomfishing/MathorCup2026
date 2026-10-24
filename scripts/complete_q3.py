"""Q3 delivery diagnostics and finite-grid sensitivity, reusing existing models."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.transforms import RobustAsinh
from beer_models.final_model import projected_paths
from beer_models.style_diagnostics import style_centers, nearest_centroid, nearest_neighbor_radius, select_transfer
from complete_modeling import load_q1_artifacts


def run(source, audit, q1, q3, output):
    ids, xf, yf, bundle, splits = load_q1_artifacts(source, audit, q1)
    x = xf.to_numpy(float); styles = ids.tasting_category_fine.to_numpy(); p = x.shape[1]
    config = json.loads((q1 / 'run_manifest.json').read_text(encoding='utf-8'))['config']
    generator = joblib.load(q3 / 'generator.joblib')
    np.testing.assert_allclose(generator.x_, generator.impute_.transform(x), rtol=0, atol=0)
    centers = style_centers(generator); zreal = generator.robust_.transform(generator.x_)
    # Evaluate the same nearest-center diagnostic on held-out real samples.
    classification = []
    for fold, (train, test) in enumerate(splits):
        impute = SimpleImputer(strategy='median').fit(x[train])
        real = impute.transform(x[train]); robust = RobustAsinh().fit(real)
        ztrain = robust.transform(real)
        local = {s: ztrain[styles[train] == s].mean(axis=0) for s in sorted(set(styles))}
        prediction = nearest_centroid(robust.transform(impute.transform(x[test])), local)
        classification.extend({'beer': ids.beer.iloc[i], 'fold': fold, 'style': styles[i], 'nearest_centroid_style': s,
                               'correct': styles[i] == s} for i, s in zip(test, prediction))
    classification = pd.DataFrame(classification)
    class_summary = classification.groupby('style').agg(n=('correct', 'size'), recall=('correct', 'mean')).reset_index()
    generated = pd.read_csv(q3 / 'generated_profiles.csv'); metadata = pd.read_csv(q3 / 'generated_metadata.csv', dtype={'parent_1': str, 'parent_2': str})
    zgen = generator.robust_.transform(generated[xf.columns].to_numpy(float))
    nearest = nearest_centroid(zgen, centers); profiles = []
    lookup = {str(beer): i for i, beer in enumerate(ids.beer)}
    # Reconstruct parents and validate generated features independently of the sampler.
    for row in metadata.itertuples():
        i = generated.index[generated.generated_id.eq(row.generated_id)][0]
        a, b = lookup[row.parent_1], lookup[row.parent_2]
        expected = row.weight * generator.x_[a] + (1 - row.weight) * generator.x_[b]
        np.testing.assert_allclose(generated.loc[i, xf.columns].to_numpy(float), expected, rtol=1e-12, atol=1e-9)
        assert styles[a] == styles[b] == generated.loc[i, 'style'] and a != b
    for style, indices in generated.groupby('style').groups.items():
        idx = np.array(indices); real_idx = generator.indices_[style]; real = zreal[real_idx]
        radius = nearest_neighbor_radius(real)
        distances = cdist(zgen[idx], real) / np.sqrt(p); nearest_distance = distances.min(axis=1)
        lower = generator.x_[real_idx].min(axis=0); upper = generator.x_[real_idx].max(axis=0)
        values = generated.loc[idx, xf.columns].to_numpy(float)
        outside = ((values < lower - 1e-10) | (values > upper + 1e-10)).sum(axis=1)
        for i, d, outside_count in zip(idx, nearest_distance, outside):
            profiles.append({'generated_id': generated.loc[i, 'generated_id'], 'style': style,
                'nearest_centroid_style': nearest[i], 'matches_requested_centroid': nearest[i] == style,
                'nearest_real_style_distance': d, 'style_loo_nn_radius_p95': radius,
                'within_style_nn_radius': d <= radius, 'outside_style_feature_range_count': int(outside_count),
                'exact_training_copy': d < 1e-10})
    profiles = pd.DataFrame(profiles)
    generated_sensory = pd.read_csv(q3 / 'generated_sensory.csv')
    actual_prediction = bundle['model'].predict(generated[xf.columns].to_numpy(float))
    saved_prediction = generated_sensory.pivot(index='generated_id', columns='target', values='prediction').loc[generated.generated_id, yf.columns]
    np.testing.assert_allclose(actual_prediction, saved_prediction, atol=1e-10)
    real_predictions = bundle['model'].predict(generator.x_); fingerprints = []
    for style in sorted(generated['style'].unique()):
        group = generated_sensory.loc[generated_sensory['style'].eq(style)]
        real_idx = generator.indices_[style]
        for j, target in enumerate(yf):
            values = group.loc[group.target.eq(target), 'prediction']
            fingerprints.append({'style': style, 'target': target, 'generated_prediction_mean': values.mean(),
                'generated_prediction_p10': values.quantile(.1), 'generated_prediction_p90': values.quantile(.9),
                'real_observed_mean': yf.iloc[real_idx, j].mean(), 'real_training_prediction_mean': real_predictions[real_idx, j].mean(),
                'primary': bool(bundle['primary'][j]), 'n_real': len(real_idx), 'n_generated': len(values)})
    tripel = generator.x_[generator.indices_['Tripel']]; ztripel = zreal[generator.indices_['Tripel']]
    center = centers['Tripel']; radius = np.quantile(np.sqrt(np.mean((ztripel - center) ** 2, axis=1)), .95)
    nn_radius = nearest_neighbor_radius(ztripel)
    interpolated, _ = generator.sample('Tripel', 100, 'interpolate', config['seed'] + 90000)
    anchors = np.vstack([tripel, tripel.mean(axis=0), interpolated])
    protect = [xf.columns.get_loc(f) for f in config['protected_features']]; stale = yf.columns.get_loc('stale_hops')
    original = pd.read_csv(q3 / 'transfer_results.csv', dtype={'beer': str}).set_index('beer')
    transferred = pd.read_csv(q3 / 'transfer_profiles.csv', dtype={'beer': str}).set_index('beer')
    sensitivity = []; transfer_diagnostics = []
    for i in np.flatnonzero(styles == 'Pils/Lager'):
        beer = str(ids.beer.iloc[i])
        if not np.isfinite(x[i, protect]).all():
            for steps in [20, 40, 80]:
                for weight in [.5, .7, .9]:
                    sensitivity.append({'beer': beer, 'steps': steps, 'target_weight': weight, 'status': 'source_defect_missing', 'selected': False})
            assert original.loc[beer, 'status'] == 'source_defect_missing'
            continue
        source_row = generator.x_[i]; zs = zreal[i]
        before = bundle['model'].predict(source_row[None])[0, stale]
        before_guard = bundle['guard'].predict(source_row[None])[0, stale]
        # One 80-step pool contains all points of the 40 and 20-step grids.
        candidates, fractions, parents = projected_paths(source_row, anchors, protect, 80)
        np.testing.assert_allclose(candidates[:, protect], np.broadcast_to(source_row[protect], (len(candidates), len(protect))), atol=0, rtol=0)
        z = generator.robust_.transform(candidates)
        distances = np.sqrt(np.mean((z - center) ** 2, axis=1)); changes = np.sqrt(np.mean((z - zs) ** 2, axis=1))
        before_distance = np.sqrt(np.mean((zs - center) ** 2))
        prediction = bundle['model'].predict(candidates)[:, stale]
        guard = bundle['guard'].predict(candidates)[:, stale]
        good = (prediction <= before + 1e-10) & (guard <= before_guard + 1e-10) & (distances < before_distance - 1e-8)
        nearest_styles = nearest_centroid(z, centers)
        target_like = good & (nearest_styles == 'Tripel') & (distances <= radius)
        for steps in [20, 40, 80]:
            stride = 80 // steps
            indices = np.arange(len(candidates)).reshape(len(anchors), 80)[:, stride - 1::stride].ravel()
            for weight in [.5, .7, .9]:
                local = select_transfer(distances[indices], changes[indices], good[indices], target_like[indices], weight)
                if local is None:
                    sensitivity.append({'beer': beer, 'steps': steps, 'target_weight': weight,
                                         'status': 'no_feasible_grid_candidate', 'selected': False})
                    continue
                k = indices[local]
                sensitivity.append({'beer': beer, 'steps': steps, 'target_weight': weight, 'selected': True,
                    'status': 'target_style_model_design' if target_like[k] else 'partial_toward_target',
                    'anchor': int(parents[k]), 'fraction': fractions[k], 'target_distance': distances[k],
                    'relative_distance_reduction': 1 - distances[k] / before_distance, 'change_distance': changes[k],
                    'selected_model_stale_delta': prediction[k] - before, 'ridge_stale_delta': guard[k] - before_guard})
                if steps == 40 and weight == .7:
                    saved = transferred.loc[beer, xf.columns].to_numpy(float)
                    np.testing.assert_allclose(candidates[k], saved, rtol=1e-12, atol=1e-9)
                    np.testing.assert_allclose(distances[k], original.loc[beer, 'target_distance'], atol=1e-12)
                    after_z = generator.robust_.transform(saved[None])
                    nearest_real = cdist(after_z, ztripel).min() / np.sqrt(p)
                    lower, upper = tripel.min(axis=0), tripel.max(axis=0)
                    outside = (saved < lower - 1e-10) | (saved > upper + 1e-10)
                    transfer_diagnostics.append({'beer': beer, 'nearest_centroid_style': nearest_styles[k],
                        'delivery_status': 'proxy_constraints_passed_style_not_authenticated',
                        'within_target_center_radius': distances[k] <= radius, 'nearest_tripel_distance': nearest_real,
                        'within_target_nn_radius': nearest_real <= nn_radius, 'tripel_loo_nn_radius_p95': nn_radius,
                        'outside_tripel_feature_ranges': int(outside.sum()), 'outside_protected_ranges': int(outside[protect].sum()),
                        'source_imputed_feature_count': int(np.isnan(x[i]).sum()),
                        'selected_model_stale_delta': prediction[k] - before, 'ridge_stale_delta': guard[k] - before_guard})
        print(f'Q3 sensitivity {beer} complete', flush=True)
    sensitivity = pd.DataFrame(sensitivity); transfer_diagnostics = pd.DataFrame(transfer_diagnostics)
    grouped = []
    for (steps, weight), group in sensitivity.groupby(['steps', 'target_weight']):
        selected = group.loc[group.selected]
        grouped.append({'steps': steps, 'target_weight': weight, 'n_sources': len(group), 'n_selected': len(selected),
            'n_target_style_designs': int(group.status.eq('target_style_model_design').sum()),
            'n_missing_defect_sources': int(group.status.eq('source_defect_missing').sum()),
            'mean_distance_reduction_selected': selected.relative_distance_reduction.mean(),
            'mean_change_selected': selected.change_distance.mean()})
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in [('style_centroid_oof', classification), ('style_centroid_by_style', class_summary),
                        ('generated_profile_diagnostics', profiles), ('sensory_fingerprints', pd.DataFrame(fingerprints)),
                        ('transfer_sensitivity', sensitivity), ('transfer_sensitivity_summary', pd.DataFrame(grouped)),
                        ('transfer_diagnostics', transfer_diagnostics)]:
        save_csv(output, name + '.csv', frame)
    metrics = pd.read_csv(q1 / 'metrics_by_target.csv')
    stale_metrics = metrics.loc[metrics.model.eq('nested_selected') & metrics.target.eq('stale_hops')].iloc[0]
    summary = {'status': 'passed', 'generated_profiles': len(profiles), 'sensory_fingerprint_rows': len(fingerprints),
        'generated_centroid_match_by_style': profiles.groupby('style').matches_requested_centroid.mean().to_dict(),
        'generated_nn_radius_pass_by_style': profiles.groupby('style').within_style_nn_radius.mean().to_dict(),
        'generated_exact_training_copies': int(profiles.exact_training_copy.sum()),
        'generated_profiles_outside_style_feature_ranges': int((profiles.outside_style_feature_range_count > 0).sum()),
        'centroid_oof_accuracy': float(classification.correct.mean()), 'centroid_oof_macro_recall': float(class_summary.recall.mean()),
        'default_transfer_model_designs': len(transfer_diagnostics),
        'transfers_outside_target_feature_ranges': int((transfer_diagnostics.outside_tripel_feature_ranges > 0).sum()),
        'transfers_inside_target_nn_radius': int(transfer_diagnostics.within_target_nn_radius.sum()),
        'sensitivity_scenarios': len(grouped), 'sensitivity_source_scenarios': len(sensitivity),
        'stale_hops_oof_mae': float(stale_metrics.mae), 'stale_hops_oof_r2': float(stale_metrics.r2),
        'style_authentication_established': False, 'physical_constraints_certified': False,
        'model_sha256': sha256(q1 / 'model.joblib'), 'generator_sha256': sha256(q3 / 'generator.joblib'),
        'source_sha256': {n: sha256(source / n) for n in ['1.csv', '4.csv', '8.xlsx']},
        'checks': ['Q1 source/model identity', 'all generated parent reconstructions', 'all generated sensory predictions',
                   'all four protected coordinates for every sensitivity candidate',
                   'default transfer selection replayed against stored profiles and distances'],
        'interpretation': 'Nearest-center and NN-radius checks are descriptive diagnostics, not style authentication. Stale hops constraints use imperfect model predictions. Model-space designs are not brewing or physical-threshold certification.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--audit', type=Path, default=Path('outputs/p0_audit_20261023_final'))
    p.add_argument('--q1', type=Path, default=Path('outputs/final_20261024/q1'))
    p.add_argument('--q3', type=Path, default=Path('outputs/final_20261024/q3'))
    p.add_argument('--output', type=Path, default=Path('outputs/q3_delivery_20261024'))
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.audit, args.q1, args.q3, args.output)
