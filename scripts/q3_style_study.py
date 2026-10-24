"""Evaluate style models and tighten transfer support without relaxing defect rules."""
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
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.baseline import metric_vectors
from beer_models.final_model import projected_paths
from beer_models.generation import StyleGenerator
from beer_models.style_models import make_style_model, make_stale_model
from complete_modeling import load_q1_artifacts


def run(source, q1, q3, output):
    config_path = Path('configs/q3_style_protocol.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    ids, xf, yf, bundle, splits = load_q1_artifacts(source, Path('outputs/p0_audit_20261023_final'), q1)
    x = xf.to_numpy(float); styles = ids.tasting_category_fine.to_numpy(); target = yf.stale_hops.to_numpy(float)
    oof = {name: np.empty(len(x), dtype=object) for name in ['logistic', 'extra_trees']}
    stale_oof = np.empty(len(x)); mean_oof = np.empty(len(x)); fold_records = []
    for fold, (train, test) in enumerate(splits):
        for name in oof:
            model = make_style_model(name, config, config['seed'] + fold).fit(x[train], styles[train])
            oof[name][test] = model.predict(x[test])
            np.testing.assert_allclose(model.named_steps['impute'].statistics_, np.nanmedian(x[train], axis=0))
        focused = make_stale_model(config, config['seed'] + fold).fit(x[train], target[train])
        stale_oof[test] = focused.predict(x[test]); mean_oof[test] = target[train].mean()
        fold_records.append({'fold': fold, 'n_train': len(train), 'n_test': len(test),
                             'focused_stale_mae': abs(stale_oof[test] - target[test]).mean()})
        print(f'Q3 supervised style fold {fold} complete', flush=True)
    centroid = pd.read_csv('outputs/q3_delivery_20261024/style_centroid_oof.csv', dtype={'beer': str}).set_index('beer')
    oof['nearest_centroid'] = centroid.loc[ids.beer.astype(str), 'nearest_centroid_style'].to_numpy()
    style_metrics = []; by_style = []; rows = []
    for name, predicted in oof.items():
        style_metrics.append({'method': name, 'accuracy': accuracy_score(styles, predicted),
                              'macro_recall': balanced_accuracy_score(styles, predicted)})
        for style in sorted(set(styles)):
            idx = styles == style
            by_style.append({'method': name, 'style': style, 'n': int(idx.sum()), 'recall': (predicted[idx] == style).mean()})
        rows.extend({'beer': ids.beer.iloc[i], 'style': styles[i], 'method': name, 'prediction': predicted[i]} for i in range(len(x)))
    old_pred = pd.read_csv(q1 / 'oof_predictions.csv', dtype={'beer': str})
    old_pred = old_pred.loc[old_pred.model.eq('nested_selected') & old_pred.target.eq('stale_hops')].set_index('beer')
    previous = old_pred.loc[ids.beer.astype(str), 'prediction'].to_numpy()
    stale_metrics = []
    for name, pred in [('mean', mean_oof), ('previous_multioutput', previous), ('focused_single_target', stale_oof)]:
        m = metric_vectors(target[:, None], pred[:, None])
        stale_metrics.append({'method': name, **{k: v[0] for k, v in m.items()}})
    output.mkdir(parents=True, exist_ok=False)
    fitted = {name: make_style_model(name, config, config['seed']).fit(x, styles) for name in ['logistic', 'extra_trees']}
    for name, model in fitted.items():
        joblib.dump(model, output / (name + '.joblib'), compress=3)
    focused = make_stale_model(config, config['seed']).fit(x, target)
    joblib.dump(focused, output / 'focused_stale.joblib', compress=3)
    generated = pd.read_csv(q3 / 'generated_profiles.csv')
    generated_predictions = {name: model.predict(generated[xf.columns].to_numpy(float)) for name, model in fitted.items()}
    generated_diagnostics = generated[['generated_id', 'style']].copy()
    for name, predicted in generated_predictions.items():
        generated_diagnostics[name + '_prediction'] = predicted
    generated_diagnostics['both_match_requested_style'] = ((generated_diagnostics.logistic_prediction == generated_diagnostics['style']) &
                                                          (generated_diagnostics.extra_trees_prediction == generated_diagnostics['style']))
    # Candidate source coordinates and anchor seeds are kept identical to the previous study.
    generator = StyleGenerator().fit(x, styles)
    zall = generator.robust_.transform(generator.x_)
    tripel = generator.x_[generator.indices_['Tripel']]
    seed_config = json.loads((q1 / 'run_manifest.json').read_text(encoding='utf-8'))['config']
    anchor_samples, _ = generator.sample('Tripel', 100, 'interpolate', seed_config['seed'] + 90000)
    anchors = np.vstack([tripel, tripel.mean(axis=0), anchor_samples])
    protect = [xf.columns.get_loc(f) for f in seed_config['protected_features']]
    lower, upper = tripel.min(axis=0), tripel.max(axis=0)
    center = zall[generator.indices_['Tripel']].mean(axis=0)
    stale_index = yf.columns.get_loc('stale_hops')
    transfers = []; profiles = []; sensory = []; constraint_rows = []
    for i in np.flatnonzero(styles == 'Pils/Lager'):
        beer = str(ids.beer.iloc[i])
        if not np.isfinite(x[i, protect]).all():
            transfers.append({'beer': beer, 'selected': False, 'status': 'source_defect_missing'})
            continue
        source_row = generator.x_[i]
        outside = (source_row[protect] < lower[protect] - 1e-10) | (source_row[protect] > upper[protect] + 1e-10)
        if outside.any():
            transfers.append({'beer': beer, 'selected': False, 'status': 'fixed_defect_outside_empirical_target_box',
                'blocking_fields': ';'.join(np.array(xf.columns)[np.array(protect)[outside]])})
            continue
        candidates, fractions, parents = projected_paths(source_row, anchors, protect, config['transfer_steps'])
        in_box = ((candidates >= lower - 1e-10) & (candidates <= upper + 1e-10)).all(axis=1)
        style_ok = np.ones(len(candidates), bool)
        for model in fitted.values():
            style_ok &= model.predict(candidates) == 'Tripel'
        pred = bundle['model'].predict(candidates); before = bundle['model'].predict(source_row[None])[0]
        guard = bundle['guard'].predict(candidates); guard_before = bundle['guard'].predict(source_row[None])[0]
        z = generator.robust_.transform(candidates)
        distance = np.sqrt(np.mean((z - center) ** 2, axis=1)); change = np.sqrt(np.mean((z - zall[i]) ** 2, axis=1))
        source_distance = np.sqrt(np.mean((zall[i] - center) ** 2))
        proxy_ok = (pred[:, stale_index] <= before[stale_index] + 1e-10) & (guard[:, stale_index] <= guard_before[stale_index] + 1e-10)
        good = in_box & style_ok & proxy_ok & (distance < source_distance - 1e-8)
        if not good.any():
            transfers.append({'beer': beer, 'selected': False, 'status': 'no_candidate_passed_strict_model_checks',
                'n_box_passed': int(in_box.sum()), 'n_box_and_style_passed': int((in_box & style_ok).sum()),
                'n_box_style_proxy_passed': int((in_box & style_ok & proxy_ok).sum())})
            continue
        eligible = np.flatnonzero(good); w = config['target_distance_weight']
        k = eligible[np.argmin(w * distance[eligible] + (1 - w) * change[eligible])]
        focused_before, focused_after = focused.predict(np.vstack([source_row, candidates[k]]))
        transfers.append({'beer': beer, 'selected': True, 'status': 'strict_empirical_model_candidate',
            'fraction': fractions[k], 'anchor': int(parents[k]), 'source_target_distance': source_distance, 'target_distance': distance[k],
            'distance_reduction_fraction': 1 - distance[k] / source_distance, 'n_passed': int(good.sum()),
            'focused_stale_delta_diagnostic_only': focused_after - focused_before,
            'style_authenticated': False, 'physical_constraints_certified': False})
        profiles.append(dict(zip(xf.columns, candidates[k]), beer=beer))
        sensory.extend({'beer': beer, 'target': t, 'before': before[j], 'after': pred[k, j]} for j, t in enumerate(yf))
        for j in protect:
            constraint_rows.append({'beer': beer, 'field': xf.columns[j], 'kind': 'exact_preservation', 'before': source_row[j], 'after': candidates[k, j]})
        for name, a, b in [('previous_stale_model', before[stale_index], pred[k, stale_index]),
                           ('ridge_stale_model', guard_before[stale_index], guard[k, stale_index])]:
            constraint_rows.append({'beer': beer, 'field': name, 'kind': 'proxy_nonincrease', 'before': a, 'after': b})
    for name, table in [('style_metrics', style_metrics), ('style_metrics_by_class', by_style), ('style_oof', rows),
                        ('stale_metrics', stale_metrics), ('fold_records', fold_records),
                        ('generated_style_diagnostics', generated_diagnostics), ('strict_transfer_results', transfers),
                        ('strict_transfer_profiles', profiles), ('strict_transfer_sensory', sensory), ('strict_transfer_constraints', constraint_rows)]:
        save_csv(output, name + '.csv', table)
    save_csv(output, 'stale_oof.csv', pd.DataFrame({'beer': ids.beer, 'observed': target, 'mean': mean_oof,
                                                  'previous': previous, 'focused': stale_oof}))
    summary = {'status': 'passed', 'config': config, 'config_sha256': sha256(config_path),
        'source_sha256': {n: sha256(source / n) for n in ['1.csv', '4.csv', '8.xlsx']},
        'style_metrics': style_metrics, 'stale_metrics': stale_metrics,
        'generated_consensus_fraction_by_style': generated_diagnostics.groupby('style').both_match_requested_style.mean().to_dict(),
        'strict_transfer_status_counts': pd.DataFrame(transfers).status.value_counts().to_dict(),
        'checks': ['fold-only imputer statistics', 'original sample folds reused',
                   'strict source blocks before imputation-based defect claims'],
        'interpretation': 'Both classifiers and empirical target box are additional model diagnostics, not true style authentication. Focused stale predictions are diagnostic only; no proxy is promoted to physical certification.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--q1', type=Path, default=Path('outputs/final_20261024/q1'))
    p.add_argument('--q3', type=Path, default=Path('outputs/final_20261024/q3'))
    p.add_argument('--output', type=Path, default=Path('outputs/q3_style_20261024'))
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.q1, args.q3, args.output)
