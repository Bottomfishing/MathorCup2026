"""Anchor transfer in measured Tripel profiles instead of predicting a whole style shift."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, sha256, write_json
from beer_models.generation import StyleGenerator
from complete_modeling import load_q1_artifacts


def run(source, q1, style_dir, resolution, output):
    previous = json.loads((resolution / 'summary.json').read_text(encoding='utf-8'))
    ids, xf, yf, bundle, _ = load_q1_artifacts(source, Path('outputs/p0_audit_20261023_final'), q1)
    assert previous['source_sha256'] == {n: sha256(source / n) for n in ['1.csv', '4.csv', '8.xlsx']}
    for name, digest in previous['style_models_sha256'].items():
        assert sha256(style_dir / (name + '.joblib')) == digest
    output.mkdir(parents=True, exist_ok=False)
    x = xf.to_numpy(float); styles = ids.tasting_category_fine.to_numpy()
    g = StyleGenerator().fit(x, styles); center = g.robust_.transform(g.x_[g.indices_['Tripel']]).mean(0)
    donor_indices = np.flatnonzero((styles == 'Tripel') & np.isfinite(x).all(axis=1))
    donor_x = x[donor_indices]; observed = yf.stale_hops.to_numpy(float)
    protected = [xf.columns.get_loc(f) for f in previous['config']['protected_features']]
    evidence = pd.read_csv(resolution / 'source_defect_evidence.csv', dtype={'beer': str}).set_index(['beer', 'field'])
    classifiers = {n: joblib.load(style_dir / (n + '.joblib')) for n in ['logistic', 'extra_trees']}
    models = {'previous_multioutput': bundle['model'], 'ridge': bundle['guard'],
              'focused': joblib.load(style_dir / 'focused_stale.joblib')}
    stale_index = yf.columns.get_loc('stale_hops')
    donor_predictions = {n: m.predict(donor_x) for n, m in models.items()}
    results = []; profiles = []; sensory = []; gates = []; all_pairs = []
    for i in np.flatnonzero(styles == 'Pils/Lager'):
        beer = str(ids.beer.iloc[i]); source_row = g.x_[i].copy()
        for j in protected:
            source_row[j] = evidence.loc[(beer, xf.columns[j]), 'constraint_cap_coordinate']
        # The other 220 coordinates are complete, measured Tripel values.
        candidates = donor_x.copy(); candidates[:, protected] = np.minimum(donor_x[:, protected], source_row[protected])
        style_ok = np.ones(len(candidates), bool)
        for m in classifiers.values():
            style_ok &= m.predict(candidates) == 'Tripel'
        raw_predictions = {n: m.predict(candidates) for n, m in models.items()}
        anchored = {}; global_ok = np.ones(len(candidates), bool)
        for name in models:
            base = donor_predictions[name]; after = raw_predictions[name]
            source_prediction = models[name].predict(source_row[None])[0]
            if name != 'focused':
                base, after, source_prediction = base[:, stale_index], after[:, stale_index], source_prediction[stale_index]
            anchored[name] = observed[donor_indices] + after - base
            global_ok &= after <= source_prediction + 1e-10
        max_anchored = np.max(np.vstack(list(anchored.values())), axis=0)
        donor_observed_ok = observed[donor_indices] <= observed[i]
        anchored_ok = max_anchored <= observed[i] + 1e-10
        z = g.robust_.transform(candidates); zs = g.robust_.transform(source_row[None])[0]
        distance = np.sqrt(((z - center) ** 2).mean(1)); change = np.sqrt(((z - zs) ** 2).mean(1))
        source_distance = np.sqrt(((zs - center) ** 2).mean())
        # Compare to OBSERVED source and reference outcomes. The previous global
        # prediction-vs-prediction gate remains a diagnostic, because its source
        # bias is not evidence against an observed lower-Stale reference.
        good = style_ok & donor_observed_ok & anchored_ok & (distance < source_distance - 1e-8)
        for k, d in enumerate(donor_indices):
            all_pairs.append({'beer': beer, 'reference_beer': str(ids.beer.iloc[d]),
                'source_observed_stale': observed[i], 'reference_observed_stale': observed[d],
                'max_reference_anchored_after': max_anchored[k], 'both_style_models_passed': bool(style_ok[k]),
                'three_original_proxy_guards_passed': bool(global_ok[k]), 'measured_reference_not_worse': bool(donor_observed_ok[k]),
                'all_reference_anchored_models_not_worse': bool(anchored_ok[k]), 'target_distance': distance[k],
                'distance_improved': bool(distance[k] < source_distance - 1e-8), 'passed_all': bool(good[k])})
        if not good.any():
            results.append({'beer': beer, 'selected': False, 'status': 'no_measured_reference_candidate'})
            continue
        w = previous['config']['target_distance_weight']; eligible = np.flatnonzero(good)
        k = eligible[np.argmin(w * distance[eligible] + (1 - w) * change[eligible])]; d = donor_indices[k]
        results.append({'beer': beer, 'selected': True, 'status': 'measured_reference_model_candidate',
            'reference_beer': str(ids.beer.iloc[d]), 'n_complete_references': len(donor_indices), 'n_passed': int(good.sum()),
            'source_target_distance': source_distance, 'target_distance': distance[k],
            'distance_reduction_fraction': 1 - distance[k] / source_distance,
            'source_observed_stale': observed[i], 'reference_observed_stale': observed[d],
            'max_reference_anchored_after': max_anchored[k], 'stale_model_margin': observed[i] - max_anchored[k],
            'three_original_proxy_guards_passed_diagnostic': bool(global_ok[k]),
            'unmodified_measured_coordinates': len(xf.columns) - len(protected),
            'style_authenticated': False, 'actual_after_stale_measured': False,
            'physical_constraints_certified': False})
        profiles.append(dict(zip(xf.columns, candidates[k]), beer=beer))
        for j, target in enumerate(yf.columns):
            sensory.append({'beer': beer, 'reference_beer': str(ids.beer.iloc[d]), 'target': target,
                'source_observed': yf.iloc[i, j], 'reference_observed': yf.iloc[d, j],
                'reference_model': donor_predictions['previous_multioutput'][k, j],
                'after_model': raw_predictions['previous_multioutput'][k, j],
                'reference_anchored_after': yf.iloc[d, j] + raw_predictions['previous_multioutput'][k, j] - donor_predictions['previous_multioutput'][k, j]})
        for name in models:
            gates.append({'beer': beer, 'model': name, 'source_observed_stale': observed[i],
                'reference_observed_stale': observed[d], 'reference_anchored_after': anchored[name][k],
                'passed': bool(anchored[name][k] <= observed[i] + 1e-10)})
        print(f'Measured reference beer {beer} <- {ids.beer.iloc[d]}: {int(good.sum())} choices', flush=True)
    for name, table in [('transfer_results', results), ('transfer_profiles', profiles), ('transfer_sensory', sensory),
                        ('stale_reference_checks', gates), ('all_reference_pairs', all_pairs)]:
        save_csv(output, name + '.csv', table)
    write_json(output / 'summary.json', {'status': 'completed', 'selected': sum(r['selected'] for r in results),
        'n_complete_measured_references': len(donor_indices), 'source_sha256': previous['source_sha256'],
        'parent_resolution_summary_sha256': sha256(resolution / 'summary.json'),
        'assumption': 'The model error at a measured reference remains unchanged after decreasing only four defect coordinates. All three reference-anchored estimates must not exceed the observed source Stale score. This assumption has no paired intervention validation in the attachments.',
        'candidate_rule': 'Complete observed Tripel rows; four defects capped by both source and reference; two style classifiers, three reference-anchored guards, lower measured reference Stale, improved target distance; inherited 0.7/0.3 objective. Original prediction-vs-prediction guards are diagnostic only in this separate observed-reference method.',
        'uncertainty_scope': 'Max of three point estimates is a model-disagreement envelope, not a confidence bound or observed after score.',
        'actual_after_stale_validated': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--q1', type=Path, default=Path('outputs/final_20261024/q1'))
    p.add_argument('--style-models', type=Path, default=Path('outputs/q3_style_20261024'))
    p.add_argument('--resolution', type=Path, default=Path('outputs/q3_resolution_20261024'))
    p.add_argument('--output', type=Path, default=Path('outputs/q3_reference_20261024'))
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.q1, args.style_models, args.resolution, args.output)
