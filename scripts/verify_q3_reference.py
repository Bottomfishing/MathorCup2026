"""Independently verify measured-reference Q3 profiles and outcome anchoring."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256, write_json
from beer_models.generation import StyleGenerator
from complete_modeling import load_q1_artifacts


def verify(source, q1, style_dir, resolution, output):
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert sha256(resolution / 'summary.json') == summary['parent_resolution_summary_sha256']
    parent = json.loads((resolution / 'summary.json').read_text(encoding='utf-8'))
    ids, xf, yf, bundle, _ = load_q1_artifacts(source, Path('outputs/p0_audit_20261023_final'), q1)
    for name, digest in summary['source_sha256'].items():
        assert sha256(source / name) == digest
    for name, digest in parent['style_models_sha256'].items():
        assert sha256(style_dir / (name + '.joblib')) == digest
    x = xf.to_numpy(float); styles = ids.tasting_category_fine.to_numpy(); beers = ids.beer.astype(str).to_numpy()
    lookup = {b: i for i, b in enumerate(beers)}; g = StyleGenerator().fit(x, styles)
    protected = [xf.columns.get_loc(f) for f in parent['config']['protected_features']]
    other = [j for j in range(len(xf.columns)) if j not in protected]
    center = g.robust_.transform(g.x_[g.indices_['Tripel']]).mean(0)
    complete_references = np.flatnonzero((styles == 'Tripel') & np.isfinite(x).all(axis=1))
    models = {'previous_multioutput': bundle['model'], 'ridge': bundle['guard'],
              'focused': joblib.load(style_dir / 'focused_stale.joblib')}
    classifiers = [joblib.load(style_dir / (name + '.joblib')) for name in ['logistic', 'extra_trees']]
    evidence = pd.read_csv(resolution / 'source_defect_evidence.csv', dtype={'beer': str}).set_index(['beer', 'field'])
    table = pd.read_csv(output / 'transfer_results.csv', dtype={'beer': str, 'reference_beer': str}).set_index('beer')
    profiles = pd.read_csv(output / 'transfer_profiles.csv', dtype={'beer': str}).set_index('beer')
    sensory = pd.read_csv(output / 'transfer_sensory.csv', dtype={'beer': str, 'reference_beer': str}).set_index(['beer', 'target'])
    checks = pd.read_csv(output / 'stale_reference_checks.csv', dtype={'beer': str}).set_index(['beer', 'model'])
    pairs = pd.read_csv(output / 'all_reference_pairs.csv', dtype={'beer': str, 'reference_beer': str})
    assert table.selected.all() and len(table) == 11 and len(sensory) == 550 and len(checks) == 33
    assert set(table.index) == set(beers[styles == 'Pils/Lager']) == set(profiles.index)
    assert len(pairs) == 11 * len(complete_references) == 253
    stale = yf.columns.get_loc('stale_hops')
    for beer, row in table.iterrows():
        i = lookup[beer]; d = lookup[row.reference_beer]
        assert d in complete_references
        source_row = g.x_[i].copy()
        for j in protected:
            source_row[j] = evidence.loc[(beer, xf.columns[j]), 'constraint_cap_coordinate']
        after = profiles.loc[beer, xf.columns].to_numpy(float)
        np.testing.assert_allclose(after[other], x[d, other], rtol=0, atol=1e-10)
        np.testing.assert_allclose(after[protected], np.minimum(source_row[protected], x[d, protected]), rtol=0, atol=1e-10)
        assert (after[protected] <= source_row[protected] + 1e-10).all()
        assert (after[protected] <= x[d, protected] + 1e-10).all()
        assert yf.stale_hops.iloc[d] <= yf.stale_hops.iloc[i]
        for classifier in classifiers:
            assert classifier.predict(after[None])[0] == 'Tripel'
        anchored = []; global_pass = True
        for name, model in models.items():
            source_pred, donor_pred, after_pred = model.predict(np.vstack([source_row, x[d], after]))
            if name != 'focused':
                source_pred, donor_pred, after_pred = source_pred[stale], donor_pred[stale], after_pred[stale]
            a = yf.stale_hops.iloc[d] + after_pred - donor_pred
            assert a <= yf.stale_hops.iloc[i] + 1e-10
            record = checks.loc[(beer, name)]
            np.testing.assert_allclose(record.reference_anchored_after, a, atol=1e-10)
            assert record.passed; anchored.append(a); global_pass &= after_pred <= source_pred + 1e-10
        assert global_pass == row.three_original_proxy_guards_passed_diagnostic
        np.testing.assert_allclose([row.source_observed_stale, row.reference_observed_stale,
            row.max_reference_anchored_after, row.stale_model_margin],
            [yf.stale_hops.iloc[i], yf.stale_hops.iloc[d], max(anchored), yf.stale_hops.iloc[i]-max(anchored)], atol=1e-10)
        donor_pred, after_pred = bundle['model'].predict(np.vstack([x[d], after]))
        s = sensory.loc[beer].loc[yf.columns]
        np.testing.assert_allclose(s.reference_observed, yf.iloc[d], atol=1e-10)
        np.testing.assert_allclose(s.reference_anchored_after, yf.iloc[d].to_numpy()+after_pred-donor_pred, atol=1e-10)
        zsource, zafter = g.robust_.transform(np.vstack([source_row, after]))
        distance = np.sqrt(((zafter-center)**2).mean()); source_distance = np.sqrt(((zsource-center)**2).mean())
        np.testing.assert_allclose([row.target_distance, row.distance_reduction_fraction], [distance, 1-distance/source_distance], atol=1e-10)
        assert distance < source_distance and not row.actual_after_stale_measured
        candidates = pairs.loc[pairs.beer.eq(beer)]
        chosen = candidates.loc[candidates.reference_beer.eq(row.reference_beer)].iloc[0]
        assert chosen.passed_all and int(candidates.passed_all.sum()) == row.n_passed
        assert set(candidates.reference_beer) == set(beers[complete_references])
    result = {'status': 'passed', 'selected_sources': 11, 'complete_measured_references': 23,
        'pairs': len(pairs), 'unchanged_measured_coordinates_per_transfer': 220,
        'source_and_reference_chemical_cap_checks': 88, 'anchored_stale_checks': 33, 'sensory_rows': 550,
        'checks': ['source, parent and model hashes', 'complete reference membership',
            'exact 220 measured coordinates', 'four chemical caps against source and reference',
            'reference observed Stale not higher', 'three anchored Stale corrections',
            'source-prediction guards reported only as diagnostic', 'dual style labels',
            'distance and full sensory reconstruction'],
        'paired_intervention_validation_available': False}
    write_json(output / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--q1', type=Path, default=Path('outputs/final_20261024/q1'))
    p.add_argument('--style-models', type=Path, default=Path('outputs/q3_style_20261024'))
    p.add_argument('--resolution', type=Path, default=Path('outputs/q3_resolution_20261024'))
    p.add_argument('--output', type=Path, default=Path('outputs/q3_reference_20261024'))
    args = p.parse_args()
    with threadpool_limits(limits=1):
        verify(args.source, args.q1, args.style_models, args.resolution, args.output)
