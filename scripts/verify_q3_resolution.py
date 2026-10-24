"""Check Q3 repaired candidates directly against sources, paths and saved predictors."""
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
from beer_models.baseline import metric_vectors
from beer_models.generation import StyleGenerator
from complete_modeling import load_q1_artifacts


def verify(source, q1, style_dir, output):
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert sha256(Path('configs/q3_resolution_protocol.json')) == summary['config_sha256']
    ids, xf, yf, bundle, splits = load_q1_artifacts(source, Path('outputs/p0_audit_20261023_final'), q1)
    for name, digest in summary['source_sha256'].items():
        assert sha256(source / name) == digest
    for name, digest in summary['style_models_sha256'].items():
        assert sha256(style_dir / (name + '.joblib')) == digest
    x = xf.to_numpy(float); styles = ids.tasting_category_fine.to_numpy(); beers = ids.beer.astype(str)
    g = StyleGenerator().fit(x, styles); tripel = g.x_[g.indices_['Tripel']]
    lower, upper = tripel.min(0), tripel.max(0)
    protected = [xf.columns.get_loc(f) for f in summary['config']['protected_features']]
    other = [j for j in range(len(xf.columns)) if j not in protected]
    center = g.robust_.transform(tripel).mean(0)
    samples, _ = g.sample('Tripel', 100, 'interpolate', summary['anchor_seed'])
    anchors = np.vstack([tripel, tripel.mean(0), samples])
    actual_anchors = pd.read_csv(output / 'anchors.csv').set_index('anchor').loc[np.arange(len(anchors)), xf.columns]
    np.testing.assert_allclose(anchors, actual_anchors, rtol=0, atol=1e-10)
    cal = pd.read_excel(source / '8.xlsx', sheet_name='Calibrated chemical dataset').set_index('beer')
    cal.index = cal.index.astype(str)
    results = pd.read_csv(output / 'transfer_results.csv', dtype={'beer': str}).set_index('beer')
    profiles = pd.read_csv(output / 'transfer_profiles.csv', dtype={'beer': str}).set_index('beer')
    sensory = pd.read_csv(output / 'transfer_sensory.csv', dtype={'beer': str}).set_index(['beer', 'target'])
    chemical = pd.read_csv(output / 'defect_checks.csv', dtype={'beer': str}).set_index(['beer', 'field'])
    proxy = pd.read_csv(output / 'stale_proxy_checks.csv', dtype={'beer': str}).set_index(['beer', 'model'])
    evidence = pd.read_csv(output / 'source_defect_evidence.csv', dtype={'beer': str}).set_index(['beer', 'field'])
    pils = np.flatnonzero(styles == 'Pils/Lager')
    assert set(results.index) == set(beers.iloc[pils]) == set(profiles.index)
    assert results.selected.all() and len(results) == summary['selected'] == 11
    assert len(sensory) == 11 * len(yf.columns) and len(chemical) == len(evidence) == 44 and len(proxy) == 33
    models = {'previous_multioutput': bundle['model'], 'ridge': bundle['guard'],
              'focused': joblib.load(style_dir / 'focused_stale.joblib')}
    classifiers = {name: joblib.load(style_dir / (name + '.joblib')) for name in ['logistic', 'extra_trees']}
    for i in pils:
        beer = beers.iloc[i]; record = results.loc[beer]; before = g.x_[i].copy()
        for j in protected:
            field = xf.columns[j]; e = evidence.loc[(beer, field)]
            if np.isnan(x[i, j]):
                assert field == 'X2.3.butanedione' and cal.loc[beer, field] == 0
                assert e.basis == 'reference_zero_cap_for_nonnegative_diacetyl' and np.isnan(e.csv_coordinate)
                before[j] = 0.
            else:
                assert e.basis == 'observed_coordinate' and e.csv_coordinate == x[i, j]
            assert e.constraint_cap_coordinate == before[j] and not e.raw_measurement_recovered
            assert e.attachment8_calibrated_value == cal.loc[beer, field]
        expected = before + record.fraction * (anchors[int(record.anchor)] - before)
        expected[protected] = before[protected] if record.policy == 'preserve' else np.minimum(expected[protected], before[protected])
        after = profiles.loc[beer, xf.columns].to_numpy(float)
        np.testing.assert_allclose(after, expected, rtol=0, atol=1e-10)
        assert (after[other] >= lower[other] - 1e-10).all() and (after <= upper + 1e-10).all()
        assert (after[protected] <= before[protected] + 1e-10).all()
        for j in protected:
            row = chemical.loc[(beer, xf.columns[j])]
            assert row.passed
            np.testing.assert_allclose([row.before_bound, row.after], [before[j], after[j]], atol=1e-10)
        for classifier in classifiers.values():
            assert classifier.predict(after[None])[0] == 'Tripel'
        stale_index = yf.columns.get_loc('stale_hops')
        for name, model in models.items():
            a, b = model.predict(np.vstack([before, after]))
            if name != 'focused':
                a, b = a[stale_index], b[stale_index]
            assert b <= a + 1e-10
            row = proxy.loc[(beer, name)]
            np.testing.assert_allclose([row.source_prediction, row.after_prediction, row.predicted_change,
                row.residual_anchored_after], [a, b, b-a, yf.stale_hops.iloc[i]+b-a], atol=1e-10)
            assert row.passed_proxy_nonincrease and not row.is_observed_after
        a, b = bundle['model'].predict(np.vstack([before, after]))
        table = sensory.loc[beer].loc[yf.columns]
        np.testing.assert_allclose(table.source_observed, yf.iloc[i], atol=1e-10)
        np.testing.assert_allclose(table.source_model, a, atol=1e-10)
        np.testing.assert_allclose(table.after_model, b, atol=1e-10)
        np.testing.assert_allclose(table.model_change, b-a, atol=1e-10)
        np.testing.assert_allclose(table.residual_anchored_after, yf.iloc[i].to_numpy()+b-a, atol=1e-10)
        zsource, zafter = g.robust_.transform(np.vstack([before, after]))
        dsource = np.sqrt(((zsource-center)**2).mean()); dafter = np.sqrt(((zafter-center)**2).mean())
        np.testing.assert_allclose([record.source_target_distance, record.target_distance, record.distance_reduction_fraction],
                                   [dsource, dafter, 1-dafter/dsource], atol=1e-10)
        assert dafter < dsource and not record.style_authenticated and not record.stale_observed_nonworsening_verified
    oof = pd.read_csv(output / 'stale_loss_oof.csv', dtype={'beer': str}).set_index('beer').loc[beers]
    metrics = pd.read_csv(output / 'stale_loss_comparison.csv').set_index('method')
    target = yf.stale_hops.to_numpy(float)
    for name in metrics.index:
        for key, value in metric_vectors(target[:, None], oof[[name]].to_numpy()).items():
            np.testing.assert_allclose(value[0], metrics.loc[name, key], atol=1e-12)
    for train, test in splits:
        np.testing.assert_allclose(oof['median'].to_numpy()[test], np.median(target[train]), atol=1e-12)
    # The preliminary same-protocol run is local only, not a required input on new machines.
    result = {'status': 'passed', 'transfers_verified': len(results), 'chemical_constraints': 44,
        'stale_proxy_constraints': 33, 'sensory_rows': len(sensory), 'reference_zero_capped': ['119', '28'],
        'checks': ['source/model hashes', 'reference zero provenance without raw measurement recovery',
            '129 anchors regenerated', 'independent path reconstruction', 'defect-aware target bounds',
            'dual style labels', 'all four chemical bounds and three stale proxies',
            'source-observed and residual-anchored sensory values', 'OOF metrics and median baseline'],
        'actual_post_transfer_stale_validated': False}
    write_json(output / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--q1', type=Path, default=Path('outputs/final_20261024/q1'))
    parser.add_argument('--style-models', type=Path, default=Path('outputs/q3_style_20261024'))
    parser.add_argument('--output', type=Path, default=Path('outputs/q3_resolution_20261024'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        verify(args.source, args.q1, args.style_models, args.output)
