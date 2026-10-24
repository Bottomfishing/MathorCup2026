"""Check saved continuation outputs against sources and locally trained models."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256, write_json
from beer_models.baseline import load_data, metric_vectors, mask_observed


def direct_grades(reference, values):
    percentile = ((reference[:, None, :] < values[None, :, :]).sum(axis=0) +
                  .5 * (reference[:, None, :] == values[None, :, :]).sum(axis=0)) / len(reference)
    return np.minimum(6, np.floor(7 * percentile)).astype(int)


def verify(source, audit, out):
    q1, q3 = out / 'q1', out / 'q3'
    manifest = json.loads((q1 / 'run_manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['source_sha256'].items():
        assert sha256(source / name) == digest, name
    assert sha256(q1 / 'model.joblib') == manifest['model_sha256']
    ids, xf, yf, primary_names = load_data(source, audit)
    features = manifest['features']; targets = manifest['targets']
    x, y = xf[features].to_numpy(float), yf[targets].to_numpy(float)
    primary = np.array([c in primary_names for c in targets])
    saved = pd.read_csv(q1 / 'oof_predictions.csv', dtype={'beer': str})
    scores = pd.read_csv(q1 / 'q1_scores_all50_long.csv', dtype={'beer': str})
    splits = pd.read_csv(q1 / 'splits.csv', dtype={'beer': str})
    np.testing.assert_array_equal(splits.beer, ids.beer.astype(str))
    config = manifest['config']; pred = np.empty_like(y); truth_grades = np.empty_like(y, int)
    missing = {(r, k): np.empty_like(y) for r in config['missing_rates'] for k in range(config['missing_repeats'])}
    selected = saved.loc[saved.model.eq('nested_selected')]
    assert not selected.duplicated(['beer', 'target']).any() and len(selected) == y.size
    for fold in sorted(splits.outer_test_fold.unique()):
        train = np.flatnonzero(splits.outer_test_fold.ne(fold)); test = np.flatnonzero(splits.outer_test_fold.eq(fold))
        model = joblib.load(q1 / f'fold{fold}.joblib'); pred[test] = model.predict(x[test])
        pipe = model.regressor_ if hasattr(model, 'regressor_') else model
        np.testing.assert_allclose(pipe.named_steps['impute'].statistics_, np.nanmedian(x[train], axis=0))
        actual = selected.pivot(index='beer', columns='target', values='prediction').loc[ids.beer.iloc[test].astype(str), targets].to_numpy()
        np.testing.assert_allclose(pred[test], actual, rtol=0, atol=1e-10)
        expected = direct_grades(y[train], pred[test]); truth_grades[test] = direct_grades(y[train], y[test])
        actual_grades = scores.pivot(index='beer', columns='target', values='relative_grade_0_6').loc[ids.beer.iloc[test].astype(str), targets]
        np.testing.assert_array_equal(expected, actual_grades)
        for rate, repeat in missing:
            masked, _ = mask_observed(x[test], rate, config['seed'] + 1000 + int(fold) * 100 + repeat)
            missing[rate, repeat][test] = model.predict(masked)
    expected_metrics = metric_vectors(y, pred)
    summary = pd.read_csv(q1 / 'model_comparison.csv').set_index('model')
    for key, value in expected_metrics.items():
        np.testing.assert_allclose(value[primary].mean(), summary.loc['nested_selected', 'macro_' + key], atol=1e-12)
    stress = pd.read_csv(q1 / 'missingness.csv').set_index(['additional_missing_rate', 'repeat'])
    for key, values in missing.items():
        np.testing.assert_allclose(abs(values[:, primary] - y[:, primary]).mean(), stress.loc[key, 'macro_mae'], atol=1e-12)
    bundle = joblib.load(q1 / 'model.joblib')
    generated = pd.read_csv(q3 / 'generated_profiles.csv')
    metadata = pd.read_csv(q3 / 'generated_metadata.csv', dtype={'parent_1': str, 'parent_2': str})
    generator = joblib.load(q3 / 'generator.joblib')
    np.testing.assert_allclose(generator.impute_.statistics_, np.nanmedian(x, axis=0))
    lookup = {str(beer): i for i, beer in enumerate(ids.beer)}
    for row in metadata.itertuples():
        a, b = lookup[row.parent_1], lookup[row.parent_2]
        expected = row.weight * generator.x_[a] + (1 - row.weight) * generator.x_[b]
        actual = generated.set_index('generated_id').loc[row.generated_id, features].to_numpy(float)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
    generated_pred = bundle['model'].predict(generated[features].to_numpy(float))
    sensory = pd.read_csv(q3 / 'generated_sensory.csv')
    actual = sensory.pivot(index='generated_id', columns='target', values='prediction').loc[generated.generated_id, targets]
    np.testing.assert_allclose(generated_pred, actual, rtol=0, atol=1e-10)
    transfers = pd.read_csv(q3 / 'transfer_results.csv', dtype={'beer': str})
    profiles = pd.read_csv(q3 / 'transfer_profiles.csv', dtype={'beer': str}).set_index('beer')
    checks = pd.read_csv(q3 / 'transfer_constraints.csv')
    assert checks.passed.all()
    protect = [features.index(c) for c in config['protected_features']]; stale = targets.index('stale_hops')
    center = generator.robust_.transform(generator.x_[generator.indices_['Tripel']]).mean(axis=0)
    for row in transfers.itertuples():
        i = lookup[row.beer]
        if not row.selected:
            assert row.status == 'source_defect_missing' and np.isnan(x[i, protect]).any()
            continue
        before = generator.impute_.transform(x[[i]])
        after = profiles.loc[[row.beer], features].to_numpy(float)
        np.testing.assert_array_equal(before[:, protect], after[:, protect])
        for model in [bundle['model'], bundle['guard']]:
            assert model.predict(after)[0, stale] <= model.predict(before)[0, stale] + 1e-10
        distance = np.sqrt(np.mean((generator.robust_.transform(after)[0] - center) ** 2))
        np.testing.assert_allclose(distance, row.target_distance, atol=1e-12)
        assert distance < row.source_target_distance
    result = {'status': 'passed', 'q1_oof_rows': len(selected), 'q1_features': len(features),
              'q1_primary_targets': int(primary.sum()), 'missing_scenarios': len(missing),
              'generated_profiles': len(generated), 'generated_sensory_rows': len(sensory),
              'selected_transfers': int(transfers.selected.sum()), 'missing_defect_sources': int((~transfers.selected).sum()),
              'checks': ['source hashes', 'fold model reload and OOF predictions', 'training-only imputation statistics',
                         'independent count-based grades', 'metrics and missingness recomputation',
                         'generation parent reconstruction', 'sensory predictions', 'transfer defect equality and dual proxy checks'],
              'limitations': 'Does not establish original rating recovery, causal effects, brewing feasibility or final competition acceptance.'}
    write_json(out / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--audit', type=Path, default=Path('outputs/p0_audit_20261023_final'))
    p.add_argument('--output', type=Path, default=Path('outputs/final_20261024'))
    args = p.parse_args()
    with threadpool_limits(limits=1):
        verify(args.source, args.audit, args.output)
