"""Reconstruct every new candidate and replay official selected-plan predictions."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256, write_json
from beer_models.local_transfer import decode, defect_predictions, rank_candidates
from beer_models.q3_raw import raw_predictions
from q3_stage2 import setup, evaluate, source_records
from r2_raw_q1 import RAW_MANIFEST, content_hash


def run(args):
    p, d, b, g, classifiers, z, radius, center, target_indices, protected, dictionary, domains = setup(args.source, args.raw, args.q1)
    out = args.output
    read = lambda name: pd.read_csv(out/name, float_precision='round_trip')
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert summary['q1_model_sha256'] == sha256(args.q1/'model.joblib')
    assert summary['protocol_sha256'] == content_hash(p)
    assert p == json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    for name, digest in summary['source_sha256'].items():
        assert sha256(args.source/name) == digest
    assert sha256(args.raw) == summary['raw_source_sha256']
    for name, digest in summary['evidence_sha256'].items():
        assert sha256(ROOT/name.replace('\\', '/')) == digest
    sources, metadata, _, _ = source_records(d, b, p)
    saved_sources = read('source_profiles.csv').set_index('beer')
    for beer, values in sources.items():
        np.testing.assert_allclose(saved_sources.loc[beer, d['features']], values, atol=1e-12, rtol=0)
    checks, selected, parameters = read('candidate_checks.csv.gz'), read('selected_profiles.csv'), read('selected_parameters.csv')
    results, neighbors = read('transfer_results.csv'), read('neighbors.csv')
    lookup = {int(beer): i for i, beer in enumerate(d['ids'].beer)}
    bounds = (g.x_[target_indices].min(0), g.x_[target_indices].max(0))
    replayed, maximum = 0, 0.
    for (beer, domain), table in checks.groupby(['beer', 'domain'], sort=False):
        source = sources[beer]
        ids = neighbors.loc[neighbors.beer.eq(beer)&neighbors.domain.eq(domain)].sort_values('slot').target_beer
        target_x = g.x_[[lookup[int(i)] for i in ids]]
        dist = np.square(z[target_indices][:, domains[domain]]-g.robust_.transform(source[None])[0, domains[domain]]).mean(1)
        expected = d['ids'].beer.iloc[target_indices[np.argsort(dist, kind='stable')[:5]]].to_numpy()
        np.testing.assert_array_equal(ids, expected)
        u = table[[f'u{j}' for j in range(6)]].to_numpy(float)
        x, progress, weights = decode(u, source, target_x, domains[domain])
        locked = np.setdiff1d(np.arange(len(source)), domains[domain])
        np.testing.assert_allclose(x[:, locked], np.broadcast_to(source[locked], (len(x), len(locked))), atol=0, rtol=0)
        lower = np.minimum(source, target_x.min(0)); upper = np.maximum(source, target_x.max(0))
        assert (x >= lower-1e-12).all() and (x <= upper+1e-12).all()
        before = defect_predictions(b, source[None], p['defects'])
        fresh = evaluate(x, u, source, before, p, d, b, g, classifiers, z, radius, center, protected, bounds)
        for col in fresh:
            if fresh[col].dtype.kind == 'b':
                np.testing.assert_array_equal(fresh[col], table[col])
            else:
                diff = float(abs(fresh[col].to_numpy()-table[col].to_numpy()).max())
                maximum = max(maximum, diff)
                np.testing.assert_allclose(fresh[col], table[col], atol=1e-10, rtol=0, err_msg=col)
        passes = table.loc[table.main_computational_pass]
        chosen = int(passes.sort_values(['objective', 'candidate']).iloc[0].candidate) if len(passes) else int(table.iloc[rank_candidates(table)[0]].candidate)
        r = results.loc[results.beer.eq(beer)&results.domain.eq(domain)].iloc[0]
        assert chosen == r.selected_candidate and len(passes) == r.main_candidate_count
        assert bool(r.selected_main_pass) == bool(len(passes))
        s = selected.loc[selected.beer.eq(beer)&selected.domain.eq(domain)].iloc[0]
        np.testing.assert_allclose(s[d['features']].to_numpy(float), x[chosen], atol=1e-12, rtol=0)
        pp = parameters.loc[parameters.beer.eq(beer)&parameters.domain.eq(domain)].iloc[0]
        np.testing.assert_array_equal(pp[[f'u{j}' for j in range(6)]].to_numpy(float), u[chosen])
        replayed += len(table)
    # Independent original public prediction path: all 11 before and 22 selected after.
    official_x = np.vstack([saved_sources[d['features']].to_numpy(float), selected[d['features']].to_numpy(float)])
    full = raw_predictions(b, official_x, d['parents'])
    fast = defect_predictions(b, official_x, p['defects'])
    idx = [d['targets'].index(t) for t in p['defects']]
    for key in full:
        np.testing.assert_allclose(full[key][:, idx], fast[key], atol=1e-12, rtol=0)
    for name, n in [('transfer_sensory52.csv.gz', 33), ('generated_sensory52.csv.gz', 300)]:
        scores = read(name)
        assert len(scores) == n*52 and scores.groupby('beer').size().eq(52).all()
        missing = scores.target.isin(['A_acetaldehyde', 'F_acetaldehyde'])
        assert scores.loc[missing, 'estimated_grade_0_6'].isna().all() and missing.sum() == n*2
        assert scores.loc[~missing, 'estimated_grade_0_6'].between(0, 6).all()
        assert not scores.actual_after_measured.any() and not scores.safety_certified.any()
    boot = read('bootstrap_checks.csv')
    draws = read('bootstrap_draws.csv')
    assert len(boot) == 22*p['bootstrap_repeats'] and not boot.calibration_applied.any()
    assert len(draws) == 250*p['bootstrap_repeats']
    for repeat, draw in draws.groupby('repeat'):
        expected = d['ids'].tasting_category_fine.value_counts().sort_index()
        actual = d['ids'].set_index('beer').loc[draw.beer, 'tasting_category_fine'].value_counts().sort_index()
        np.testing.assert_array_equal(expected, actual)
        in_bag = boot.loc[boot.repeat.eq(repeat)]
        np.testing.assert_array_equal(in_bag.source_in_bag, in_bag.beer.isin(draw.beer))
    assert len(read('parameter_sensitivity.csv')) == 22*32
    assert replayed == summary['new_candidate_count'] == 22638
    assert int(results.loc[results.domain.eq('yeast_block'), 'selected_main_pass'].sum()) == summary['main_conditional_plans'] == 0
    assert int(results.loc[results.domain.eq('expanded_sensitivity'), 'selected_main_pass'].sum()) == summary['expanded_sensitivity_plans'] == 6
    assert not results.all_five_true_requirements_certified.any() and not results.actual_after_measured.any()
    assert not results.stale_nonworsening_established.any() and not results.physical_feasibility_certified.any()
    assessment = read('plan_assessment.csv')
    assert len(assessment) == 22 and not assessment.adopted_as_reliable_transfer.any()
    fingerprint = read('generated_fingerprints52.csv')
    assert len(fingerprint) == 3*52 and fingerprint.generated_spectra.eq(100).all()
    assert fingerprint.loc[fingerprint.target.isin(['A_acetaldehyde', 'F_acetaldehyde']), 'mean_model_grade'].isna().all()
    record = dict(status='passed', reconstructed_and_five_defect_repredicted_new_points=replayed,
        fresh_style_classifier_fits=2, independent_full_official_path_points=len(official_x),
        maximum_reconstruction_numeric_difference=maximum, old_grid_control_points=33,
        old_full_grid_independently_repredicted=False, bootstrap_refits_independently_repeated=False,
        scientific_scope='Numerical implementation and fixed-plan sensitivity; no physical or actual sensory nonworsening certification')
    write_json(out/'verification.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_stage2_20261026')
    with threadpool_limits(limits=1):
        run(parser.parse_args())
