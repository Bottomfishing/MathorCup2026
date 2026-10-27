"""Replay the adopted sparse path and fixed fresh checks without new fitting."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256, write_json
from beer_models.robust_transfer import panel_five, refit_metrics
from beer_models.q3_raw import raw_predictions
from beer_models.local_transfer import defect_predictions
from q3_robust_transfer import evaluate, chemical_limits, ensemble_arrays, OLD
from beer_models.robust_transfer import parameter_draws
from q3_risk_aware_pruning import FIRST, surplus
from q3_stage2 import setup, source_records
from r2_raw_q1 import RAW_MANIFEST, content_hash


def run(args):
    out = args.output
    read = lambda name: pd.read_csv(out/name, float_precision='round_trip')
    r, p = [json.loads((out/name).read_text(encoding='utf-8')) for name in ['protocol.json', 'base_protocol.json']]
    seal = json.loads((out/'sealed_selection.json').read_text(encoding='utf-8'))
    assert seal['protocol_sha256'] == content_hash(r)
    for name, digest in seal['selection_sha256'].items(): assert sha256(out/name) == digest
    _, d, b, g, classifiers, z, radius, center, _, protected, dictionary, _ = setup(args.source, args.raw, args.q1)
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert summary['q1_model_sha256'] == sha256(args.q1/'model.joblib')
    assert summary['raw_sha256'] == sha256(args.raw)
    for name, digest in summary['source_sha256'].items(): assert sha256(args.source/name) == digest
    old_p = json.loads((ROOT/'configs/q3_stage2_protocol_20261026.json').read_text(encoding='utf-8'))
    sources, _, _, _ = source_records(d, b, old_p)
    indices = [d['targets'].index(t) for t in old_p['defects']]
    reducible = np.array([d['features'].index(f) for f in p['reducible_fields']])
    reference = pd.read_excel(args.source/'8.xlsx', sheet_name='Calibrated chemical dataset').set_index('beer')
    reference.index = reference.index.astype(int)
    development = [joblib.load(FIRST/'models'/f'{part}_{i}.joblib') for part, n in [('design', 8), ('validation', 12)] for i in range(n)]
    fresh = [joblib.load(out/'models'/f'validation_{i}.joblib') for i in range(12)]
    selected, assessment, design, old_sparse = read('selected_profiles.csv').set_index('beer'), read('plan_assessment.csv'), read('design_assessment.csv').set_index('beer'), pd.read_csv(FIRST/'selected_profiles.csv', float_precision='round_trip').set_index(['beer', 'budget'])
    old_broad = pd.read_csv(OLD/'selected_profiles.csv', float_precision='round_trip').loc[lambda t: t.domain.eq('expanded_sensitivity')].set_index('beer')
    old_final = pd.read_csv(FIRST/'final_choices_before_validation.csv').set_index('beer')
    vrows = read('validation_checks.csv')
    stress_replays, greedy_rounds = 0, 0
    for ordinal, (beer, source) in enumerate(sources.items()):
        lower, _ = chemical_limits(source, d, reference, p)
        x = selected.loc[beer, d['features']].to_numpy(float)
        assert (x[protected] <= source[protected]+1e-10).all()
        assert x[protected[2]] == source[protected[2]]
        assert (abs(x-source) > 1e-10).sum() <= 144
        before_dev = [panel_five(model['panel'], source[None], indices) for model in development]
        def check(values, budget=144):
            t = evaluate(values, source, b, indices, classifiers, g, z, center, radius, protected, lower, reducible, development, before_dev, p, budget)
            t['guard_surplus'] = surplus(t)
            return t
        nominal = check(x[None]).iloc[0]
        for col in nominal.index:
            value = design.loc[beer, col]
            if isinstance(nominal[col], (bool, np.bool_)): assert bool(value) == bool(nominal[col])
            else: np.testing.assert_allclose(value, nominal[col], atol=1e-10, rtol=0)
        if bool(design.loc[beer, 'design_selected']):
            label = design.loc[beer, 'selected_seed']
            target_name, level = label.split('_reduction_')
            target_beer = int(target_name.split('_')[1]); level = float(level)
            pos = int(np.flatnonzero(d['ids'].beer.to_numpy() == target_beer)[0])
            current = g.x_[pos].copy(); current[protected[2]] = source[protected[2]]
            current[reducible] = source[reducible]-level*(source[reducible]-lower)
            while (abs(current-source) > 1e-10).sum() > 144:
                zs, zc = g.robust_.transform(source[None])[0], g.robust_.transform(current[None])[0]
                fields = np.flatnonzero((abs(current-source) > 1e-10)&~np.isin(np.arange(224), protected))
                ordered = fields[np.argsort(abs(zc[fields]-zs[fields]), kind='stable')]
                moved = False
                for size in [16, 8, 4, 1]:
                    blocks = [ordered[j:j+size] for j in range(0, len(ordered), size)]
                    batch = np.broadcast_to(current, (len(blocks), 224)).copy()
                    for j, block in enumerate(blocks): batch[j, block] = source[block]
                    tests = check(batch, 224)
                    good = tests.loc[tests.qualified].sort_values(['changed_fields', 'guard_surplus', 'objective'], ascending=[True, False, True])
                    if len(good): current = batch[int(good.index[0])]; moved = True; break
                assert moved, 'Selected sparse path could not be replayed'
                greedy_rounds += 1
                assert greedy_rounds <= 80
            np.testing.assert_allclose(current, x, atol=1e-12, rtol=0)
            seed_index = [f'Tripel_{int(d["ids"].beer.iloc[i])}_reduction_{float(t)}' for i in g.indices_['Tripel'] for t in r['chemical_seed_levels']].index(label)
            design_jitter = check(parameter_draws(source, x, 8, r['seed']+ordinal*100+seed_index))
            assert int(design_jitter.qualified.sum()) == int(design.loc[beer, 'design_stress_passes']) >= 7
            stress = check(parameter_draws(source, x, 32, r['seed']+90000+ordinal))
            stress_replays += 40
            saved = assessment.loc[assessment.beer.eq(beer)&assessment.method.eq('risk_aware_pruned')].iloc[0]
            np.testing.assert_allclose(saved.parameter_qualified_fraction, stress.qualified.mean(), atol=0, rtol=0)
        profiles = np.vstack([x, old_sparse.loc[(beer, int(old_final.loc[beer, 'budget'])), d['features']].to_numpy(float), old_broad.loc[beer, d['features']].to_numpy(float)])
        before = [panel_five(model['panel'], source[None], indices) for model in fresh]
        arrays = ensemble_arrays(fresh, profiles, before, indices)
        metrics = refit_metrics(*arrays, p['validation_adoption'])
        for j, method in enumerate(['risk_aware_pruned', 'previous_sparse', 'previous_broad']):
            record = assessment.loc[assessment.beer.eq(beer)&assessment.method.eq(method)].iloc[0]
            np.testing.assert_allclose(record.validation_joint_fraction, metrics['joint_fraction'][j], atol=0, rtol=0)
            assert bool(record.validation_qualified) == bool(metrics['qualified'][j])
            checks = vrows.loc[vrows.beer.eq(beer)&vrows.method.eq(method)].sort_values('repeat')
            np.testing.assert_array_equal(checks.source_oob, [beer not in model['sampled_beers'] for model in fresh])
    current = read('current_plan_status.csv')
    expected = current.design_selected&current.validation_qualified&(current.parameter_qualified_fraction >= .8)
    np.testing.assert_array_equal(expected, current.stable_four_proxy_conditional)
    assert current.loc[current.stable_four_proxy_conditional, 'beer'].tolist() == [119]
    assert not current.all_five_true_nonworsening_certified.any() and not current.actual_after_measured.any()
    scores = read('sensory52.csv.gz'); missing = scores.target.isin(['A_acetaldehyde', 'F_acetaldehyde'])
    assert len(scores) == 22*52 and missing.sum() == 44 and scores.loc[missing, 'estimated_grade_0_6'].isna().all()
    assert not scores.actual_after_measured.any() and not scores.safety_certified.any()
    values = selected[d['features']].to_numpy(float)
    official, subset = raw_predictions(b, values, d['parents']), defect_predictions(b, values, old_p['defects'])
    for key in subset: np.testing.assert_allclose(subset[key], official[key][:, indices], atol=1e-12, rtol=0)
    chemical = read('protected_chemical_checks.csv')
    assert int((reference.loc[list(sources), 'X4VG'] == 0).sum()) == 9
    assert set(reference.loc[list(sources)].loc[lambda t:t.X4VG > 0].index) == {230, 119}
    assert chemical.loc[chemical.mapping_status.eq('zero_censored_reference_no_inverse'), 'concentration_after'].isna().all()
    corrected = ROOT/'outputs/q3_positive4vg_20261027'
    correction_seal = json.loads((corrected/'sealed_selection.json').read_text(encoding='utf-8'))
    for name, digest in correction_seal['selection_sha256'].items(): assert sha256(corrected/name) == digest
    write_json(out/'verification.json', dict(status='passed', cached_development_models=20, cached_fresh_validation_models=12,
        selected_and_controls_fresh_replay_points=33, adopted_case_greedy_deletion_rounds_replayed=greedy_rounds,
        adopted_case_parameter_draws_replayed=stress_replays, full_official_Q1_selected_replay_points=11,
        current_stability_qualified_four_proxy_cases=[119], data_source4vg_censored_count=9,
        source_positive4vg_beers=[230, 119], independent_refits_in_verifier=0,
        causal_or_all_five_safety_certified=False, all_selection_hashes_preserved=True))
    print('Pruned conditional case and matched refit checks passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_pruning_20261027')
    with threadpool_limits(limits=1): run(parser.parse_args())
