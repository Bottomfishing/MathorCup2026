"""Source-specific4VG scale correction with frozen model and stress gates."""
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
from beer_audit.audit import save_csv, sha256, write_json
from beer_models.robust_transfer import panel_five, refit_metrics
from q3_robust_transfer import chemical_limits, evaluate, ensemble_arrays
from q3_risk_aware_pruning import parameter_draws, surplus, FIRST
from q3_stage2 import setup, source_records
from r2_raw_q1 import content_hash, RAW_MANIFEST

PROTOCOL = ROOT/'configs/q3_pruning_corrected_protocol_20261027.json'


def run(args):
    r = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    p = json.loads((FIRST/'protocol.json').read_text(encoding='utf-8'))
    p['reducible_fields'], p['allow_positive_4vg_reduction'] = r['reducible_fields'], True
    out = args.output
    if not args.validate_only:
        out.mkdir(parents=True, exist_ok=False)
        write_json(out/'protocol.json', r)
    _, d, b, g, classifiers, z, radius, center, _, protected, dictionary, _ = setup(args.source, args.raw, args.q1)
    old_p = json.loads((ROOT/'configs/q3_stage2_protocol_20261026.json').read_text(encoding='utf-8'))
    sources, _, _, _ = source_records(d, b, old_p)
    indices = [d['targets'].index(t) for t in old_p['defects']]
    reducible = np.array([d['features'].index(f) for f in p['reducible_fields']])
    reference = pd.read_excel(args.source/'8.xlsx', sheet_name='Calibrated chemical dataset').set_index('beer')
    reference.index = reference.index.astype(int)
    development = [joblib.load(FIRST/'models'/f'{part}_{i}.joblib') for part, n in [('design', 8), ('validation', 12)] for i in range(n)]
    prior = pd.read_csv(FIRST/'selected_profiles.csv', float_precision='round_trip').set_index(['beer', 'budget'])
    final = pd.read_csv(FIRST/'final_choices_before_validation.csv').set_index('beer')
    if not args.validate_only:
        selected, assessments, candidates, stress_checks, bounds = [], [], [], [], []
        for ordinal, (beer, source) in enumerate(sources.items()):
            lower, bound = chemical_limits(source, d, reference, p, source_beer=beer)
            bounds.extend(dict(beer=beer, **item) for item in bound)
            positive = bool(reference.loc[beer, 'X4VG'] > 0)
            before = [panel_five(model['panel'], source[None], indices) for model in development]
            def check(x):
                t = evaluate(x, source, b, indices, classifiers, g, z, center, radius, protected, lower, reducible, development, before, p, 144)
                t['guard_surplus'] = surplus(t)
                return t
            qualified = []
            if positive:
                rows, labels = [], []
                for budget in [96, 144]:
                    for progress in [0., .5, 1.]:
                        x = prior.loc[(beer, budget), d['features']].to_numpy(float).copy()
                        x[protected[2]] = source[protected[2]]-progress*(source[protected[2]]-lower[-1])
                        rows.append(x); labels.append((budget, progress))
                rows = np.stack(rows)
                table = check(rows)
                for j in range(len(rows)):
                    candidates.append(dict(beer=beer, seed_budget=labels[j][0], reduction_progress=labels[j][1], **table.iloc[j].to_dict()))
                    if table.qualified.iloc[j]:
                        x = parameter_draws(source, rows[j], 8, 20261227+ordinal*100+j)
                        stress = check(x)
                        hits = int(stress.qualified.sum())
                        stress_checks.extend(dict(beer=beer, candidate=j, draw=k, **row.to_dict()) for k, row in stress.iterrows())
                        if hits >= 7: qualified.append((rows[j], table.iloc[j], hits, labels[j]))
            if qualified:
                x, state, hits, label = sorted(qualified, key=lambda item: (-item[1].guard_surplus, item[1].objective, item[1].changed_fields))[0]
                passed = True
            else:
                budget = int(final.loc[beer, 'budget'])
                x = prior.loc[(beer, budget), d['features']].to_numpy(float)
                state, hits, label, passed = check(x[None]).iloc[0], 0, (budget, 0.), False
            selected.append(dict(beer=beer, **dict(zip(d['features'], x))))
            assessments.append(dict(beer=beer, source_4vg_positive=positive, seed_budget=label[0], reduction_progress=label[1], design_stress_passes=hits, design_selected=passed, **state.to_dict()))
            print(f'Corrected source{beer}: positive_reference={positive}, design+stress={passed}', flush=True)
        for name, rows in [('selected_profiles.csv', selected), ('design_assessment.csv', assessments), ('candidate_checks.csv', candidates), ('design_stress_checks.csv', stress_checks), ('chemical_bounds.csv', bounds)]:
            if name == 'design_stress_checks.csv' and not rows:
                rows = pd.DataFrame(columns=['beer', 'candidate', 'draw', 'qualified'])
            save_csv(out, name, rows)
        write_json(out/'sealed_selection.json', dict(protocol_sha256=content_hash(r), selection_sha256={name: sha256(out/name) for name in ['selected_profiles.csv', 'design_assessment.csv']}, newly_sealed_validation_outcomes_read=False))
        print('Corrected selections saved and sealed. Validate with --validate-only after fresh model cache is ready.', flush=True)
        return
    seal = json.loads((out/'sealed_selection.json').read_text(encoding='utf-8'))
    assert content_hash(r) == seal['protocol_sha256']
    for name, digest in seal['selection_sha256'].items(): assert sha256(out/name) == digest
    assert not (out/'validation_summary.csv').exists()
    selected = pd.read_csv(out/'selected_profiles.csv', float_precision='round_trip').set_index('beer')
    assessments = pd.read_csv(out/'design_assessment.csv').set_index('beer')
    cache = ROOT/'outputs/q3_pruning_20261027/models'
    fresh = [joblib.load(cache/f'validation_{i}.joblib') for i in range(12)]
    rows, defects, results, parameter_checks = [], [], [], []
    for ordinal, (beer, source) in enumerate(sources.items()):
        x = selected.loc[beer, d['features']].to_numpy(float)
        baseline = [panel_five(model['panel'], source[None], indices) for model in fresh]
        arrays = ensemble_arrays(fresh, x[None], baseline, indices)
        metrics = refit_metrics(*arrays, p['validation_adoption'])
        lower, _ = chemical_limits(source, d, reference, p, source_beer=beer)
        before = [panel_five(model['panel'], source[None], indices) for model in development]
        jitter = parameter_draws(source, x, 32, 20261327+ordinal)
        t = evaluate(jitter, source, b, indices, classifiers, g, z, center, radius, protected, lower, reducible, development, before, p, 144)
        fraction = float(t.qualified.mean())
        parameter_checks.extend(dict(beer=beer, draw=k, **row.to_dict()) for k, row in t.iterrows())
        for repeat, model in enumerate(fresh):
            joint = bool(arrays[3][repeat, 0] and (arrays[0][repeat, 0] <= 0).all() and (arrays[1][repeat, 0, :4] <= 1e-10).all())
            rows.append(dict(beer=beer, repeat=repeat, source_oob=beer not in model['sampled_beers'], joint_pass=joint))
            for j, target in enumerate(old_p['defects']): defects.append(dict(beer=beer, repeat=repeat, target=target, grade_delta=int(arrays[0][repeat, 0, j]), nonzero_delta=float(arrays[1][repeat, 0, j])))
        result = dict(beer=beer, design_selected=bool(assessments.loc[beer, 'design_selected']), validation_joint_fraction=metrics['joint_fraction'][0],
            validation_qualified=bool(metrics['qualified'][0]), parameter_qualified_fraction=fraction,
            stable_four_proxy_conditional=bool(assessments.loc[beer, 'design_selected'] and metrics['qualified'][0] and fraction >= .8),
            all_five_true_nonworsening_certified=False, stale_evidence_sufficient=False, actual_after_measured=False)
        for j, target in enumerate(old_p['defects'][:4]): result[target+'_validation_q80'] = metrics['nonzero_upper'][0, j]
        results.append(result)
    for name, data in [('validation_summary.csv', results), ('validation_checks.csv', rows), ('validation_defects.csv', defects), ('parameter_checks.csv', parameter_checks)]: save_csv(out, name, data)
    for name, digest in seal['selection_sha256'].items(): assert sha256(out/name) == digest
    write_json(out/'summary.json', dict(status='completed_source_scale_correction_and_sealed_checks', stable_four_proxy_conditional_plans=sum(row['stable_four_proxy_conditional'] for row in results),
        all_five_true_nonworsening_certified=0, q1_model_changed=False, phase3_started=False, development_models=20, fresh_validation_models=12,
        model_cache='outputs/q3_pruning_20261027/models', selection_sha256_unchanged=seal['selection_sha256']))
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_positive4vg_20261027')
    parser.add_argument('--validate-only', action='store_true')
    with threadpool_limits(limits=1): run(parser.parse_args())
