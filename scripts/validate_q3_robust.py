"""Open sealed refit checks on fixed Q3 plans, with matched old-plan controls."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, sha256, write_json
from beer_models.robust_transfer import sparse_decode, panel_five, refit_metrics
from beer_models.local_transfer import defect_predictions
from beer_models.tail_risk import survival
from q3_stage2 import setup, source_records
from q3_robust_transfer import fit_refits, ensemble_arrays, evaluate, PROTOCOL, OLD
from r2_raw_q1 import RAW_MANIFEST, content_hash
from beer_models.raw_panel import portrait


def run(args):
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    out = args.output
    read = lambda name: pd.read_csv(out/name, float_precision='round_trip')
    seal = json.loads((out/'sealed_selection.json').read_text(encoding='utf-8'))
    assert seal['protocol_sha256'] == content_hash(p)
    for name, digest in seal['selection_sha256'].items(): assert sha256(out/name) == digest
    assert not (out/'validation_summary.csv').exists(), 'Validation already opened; do not rerun to select again'
    _, d, b, g, classifiers, z, radius, center, target_indices, protected, dictionary, domains = setup(args.source, args.raw, args.q1)
    old_protocol = json.loads((ROOT/'configs/q3_stage2_protocol_20261026.json').read_text(encoding='utf-8'))
    sources, _, _, _ = source_records(d, b, old_protocol)
    indices = [d['targets'].index(t) for t in old_protocol['defects']]
    new = read('selected_profiles.csv'); new['method'] = 'robust_'+new.budget.astype(str)
    old = pd.read_csv(OLD/'selected_profiles.csv', float_precision='round_trip')
    old = old.loc[old.domain.eq('expanded_sensitivity')].copy(); old['method'] = 'previous_broad'
    plans = pd.concat([new[['beer', 'method', *d['features']]], old[['beer', 'method', *d['features']]]], ignore_index=True)
    plans['plan_id'] = plans.method+'_'+plans.beer.astype(str)
    save_csv(out, 'fixed_validation_profiles.csv', plans)
    models = fit_refits(d, p, 'validation', out)
    rows, defects, summaries = [], [], []
    validation_requirements = p['validation_adoption']
    for beer, group in plans.groupby('beer', sort=False):
        baseline = [panel_five(model['panel'], sources[beer][None], indices) for model in models]
        arrays = ensemble_arrays(models, group[d['features']].to_numpy(float), baseline, indices)
        metrics = refit_metrics(*arrays, validation_requirements)
        grade_delta, nonzero_delta, margins, styles = arrays
        for i, row in enumerate(group.itertuples()):
            oob_joint = []
            for repeat, model in enumerate(models):
                joint = bool(styles[repeat, i] and (grade_delta[repeat, i] <= 0).all() and (nonzero_delta[repeat, i, :4] <= 1e-10).all())
                source_oob = beer not in model['sampled_beers']
                if source_oob: oob_joint.append(joint)
                rows.append(dict(beer=int(beer), method=row.method, repeat=repeat, source_oob=source_oob,
                    joint_pass=joint, both_style_pass=bool(styles[repeat, i]),
                    five_grade_pass=bool((grade_delta[repeat, i] <= 0).all()),
                    four_nonzero_pass=bool((nonzero_delta[repeat, i, :4] <= 1e-10).all()),
                    logistic_margin=margins[repeat, i, 0], extra_trees_margin=margins[repeat, i, 1], calibration_applied=False))
                for j, target in enumerate(old_protocol['defects']):
                    defects.append(dict(beer=int(beer), method=row.method, repeat=repeat, target=target,
                        grade_delta=int(grade_delta[repeat, i, j]), nonzero_delta=nonzero_delta[repeat, i, j], source_oob=source_oob))
            item = dict(beer=int(beer), method=row.method, validation_joint_fraction=metrics['joint_fraction'][i],
                validation_qualified=bool(metrics['qualified'][i]), validation_grade_fraction=metrics['grade_fraction'][i],
                validation_style_fraction=metrics['style_fraction'][i], validation_supported_fraction=metrics['supported_fraction'][i],
                validation_five_nonzero_fraction=metrics['five_nonzero_fraction'][i],
                source_oob_refits=len(oob_joint), source_oob_joint_fraction=float(np.mean(oob_joint)) if oob_joint else np.nan)
            for j, target in enumerate(old_protocol['defects'][:4]): item[target+'_validation_delta_q80'] = metrics['nonzero_upper'][i, j]
            summaries.append(item)
    save_csv(out, 'validation_checks.csv', rows); save_csv(out, 'validation_defects.csv', defects)
    validation = pd.DataFrame(summaries)
    save_csv(out, 'validation_summary.csv', validation)
    # All selections were sealed; never switch budget based on these results.
    import joblib
    design_models = [joblib.load(out/'models'/f'design_{i}.joblib') for i in range(p['design_ensemble']['refits'])]
    parameters, fields, neighbors, bounds = [read(name) for name in ['selected_parameters.csv', 'mutable_fields.csv', 'neighbors.csv', 'chemical_bounds.csv']]
    reducible = np.array([d['features'].index(f) for f in p['reducible_fields']])
    groups = np.array([p['groups'].index(v) if v in p['groups'] else 5 for v in dictionary.origin_paper])
    lookup = {int(beer): i for i, beer in enumerate(d['ids'].beer)}
    jitters, jitter_summary = [], []
    for ordinal, row in enumerate(parameters.itertuples()):
        source = sources[row.beer]
        mask = [d['features'].index(f) for f in fields.loc[fields.beer.eq(row.beer)&fields.budget.eq(row.budget), 'field']]
        ids = neighbors.loc[neighbors.beer.eq(row.beer)&neighbors.budget.eq(row.budget)].sort_values('slot').target_beer
        nearby = g.x_[[lookup[int(i)] for i in ids]]
        lower = bounds.loc[bounds.beer.eq(row.beer)].set_index('field').loc[p['reducible_fields'], 'lower'].to_numpy(float)
        u = np.array([getattr(row, f'u{j}') for j in range(14)])
        rng = np.random.default_rng(p['seed']+90000+ordinal)
        perturbed = np.clip(u+rng.normal(0, p['validation_adoption']['parameter_sd'], (p['validation_adoption']['parameter_draws'], len(u))), 0, 1)
        x = sparse_decode(perturbed, source, nearby, mask, groups, reducible, lower)
        before = [panel_five(model['panel'], source[None], indices) for model in design_models]
        t = evaluate(x, source, b, indices, classifiers, g, z, center, radius, protected, lower, reducible, design_models, before, p, row.budget)
        t['beer'], t['budget'], t['draw'] = row.beer, row.budget, np.arange(len(t))
        jitters.append(t)
        jitter_summary.append(dict(beer=row.beer, budget=row.budget, parameter_qualified_fraction=float(t.qualified.mean()), parameter_nominal_fraction=float(t.nominal_pass.mean())))
    save_csv(out, 'parameter_checks.csv', pd.concat(jitters, ignore_index=True))
    jitter_summary = pd.DataFrame(jitter_summary)
    save_csv(out, 'parameter_summary.csv', jitter_summary)
    choices = read('design_choices.csv'); choices['method'] = 'robust_'+choices.budget.astype(str)
    assessment = choices.merge(validation, on=['beer', 'method'], validate='one_to_one').merge(jitter_summary, on=['beer', 'budget'], validate='one_to_one')
    final = read('final_choices_before_validation.csv')
    assessment['selected_budget_before_validation'] = [int(final.set_index('beer').loc[beer, 'budget']) == budget for beer, budget in zip(assessment.beer, assessment.budget)]
    assessment['stable_four_proxy_conditional_plan'] = assessment.qualified&assessment.validation_qualified&(assessment.parameter_qualified_fraction >= p['validation_adoption']['minimum_parameter_pass_fraction'])&assessment.selected_budget_before_validation
    assessment['actual_after_measured'] = False
    assessment['all_five_true_nonworsening_certified'] = False
    assessment['stale_evidence_sufficient'] = False
    save_csv(out, 'plan_assessment.csv', assessment)
    final_assessment = assessment.loc[assessment.selected_budget_before_validation]
    previous = validation.loc[validation.method.eq('previous_broad')].set_index('beer')
    compare = final_assessment[['beer', 'budget', 'changed_fields', 'distance_improvement', 'design_joint_fraction', 'validation_joint_fraction', 'parameter_qualified_fraction', 'stable_four_proxy_conditional_plan']].copy()
    compare['previous_validation_joint_fraction'] = compare.beer.map(previous.validation_joint_fraction)
    old_results = pd.read_csv(OLD/'transfer_results.csv').loc[lambda t: t.domain.eq('expanded_sensitivity')].set_index('beer')
    old_source = pd.read_csv(OLD/'source_profiles.csv').set_index('beer')
    old_profile = old.set_index('beer')
    compare['previous_changed_fields'] = [int((abs(old_profile.loc[beer, d['features']].to_numpy(float)-old_source.loc[beer, d['features']].to_numpy(float)) > 1e-10).sum()) for beer in compare.beer]
    compare['previous_distance_improvement'] = compare.beer.map(old_results.distance_improvement)
    save_csv(out, 'matched_comparison.csv', compare)
    # Current official Q1 portraits, including rejected plans, with both missing acetaldehydes blank.
    before_frame = pd.DataFrame([dict(beer=f'source_{beer}', **dict(zip(d['features'], x))) for beer, x in sources.items()])
    inference = plans[['plan_id', *d['features']]].rename(columns={'plan_id': 'beer'})
    scores, _, _, _ = portrait(pd.concat([before_frame, inference], ignore_index=True), b)
    scores.to_csv(out/'sensory52.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    chemical_rows, defect_rows, change_rows = [], [], []
    before = defect_predictions(b, np.stack([sources[int(i)] for i in plans.beer]), old_protocol['defects'])
    after = defect_predictions(b, plans[d['features']].to_numpy(float), old_protocol['defects'])
    for i, row in enumerate(plans.itertuples()):
        source, values = sources[row.beer], plans.iloc[i][d['features']].to_numpy(float)
        zs, za = g.robust_.transform(source[None])[0], g.robust_.transform(values[None])[0]
        for j, field in enumerate(d['features']):
            change_rows.append(dict(beer=row.beer, method=row.method, field=field, before=source[j], after=values[j],
                coordinate_delta=values[j]-source[j], robust_delta=za[j]-zs[j], unit=dictionary.loc[field, 'unit'],
                transform=dictionary.loc[field, 'candidate_transform'], physical_control_validated=False))
        for j, target in enumerate(old_protocol['defects']):
            item = dict(beer=row.beer, method=row.method, target=target, grade_before=before['grade'][i, j], grade_after=after['grade'][i, j])
            for version in ['uncalibrated', 'calibrated']:
                for cut, label in [(0, 'ge1'), (4, 'ge5'), (5, 'six')]:
                    a, c = survival(before[version])[i, j, cut], survival(after[version])[i, j, cut]
                    item[version+'_'+label+'_before'], item[version+'_'+label+'_after'], item[version+'_'+label+'_delta'] = a, c, c-a
            defect_rows.append(item)
            if j < 4:
                field = old_protocol['protected_coordinates'][j]; k = d['features'].index(field)
                chemical_rows.append(dict(beer=row.beer, method=row.method, field=field, coordinate_before=source[k], coordinate_after=values[k],
                    concentration_before=10**source[k] if j == 0 else source[k] if j in [1, 3] else np.nan,
                    concentration_after=10**values[k] if j == 0 else values[k] if j in [1, 3] else np.nan,
                    unit=dictionary.loc[field, 'unit'], conditional_zero_source=bool(row.beer in [119, 28] and j == 1),
                    coordinate_mapping_established=j != 2, matrix_threshold_applicability_established=False))
    save_csv(out, 'chemical_changes.csv', change_rows); save_csv(out, 'defect_before_after.csv', defect_rows)
    save_csv(out, 'protected_chemical_checks.csv', chemical_rows)
    for name, digest in seal['selection_sha256'].items(): assert sha256(out/name) == digest
    write_json(out/'validation_audit.json', dict(selection_sha256_unchanged=seal['selection_sha256'],
        validation_used_for_selection=False, budgets_switched_after_validation=False, source_beers=250,
        external_independent_data=False, design_refits=8, validation_refits=12))
    write_json(out/'summary.json', dict(status='completed_frozen_robust_design_and_validation',
        final_source_plans=len(final_assessment), stable_four_proxy_conditional_plans=int(final_assessment.stable_four_proxy_conditional_plan.sum()),
        nominal_final_plans=int(final_assessment.nominal_pass.sum()), design_qualified_final_plans=int(final_assessment.qualified.sum()),
        all_five_true_nonworsening_certified=0, stale_evidence_sufficient=False, q1_model_changed=False, phase3_started=False,
        validation_joint_min=float(final_assessment.validation_joint_fraction.min()), validation_joint_max=float(final_assessment.validation_joint_fraction.max()),
        selected_sha256=sha256(out/'final_choices_before_validation.csv')))
    print(compare.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_robust_20261027')
    with threadpool_limits(limits=1): run(parser.parse_args())
