"""Stage2 local block transfer; chemical, forecast and real-effect evidence stay distinct."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.generation import StyleGenerator
from beer_models.style_models import make_style_model
from beer_models.defect_constraints import defect_aware_box, project_defects
from beer_models.local_transfer import decode, structural_metrics, defect_predictions, guard_metrics, rank_candidates
from r2_raw_q1 import load_inputs, RAW_MANIFEST, content_hash, PROTOCOL as BASE
from task4_raw_q3 import portrait

PROTOCOL = ROOT/'configs/q3_stage2_protocol_20261026.json'
DICTIONARY = ROOT/'outputs/p0_audit_20261023_final/feature_dictionary.csv'
BOUND_EVIDENCE = ROOT/'outputs/q3_resolution_20261024/source_defect_evidence.csv'
HISTORICAL = ROOT/'outputs/q3_full_grid_20261026_v2/full_grid_checks.csv.gz'


def setup(source, raw, q1):
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    d = load_inputs(source, raw, json.loads(BASE.read_text(encoding='utf-8')))
    b = joblib.load(q1/'model.joblib')
    assert b['selected_model'] == 'extra_trees__median' and b['raw_panel_model'].penalties is None
    assert b['features'] == d['features'] and b['targets'] == d['targets']
    generator = StyleGenerator().fit(d['x'], d['ids'].tasting_category_fine.to_numpy())
    sc = json.loads((ROOT/'configs/q3_style_protocol.json').read_text(encoding='utf-8'))
    classifiers = {name: make_style_model(name, sc, sc['seed']).fit(d['x'], d['ids'].tasting_category_fine)
                   for name in ['logistic', 'extra_trees']}
    z = generator.robust_.transform(generator.x_)
    distances = cdist(z, z)/np.sqrt(z.shape[1]); np.fill_diagonal(distances, np.inf)
    radius = float(np.quantile(distances.min(1), p['support_quantile']))
    target_indices = generator.indices_[p['target_style']]
    center = z[target_indices].mean(0)
    protected = [d['features'].index(f) for f in p['protected_coordinates']]
    dictionary = pd.read_csv(DICTIONARY).set_index('field_name_R').loc[d['features']]
    main = np.array([i for i, f in enumerate(d['features']) if dictionary.loc[f, 'origin_paper'] == 'Yeast' and i not in protected])
    expanded = np.array([i for i in range(len(d['features'])) if i not in protected])
    return p, d, b, generator, classifiers, z, radius, center, target_indices, protected, dictionary, dict(yeast_block=main, expanded_sensitivity=expanded)


def evaluate(x, u, source, before, p, d, b, generator, classifiers, z, radius, center, protected, target_bounds):
    values = structural_metrics(x, source, generator.robust_, z, center, radius, classifiers, p['target_style'])
    predictions = defect_predictions(b, x, p['defects'])
    guards, deltas = guard_metrics(before, predictions, p['numeric_tolerance'])
    values.update(guards)
    values['chemical_pass'] = np.isclose(x[:, protected], source[protected], atol=p['numeric_tolerance'], rtol=0).all(1)
    values['distance_pass'] = values['distance_improvement'] >= p['minimum_distance_improvement']
    structural_fields = ['chemical_pass', 'distance_pass', 'support_pass', 'logistic_style_pass', 'extra_trees_style_pass']
    values['structural_violation_count'] = sum(~values[k] for k in structural_fields)
    values['structural_pass'] = values['structural_violation_count'] == 0
    values['main_computational_pass'] = values['structural_pass']&values['grade_pass']&values['supported_nonzero_pass']
    values['with_stale_nonzero_pass'] = values['main_computational_pass']&values['five_nonzero_pass']
    values['with_both_nonzero_pass'] = values['main_computational_pass']&values['both_nonzero_pass']
    values['with_thirty_guard_pass'] = values['main_computational_pass']&values['thirty_guard_pass']
    values['old_target_box_pass'] = defect_aware_box(x, *target_bounds, protected)
    values['objective'] = p['objective']['target_distance_weight']*values['target_distance']+p['objective']['change_distance_weight']*values['change_distance']
    for j, target in enumerate(p['defects']):
        values[target+'_grade_delta'] = deltas['grade'][:, j]
        for version in ['calibrated', 'uncalibrated']:
            for k, event in enumerate(['ge1', 'ge5', '6']):
                values[target+'_'+version+'_delta_'+event] = deltas[version][:, j, k]
    for k in range(u.shape[1]):
        values[f'u{k}'] = u[:, k]
    return pd.DataFrame(values)


def history_control(p, d, b, output):
    grid = pd.read_csv(HISTORICAL, float_precision='round_trip')
    anchors = pd.read_csv(ROOT/'outputs/q3_resolution_20261024/anchors.csv').set_index('anchor')[d['features']].to_numpy(float)
    old_sources = pd.read_csv(ROOT/'outputs/q3_task3_20261025/transfer_source_profiles.csv').set_index('beer')
    protected = [d['features'].index(f) for f in p['protected_coordinates']]
    comparisons, ladder = [], []
    for beer, group in grid.groupby('beer', sort=False):
        source = old_sources.loc[beer, d['features']].to_numpy(float)
        before = defect_predictions(b, source[None], p['defects'])
        sample = group.iloc[[0, len(group)//2, len(group)-1]]
        points = np.stack([project_defects((source+row.fraction*(anchors[int(row.anchor)]-source))[None], source, protected, row.policy)[0]
                           for row in sample.itertuples()])
        _, delta = guard_metrics(before, defect_predictions(b, points, p['defects']))
        maximum = 0.
        for j, target in enumerate(p['defects']):
            actual = delta['grade'][:, j]
            np.testing.assert_array_equal(actual, sample[target+'_grade_delta'])
            for version in ['calibrated', 'uncalibrated']:
                for k, event in enumerate(['ge1', 'ge5', '6']):
                    diff = abs(delta[version][:, j, k]-sample[target+'_'+version+'_delta_'+event].to_numpy())
                    maximum = max(maximum, float(diff.max()))
                    np.testing.assert_allclose(delta[version][:, j, k], sample[target+'_'+version+'_delta_'+event], atol=1e-10, rtol=0)
        comparisons.append(dict(beer=int(beer), independently_replayed_points=3, maximum_probability_delta_difference=maximum))
        structural = group[['range_pass', 'chemical_pass', 'style_pass', 'distance_improves']].all(axis=1)
        grade = group.all_raw_grades_pass
        supported = (group[[t+'_uncalibrated_delta_ge1' for t in p['supported_probability_targets']]] <= 1e-10).all(axis=1)
        all5 = (group[[t+'_uncalibrated_delta_ge1' for t in p['defects']]] <= 1e-10).all(axis=1)
        for name, mask in [('old_structure', structural), ('old_structure_grade', structural&grade),
            ('old_structure_grade_four_nonzero', structural&grade&supported), ('old_structure_grade_five_nonzero', structural&grade&all5),
            ('old_complete_thirty', group.all_current_constraints_pass)]:
            ladder.append(dict(beer=int(beer), check=name, count=int(mask.sum()), scope='stored_old_structure_and_new_guard_subset_not_new_domain_success'))
    save_csv(output, 'historical_33_point_control.csv', comparisons)
    save_csv(output, 'historical_guard_ladder.csv', ladder)
    counts = pd.read_csv(ROOT/'outputs/q3_conflict_audit_20261026/individual_guard_counts.csv')
    save_csv(output, 'historical_diacetyl_4vg_conditions.csv', counts.loc[counts.guard.str.startswith(('diacetyl_', 'X4vg_'))])
    return dict(old_grid_rows=len(grid), old_grid_prediction_replays=33, stored_old_complete_passes=int(grid.all_current_constraints_pass.sum()))


def source_records(d, b, p):
    evidence = pd.read_csv(BOUND_EVIDENCE)
    imputer = b['risk_panel_model'].models['extra_trees'].imputer
    sources, metadata, missing = {}, [], []
    for i in np.flatnonzero(d['ids'].tasting_category_fine.eq(p['source_style'])):
        beer = int(d['ids'].beer.iloc[i]); raw = d['x'][i].copy()
        source = imputer.transform(raw[None])[0]
        for j in np.flatnonzero(~np.isfinite(raw)):
            missing.append(dict(beer=beer, field=d['features'][j], imputed_coordinate=source[j], recovered_measurement=False))
        source_evidence = evidence.loc[evidence.beer.eq(beer)]
        conditional = False
        for field in p['protected_coordinates']:
            j = d['features'].index(field); row = source_evidence.loc[source_evidence.field.eq(field)].iloc[0]
            if not np.isfinite(raw[j]):
                assert field == 'X2.3.butanedione' and row.attachment8_calibrated_value == 0
                source[j] = 0.; conditional = True
        sources[beer] = source
        metadata.append(dict(beer=beer, source_index=int(i), conditional_zero_bound=conditional,
            missing_feature_count=int((~np.isfinite(raw)).sum())))
    return sources, pd.DataFrame(metadata), pd.DataFrame(missing), evidence


def run(args):
    p, d, b, generator, classifiers, z, radius, center, target_indices, protected, dictionary, domains = setup(args.source, args.raw, args.q1)
    output = args.output
    output.mkdir(parents=True, exist_ok=False); (output/'models').mkdir()
    write_json(output/'protocol.json', p)
    for name, model in classifiers.items():
        joblib.dump(model, output/'models'/f'style_{name}.joblib', compress=3)
    sources, source_meta, missing, evidence = source_records(d, b, p)
    save_csv(output, 'source_status.csv', source_meta); save_csv(output, 'source_missingness.csv', missing)
    save_csv(output, 'source_defect_evidence.csv', evidence)
    variables = dictionary.reset_index().rename(columns={'index': 'field'})
    variables['mutable_main'] = [j in domains['yeast_block'] for j in range(len(d['features']))]
    variables['mutable_expanded'] = [j in domains['expanded_sensitivity'] for j in range(len(d['features']))]
    variables['coordinate_unit'] = 'supplied_coordinate_not_uniform_concentration'
    variables['physical_independent_control_validated'] = False
    save_csv(output, 'control_variables.csv', variables)
    hist = history_control(p, d, b, output)
    target_bounds = (generator.x_[target_indices].min(0), generator.x_[target_indices].max(0))
    neighbors_log, selections, selected_profiles, all_checks, all_parameters = [], [], [], [], []
    selected_meta = []
    for bi, (beer, source) in enumerate(sources.items()):
        before = defect_predictions(b, source[None], p['defects'])
        for di, domain in enumerate(p['domains']):
            mutable = domains[domain]
            dist = np.square(z[target_indices][:, mutable]-generator.robust_.transform(source[None])[0, mutable]).mean(1)
            nearest = target_indices[np.argsort(dist, kind='stable')[:p['search']['nearest_targets']]]
            neighborhood = generator.x_[nearest]
            neighbors_log.extend(dict(beer=beer, domain=domain, slot=k, target_beer=int(d['ids'].beer.iloc[i])) for k, i in enumerate(nearest))
            seed = p['search']['seed']+bi*100+di
            sampler = qmc.Sobol(len(nearest)+1, scramble=True, seed=seed)
            initial = sampler.random_base2(p['search']['sobol_power'])
            vertices = np.column_stack([np.ones(len(nearest)), np.eye(len(nearest))])
            chunks = [np.vstack([initial, vertices])]
            tables, params = [], []
            rng = np.random.default_rng(seed)
            for iteration in range(1+p['search']['refinement_rounds']):
                u = chunks[-1]
                x, _, _ = decode(u, source, neighborhood, mutable)
                table = evaluate(x, u, source, before, p, d, b, generator, classifiers, z, radius, center, protected, target_bounds)
                table['iteration'] = iteration
                tables.append(table); params.append(u)
                aggregate = pd.concat(tables, ignore_index=True)
                parameter_matrix = np.concatenate(params)
                if iteration < p['search']['refinement_rounds']:
                    elite = rank_candidates(aggregate)[:p['search']['elite_size']]
                    proposal = parameter_matrix[rng.choice(elite, p['search']['refinement_size'])]+rng.normal(0, p['search']['refinement_scales'][iteration], (p['search']['refinement_size'], len(nearest)+1))
                    chunks.append(np.clip(proposal, 0, 1))
            aggregate['candidate'] = np.arange(len(aggregate))
            aggregate['beer'], aggregate['domain'] = beer, domain
            all_checks.append(aggregate)
            passes = aggregate.loc[aggregate.main_computational_pass]
            chosen = int(passes.sort_values(['objective', 'candidate']).iloc[0].candidate) if len(passes) else int(rank_candidates(aggregate)[0])
            chosen_row = aggregate.iloc[chosen]
            chosen_u = parameter_matrix[chosen]
            chosen_x = decode(chosen_u[None], source, neighborhood, mutable)[0][0]
            selected_profiles.append(dict(beer=beer, domain=domain, **dict(zip(d['features'], chosen_x))))
            all_parameters.append(dict(beer=beer, domain=domain, candidate=chosen, **{f'u{k}': value for k, value in enumerate(chosen_u)}))
            status = 'conditional_plan_stale_unestablished' if len(passes) else 'restricted_main_constraints_not_met'
            selections.append(dict(beer=beer, domain=domain, status=status, selected_candidate=chosen,
                main_candidate_count=len(passes), structural_candidates=int(aggregate.structural_pass.sum()),
                five_nonzero_candidates=int(aggregate.with_stale_nonzero_pass.sum()),
                both_nonzero_candidates=int(aggregate.with_both_nonzero_pass.sum()), thirty_guard_candidates=int(aggregate.with_thirty_guard_pass.sum()),
                selected_main_pass=bool(chosen_row.main_computational_pass), selected_thirty_pass=bool(chosen_row.with_thirty_guard_pass),
                objective=chosen_row.objective, distance_improvement=chosen_row.distance_improvement,
                source_distance=chosen_row.source_target_distance, target_distance=chosen_row.target_distance,
                change_distance=chosen_row.change_distance, conditional_zero_bound=bool(source_meta.set_index('beer').loc[beer].conditional_zero_bound),
                stale_nonworsening_established=False, physical_feasibility_certified=False, actual_after_measured=False,
                all_five_true_requirements_certified=False))
            selected_meta.append(dict(beer=beer, domain=domain, logistic_style_pass=bool(chosen_row.logistic_style_pass),
                extra_trees_style_pass=bool(chosen_row.extra_trees_style_pass), support_pass=bool(chosen_row.support_pass),
                supported_nonzero_pass=bool(chosen_row.supported_nonzero_pass), grade_pass=bool(chosen_row.grade_pass),
                old_target_box_pass=bool(chosen_row.old_target_box_pass)))
            print(f'{beer} {domain}: {len(passes)}/{len(aggregate)} main computational passes; conditional only', flush=True)
    tables = pd.concat(all_checks, ignore_index=True)
    tables.to_csv(output/'candidate_checks.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    save_csv(output, 'selected_profiles.csv', selected_profiles); save_csv(output, 'selected_parameters.csv', all_parameters)
    save_csv(output, 'transfer_results.csv', selections); save_csv(output, 'selected_diagnostics.csv', selected_meta)
    save_csv(output, 'neighbors.csv', neighbors_log)
    save_csv(output, 'source_profiles.csv', [dict(beer=beer, **dict(zip(d['features'], source))) for beer, source in sources.items()])
    before_frame = pd.DataFrame([dict(beer=f'source_{beer}', **dict(zip(d['features'], source))) for beer, source in sources.items()])
    after_frame = pd.DataFrame(selected_profiles)
    after_frame['beer'] = after_frame.domain+'_'+after_frame.beer.astype(str)
    long, _, _, _ = portrait(pd.concat([before_frame, after_frame.drop(columns='domain')], ignore_index=True), b)
    long.to_csv(output/'transfer_sensory52.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    generated = pd.read_csv(ROOT/'outputs/task4_raw_q3_20261026/generated_profiles.csv', dtype={'generated_id': str})
    g = generated.rename(columns={'generated_id': 'beer'})
    g['beer'] = g.beer.astype(str)
    sensory, _, _, _ = portrait(g, b)
    sensory.to_csv(output/'generated_sensory52.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    summary = dict(status='completed_computational_search_not_real_brewing_certification',
        protocol_sha256=content_hash(p), q1_model_sha256=sha256(args.q1/'model.joblib'),
        source_sha256={name: sha256(args.source/name) for name in ['1.csv', '3.xlsx', '4.csv', '8.xlsx']},
        raw_source_sha256=sha256(args.raw), n_sources=len(sources), mutable_main_fields=len(domains['yeast_block']),
        main_conditional_plans=sum(r['selected_main_pass'] for r in selections if r['domain'] == p['main_domain']),
        expanded_sensitivity_plans=sum(r['selected_main_pass'] for r in selections if r['domain'] != p['main_domain']),
        fully_evidence_certified_transfers=0, stale_status='insufficient_heldout_probability_support',
        historical_control=hist, new_candidate_count=len(tables), support_radius=radius,
        generations_reused=300, q1_experimental_models_adopted=False, phase3_started=False,
        evidence_sha256={path.relative_to(ROOT).as_posix(): sha256(path) for path in [DICTIONARY, BOUND_EVIDENCE, HISTORICAL, ROOT/'outputs/task4_raw_q3_20261026/generated_profiles.csv']})
    write_json(output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_stage2_20261026')
    with threadpool_limits(limits=1):
        run(parser.parse_args())
