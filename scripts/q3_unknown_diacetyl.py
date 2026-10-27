"""Replace source119 zero assumption by an explicit unknown-preserving rule."""
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
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.uncertain_coordinate import (preserve_unknown, reachable_forest_points, logistic_margin_lower_bound,
    logistic_target_crossings, distance_improvement_lower_bound, maximum_nearest_distance)
from beer_models.robust_transfer import panel_five, refit_metrics, style_margins
from q3_robust_transfer import ensemble_arrays
from q3_risk_aware_pruning import parameter_draws, FIRST
from q3_stage2 import setup, source_records
from r2_raw_q1 import RAW_MANIFEST, content_hash

PROTOCOL = ROOT/'configs/q3_stale_missing_protocol_20261027.json'
CURRENT = ROOT/'outputs/q3_pruning_20261027'


def load_context(args):
    _, d, b, g, classifiers, z, radius, center, _, protected, dictionary, _ = setup(args.source, args.raw, args.q1)
    op = json.loads((ROOT/'configs/q3_stage2_protocol_20261026.json').read_text(encoding='utf-8'))
    sources, _, _, _ = source_records(d, b, op)
    source = sources[119]
    proposed = pd.read_csv(CURRENT/'selected_profiles.csv', float_precision='round_trip').set_index('beer').loc[119, d['features']].to_numpy(float)
    index = d['features'].index('X2.3.butanedione')
    reference = pd.read_excel(args.source/'8.xlsx', sheet_name='Calibrated chemical dataset').set_index('beer')
    reference.index = reference.index.astype(int)
    values = d['x'][:, index]; refs = reference.loc[d['ids'].beer, 'X2.3.butanedione'].to_numpy(float)
    compatible = np.isfinite(values)&np.isfinite(refs)&(refs >= 0)&np.isclose(values, refs, atol=1e-6, rtol=0)
    domain = (0., float(values[compatible].max()))
    dev = [joblib.load(FIRST/'models'/f'{part}_{i}.joblib') for part, count in [('design', 8), ('validation', 12)] for i in range(count)]
    validation = [joblib.load(CURRENT/'models'/f'validation_{i}.joblib') for i in range(12)]
    indices = [d['targets'].index(t) for t in op['defects']]
    requirements = json.loads((CURRENT/'base_protocol.json').read_text(encoding='utf-8'))
    return d, b, g, classifiers, z, radius, center, source, proposed, index, domain, dev, validation, indices, requirements


def events(source, proposed, index, domain, models, nominal_panel, classifiers):
    lower, upper = domain
    parts = [np.array([lower, upper])]
    for panel, styles in [(nominal_panel, classifiers)]+[(m['panel'], m['classifiers']) for m in models]:
        for fixed in [source, proposed]:
            parts.append(reachable_forest_points(panel.model, fixed, index, lower, upper))
        parts.append(reachable_forest_points(styles['extra_trees'][-1], proposed, index, lower, upper))
        parts.append(logistic_target_crossings(styles['logistic'], proposed, index, lower, upper))
    knots = np.unique(np.concatenate(parts))
    knots = knots[(knots >= lower)&(knots <= max(upper, float(np.float32(upper))))]
    # Every open interval between routing/class events also gets a representative.
    return np.unique(np.r_[knots, (knots[1:]+knots[:-1])/2])


def inspect_profile(source, proposed, index, domain, models, nominal_panel, classifiers, indices, requirements, g, z, center, radius):
    points = events(source, proposed, index, domain, models, nominal_panel, classifiers)
    a = np.broadcast_to(source, (len(points), len(source))).copy(); a[:, index] = points
    c = np.broadcast_to(proposed, a.shape).copy(); c[:, index] = points
    before = [panel_five(m['panel'], a, indices) for m in models]
    arrays = ensemble_arrays(models, c, before, indices)
    metrics = refit_metrics(*arrays, requirements)
    nominal_a, nominal_c = panel_five(nominal_panel, a, indices), panel_five(nominal_panel, c, indices)
    nominal_delta = nominal_c['nonzero']-nominal_a['nonzero']
    nominal_ok = ((nominal_c['grade'] <= nominal_a['grade']).all(1)&(nominal_delta[:, :4] <= 1e-10).all(1))
    nominal_margin, _ = style_margins(classifiers, c)
    nominal_ok &= (nominal_margin[:, 1] >= .01)
    logit_bound = logistic_margin_lower_bound(classifiers['logistic'], proposed, index, *domain)
    model_bounds = [logistic_margin_lower_bound(m['classifiers']['logistic'], proposed, index, *domain) for m in models]
    design_logit_bound = float(np.quantile(model_bounds, requirements.get('style_margin_quantile', .2)))
    distance = distance_improvement_lower_bound(g.robust_, source, proposed, center, index, *domain)
    support = maximum_nearest_distance(g.robust_, proposed, z, index, *domain)
    geometric = logit_bound >= .01 and distance >= .1 and support <= radius+1e-10
    # Logistic q20 bound is needed only for the DEVELOPMENT margin constraint.
    margin_certified = 'minimum_style_margin_quantile' not in requirements or design_logit_bound >= requirements['minimum_style_margin_quantile']
    qualified = bool(nominal_ok.all() and metrics['qualified'].all() and geometric and margin_certified)
    table = pd.DataFrame(dict(diacetyl=points, nominal_guard=nominal_ok, refit_qualified=metrics['qualified'],
        joint_fraction=metrics['joint_fraction'], nominal_et_margin=nominal_margin[:, 1]))
    for j in range(4):
        table[f'nonzero_delta_q80_{j}'] = metrics['nonzero_upper'][:, j]
        table[f'nominal_delta_{j}'] = nominal_delta[:, j]
    summary = dict(qualified_on_empirical_support=qualified, evaluated_representatives=len(points), min_joint_fraction=float(metrics['joint_fraction'].min()),
        max_nonzero_delta_q80=metrics['nonzero_upper'].max(0).tolist(), nominal_all_tree_guards=bool(nominal_ok.all()),
        nominal_logistic_margin_lower_bound=logit_bound, logistic_q20_margin_lower_bound=design_logit_bound,
        distance_improvement_lower_bound=distance, maximum_nearest_support_distance=support,
        nominal_and_geometric_certified=bool(nominal_ok.all() and geometric), logistic_design_margin_certified=bool(margin_certified))
    return summary, table


def run(args):
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    out = args.output
    if args.recompute:
        previous = json.loads((out/'summary.json').read_text(encoding='utf-8'))
        assert previous['protocol_sha256'] == content_hash(p)
        assert previous['q1_model_sha256'] == sha256(args.q1/'model.joblib')
    else:
        out.mkdir(parents=True, exist_ok=False)
    d, b, g, classifiers, z, radius, center, source, proposed, index, domain, dev, validation, indices, requirements = load_context(args)
    nominal = b['risk_panel_model'].models['extra_trees']
    write_json(out/'protocol.json', p)
    save_csv(out, 'source_evidence.csv', [dict(beer=119, original_coordinate_missing=True, calibrated_reference=0., raw_value_recovered=False,
        empirical_model_support_lower=domain[0], empirical_model_support_upper=domain[1], support_is_real_source_bound=False,
        support_unit='supplied_coordinate_reference_unit_not_automatically_source_unit')])
    summaries = {}
    for name, models, req in [('development', dev, requirements['design_requirements']), ('validation', validation, requirements['validation_adoption'])]:
        summary, table = inspect_profile(source, proposed, index, domain, models, nominal, classifiers, indices, req, g, z, center, radius)
        save_csv(out, name+'_interval_checks.csv', table); summaries[name] = summary
        print(f'{name}: {json.dumps(summary)}', flush=True)
    # Ordinary missing-input prediction uses EACH model's training imputer, not0.
    missing_source, missing_after = source.copy(), proposed.copy()
    missing_source[index], missing_after[index] = np.nan, np.nan
    na_rows = []
    for name, models, req in [('development', dev, requirements['design_requirements']), ('validation', validation, requirements['validation_adoption'])]:
        before = [panel_five(m['panel'], missing_source[None], indices) for m in models]
        metrics = refit_metrics(*ensemble_arrays(models, missing_after[None], before, indices), req)
        na_rows.append(dict(role=name, ordinary_missing_joint_fraction=float(metrics['joint_fraction'][0]), ordinary_missing_refit_qualified=bool(metrics['qualified'][0])))
    save_csv(out, 'ordinary_missing_checks.csv', na_rows)
    stress_rows = []
    if summaries['development']['qualified_on_empirical_support'] and summaries['validation']['qualified_on_empirical_support']:
        jitter = parameter_draws(source, proposed, 32, 20261205)
        for k, profile in enumerate(jitter):
            summary, _ = inspect_profile(source, profile, index, domain, dev, nominal, classifiers, indices, requirements['design_requirements'], g, z, center, radius)
            stress_rows.append(dict(draw=k, **summary))
            print(f'Unknown-coordinate stress {k+1}/32: {summary["qualified_on_empirical_support"]}', flush=True)
    save_csv(out, 'joint_parameter_unknown_checks.csv', pd.DataFrame(stress_rows) if stress_rows else pd.DataFrame(columns=['draw','qualified_on_empirical_support']))
    save_csv(out, 'source119_profile.csv', [dict(beer=119, **dict(zip(d['features'], missing_source)))])
    save_csv(out, 'transfer119_profile.csv', [dict(beer='transfer119', **dict(zip(d['features'], missing_after)))])
    stress_fraction = float(np.mean([r['qualified_on_empirical_support'] for r in stress_rows])) if stress_rows else None
    zero_independent = bool(summaries['development']['qualified_on_empirical_support'] and summaries['validation']['qualified_on_empirical_support'] and stress_fraction is not None and stress_fraction >= .8)
    write_json(out/'transfer_rule.json', dict(beer=119, field='X2.3.butanedione', operation='preserve_unknown_source_value',
        source_value=None, after_value=None, chemical_delta_identically_zero_for_any_source_value=True,
        measured_value_recovered=False, model_qualification_no_longer_requires_exact_zero=zero_independent,
        finite_model_support_domain=list(domain), true_source_bound_established=False, physical_or_Stale_safety_certified=False))
    write_json(out/'summary.json', dict(status='completed', protocol_sha256=content_hash(p), zero_assumption_removed_for_chemical_rule=True,
        zero_assumption_removed_for_model_qualification=zero_independent, interval_results=summaries, joint_unknown_parameter_fraction=stress_fraction,
        q1_model_sha256=sha256(args.q1/'model.joblib'), raw_sha256=sha256(args.raw),
        source_sha256={name: sha256(args.source/name) for name in ['1.csv','3.xlsx','4.csv','8.xlsx']},
        target_profile_sha256=sha256(CURRENT/'selected_profiles.csv'), true_unknown_concentration_recovered=False,
        true_all_five_safety_certified=False, phase3_started=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_unknown_diacetyl_20261027')
    parser.add_argument('--recompute', action='store_true', help='Recompute this same analysis after numerical fixes; verify saved protocol/model first')
    with threadpool_limits(limits=1): run(parser.parse_args())
