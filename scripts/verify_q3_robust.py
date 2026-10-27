"""Verify Q3 sparse-domain reconstruction, selection seal, and fixed-plan refits."""
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
from beer_audit.audit import write_json, sha256
from beer_models.robust_transfer import sparse_decode, panel_five, refit_metrics, design_order, style_margins
from scipy.spatial.distance import cdist
from beer_models.q3_raw import raw_predictions
from beer_models.local_transfer import defect_predictions
from q3_robust_transfer import source_domain, chemical_limits, ensemble_arrays, evaluate, PROTOCOL
from q3_stage2 import setup, source_records
from r2_raw_q1 import content_hash, RAW_MANIFEST


def run(args):
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    out = args.output
    read = lambda name: pd.read_csv(out/name, float_precision='round_trip')
    provenance = json.loads((out/'provenance.json').read_text(encoding='utf-8'))
    assert provenance['protocol_sha256'] == content_hash(p)
    assert provenance['q1_sha256'] == sha256(args.q1/'model.joblib')
    assert provenance['raw_sha256'] == sha256(args.raw)
    for field, digest in provenance['input_sha256'].items(): assert sha256(args.source/field) == digest
    seal = json.loads((out/'sealed_selection.json').read_text(encoding='utf-8'))
    for name, digest in seal['selection_sha256'].items(): assert sha256(out/name) == digest
    audit = json.loads((out/'validation_audit.json').read_text(encoding='utf-8'))
    assert not audit['validation_used_for_selection'] and not audit['budgets_switched_after_validation']
    _, d, b, g, classifiers, z, radius, center, target_indices, protected, dictionary, domains = setup(args.source, args.raw, args.q1)
    old_p = json.loads((ROOT/'configs/q3_stage2_protocol_20261026.json').read_text(encoding='utf-8'))
    sources, _, _, _ = source_records(d, b, old_p)
    indices = [d['targets'].index(t) for t in old_p['defects']]
    reducible = np.array([d['features'].index(f) for f in p['reducible_fields']])
    reference = pd.read_excel(args.source/'8.xlsx', sheet_name='Calibrated chemical dataset').set_index('beer')
    reference.index = reference.index.astype(int)
    candidates, selected, choices, final, parameters = [read(name) for name in ['design_candidates.csv.gz', 'selected_profiles.csv', 'design_choices.csv', 'final_choices_before_validation.csv', 'selected_parameters.csv']]
    design = [joblib.load(out/'models'/f'design_{i}.joblib') for i in range(p['design_ensemble']['refits'])]
    n_reconstructed, n_design_replayed, worst = 0, 0, 0.
    for (beer, budget), table in candidates.groupby(['beer', 'budget'], sort=False):
        source = sources[beer]
        mutable, nearest, groups, contribution = source_domain(source, budget, d, g, classifiers, dictionary, protected, p)
        lower, _ = chemical_limits(source, d, reference, p)
        u = table[[f'u{k}' for k in range(14)]].to_numpy(float)
        x = sparse_decode(u, source, g.x_[nearest], mutable, groups, reducible, lower)
        fixed = np.setdiff1d(np.arange(len(source)), np.r_[mutable, reducible])
        np.testing.assert_allclose(x[:, fixed], np.broadcast_to(source[fixed], (len(x), len(fixed))), atol=0, rtol=0)
        assert (x[:, reducible] <= source[reducible]+1e-12).all() and (x[:, reducible] >= lower-1e-12).all()
        assert ((abs(x-source) > 1e-10).sum(1) <= budget).all()
        good = table.loc[table.qualified]
        pick = int(good.sort_values(['objective', 'changed_fields', 'candidate']).iloc[0].candidate) if len(good) else int(table.iloc[design_order(table)[0]].candidate)
        stored = choices.loc[choices.beer.eq(beer)&choices.budget.eq(budget)].iloc[0]
        assert pick == stored.selected_candidate and len(good) == stored.design_qualified_count
        profile = selected.loc[selected.beer.eq(beer)&selected.budget.eq(budget), d['features']].to_numpy(float)[0]
        np.testing.assert_allclose(profile, x[pick], atol=1e-12, rtol=0)
        params = parameters.loc[parameters.beer.eq(beer)&parameters.budget.eq(budget), [f'u{k}' for k in range(14)]].to_numpy(float)[0]
        np.testing.assert_array_equal(params, u[pick])
        # Recompute all nominal grade/risk deltas without refitting the official model.
        base = panel_five(b['risk_panel_model'].models['extra_trees'], source[None], indices)
        pred = panel_five(b['risk_panel_model'].models['extra_trees'], x, indices)
        for j in range(4):
            np.testing.assert_allclose(pred['nonzero'][:, j]-base['nonzero'][0, j], table[f'nominal_nonzero_delta_{j}'], atol=1e-10, rtol=0)
        zz, zs = g.robust_.transform(x), g.robust_.transform(source[None])[0]
        initial = float(np.sqrt(np.square(zs-center).mean()))
        target = np.sqrt(np.square(zz-center).mean(1)); change = np.sqrt(np.square(zz-zs).mean(1))
        support = cdist(zz, z).min(1)/np.sqrt(len(source))
        margin, style = style_margins(classifiers, x)
        structural = (1-target/initial < .1).astype(int)+(support > radius+1e-10)+(margin < .01).sum(1)+(~style)
        sensory = (pred['grade']-base['grade'] > 0).sum(1)+(pred['nonzero'][:, :4]-base['nonzero'][:, :4] > 1e-10).sum(1)
        np.testing.assert_array_equal(table.structural_violations, structural)
        np.testing.assert_array_equal(table.nominal_sensory_violations, sensory)
        np.testing.assert_array_equal(table.nominal_pass, (structural == 0)&(sensory == 0))
        np.testing.assert_array_equal(table.changed_fields, (abs(x-source) > 1e-10).sum(1))
        np.testing.assert_allclose(table.objective, .5*target/initial+.35*change+.15*table.changed_fields.to_numpy()/224, atol=1e-10, rtol=0)
        # Rebuild every recorded joint/quantile decision; independent cached ensemble replay at3 points/domain.
        qcols = [f'nonzero_q80_{j}' for j in range(4)]
        mcols = [name+'_margin_q20' for name in classifiers]
        design_ok = (table.design_joint_fraction >= .75)&(table[qcols] <= -.002+1e-10).all(axis=1)&(table[mcols] >= 0).all(axis=1)
        np.testing.assert_array_equal(table.design_pass, design_ok)
        np.testing.assert_array_equal(table.qualified, table.nominal_pass&design_ok)
        fixed_samples = np.unique([0, len(table)//2, pick])
        before = [panel_five(model['panel'], source[None], indices) for model in design]
        fresh = evaluate(x[fixed_samples], source, b, indices, classifiers, g, z, center, radius, protected, lower, reducible, design, before, p, budget)
        for col in fresh:
            a, c = fresh[col].to_numpy(), table.iloc[fixed_samples][col].to_numpy()
            if fresh[col].dtype.kind == 'b': np.testing.assert_array_equal(a, c)
            else:
                worst = max(worst, float(abs(a-c).max()))
                np.testing.assert_allclose(a, c, atol=1e-10, rtol=0, err_msg=col)
        n_reconstructed += len(table); n_design_replayed += len(fixed_samples)
    for beer, group in choices.groupby('beer', sort=False):
        good = group.loc[group.qualified]
        chosen = good.sort_values(['objective', 'changed_fields', 'budget']).iloc[0] if len(good) else group.iloc[design_order(group)[0]]
        assert int(final.set_index('beer').loc[beer, 'budget']) == int(chosen.budget)
    validation = [joblib.load(out/'models'/f'validation_{i}.joblib') for i in range(p['validation_ensemble']['refits'])]
    plans = read('fixed_validation_profiles.csv')
    summaries = read('validation_summary.csv').set_index(['beer', 'method'])
    validation_checks = read('validation_checks.csv')
    for beer, group in plans.groupby('beer', sort=False):
        before = [panel_five(model['panel'], sources[beer][None], indices) for model in validation]
        arrays = ensemble_arrays(validation, group[d['features']].to_numpy(float), before, indices)
        fresh = refit_metrics(*arrays, p['validation_adoption'])
        for i, row in enumerate(group.itertuples()):
            saved = summaries.loc[(beer, row.method)]
            np.testing.assert_allclose(fresh['joint_fraction'][i], saved.validation_joint_fraction, atol=0, rtol=0)
            assert bool(fresh['qualified'][i]) == bool(saved.validation_qualified)
            for j, target in enumerate(old_p['defects'][:4]):
                np.testing.assert_allclose(fresh['nonzero_upper'][i, j], saved[target+'_validation_delta_q80'], atol=1e-10, rtol=0)
    scores = read('sensory52.csv.gz')
    assert len(scores) == 44*52
    assert scores.groupby('beer').size().eq(52).all()
    missing = scores.target.isin(['A_acetaldehyde', 'F_acetaldehyde'])
    assert scores.loc[missing, 'estimated_grade_0_6'].isna().all() and missing.sum() == 88
    assert not scores.actual_after_measured.any() and not scores.safety_certified.any()
    x = selected[d['features']].to_numpy(float)
    official = raw_predictions(b, x, d['parents'])
    subset = defect_predictions(b, x, old_p['defects'])
    for key in subset: np.testing.assert_allclose(subset[key], official[key][:, indices], atol=1e-12, rtol=0)
    assessment = read('plan_assessment.csv')
    expected = assessment.qualified&assessment.validation_qualified&(assessment.parameter_qualified_fraction >= .8)&assessment.selected_budget_before_validation
    np.testing.assert_array_equal(assessment.stable_four_proxy_conditional_plan, expected)
    assert not assessment.all_five_true_nonworsening_certified.any() and not assessment.stale_evidence_sufficient.any()
    assert not assessment.actual_after_measured.any()
    for partition, count in [('design', 8), ('validation', 12)]:
        draws = read(partition+'_draws.csv')
        assert len(draws) == count*250
        for _, draw in draws.groupby('repeat'):
            counts = d['ids'].set_index('beer').loc[draw.beer, 'tasting_category_fine'].value_counts().sort_index()
            np.testing.assert_array_equal(counts, d['ids'].tasting_category_fine.value_counts().sort_index())
    write_json(out/'verification.json', dict(status='passed', all_new_spectra_reconstructed=n_reconstructed,
        nominal_four_nonzero_repredicted_points=n_reconstructed, design_ensemble_cached_replay_points=n_design_replayed,
        design_ensemble_refits=8, validation_ensemble_cached_replay_points=len(plans), validation_ensemble_refits=12,
        full_official_Q1_selected_replays=len(selected), maximum_sampled_replay_difference=worst,
        all_design_candidates_independently_ensemble_repredicted=False, new_independent_bootstrap_fits_in_verifier=0,
        selection_seal_unchanged=True, external_data_validation=False, physical_safety_certified=False))
    print('Q3 robust numerical checks passed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_robust_20261027')
    with threadpool_limits(limits=1): run(parser.parse_args())
