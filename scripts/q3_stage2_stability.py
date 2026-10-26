"""Fixed-plan parameter sensitivity and paired beer bootstrap; no reselection."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json
from beer_models.local_transfer import decode
from beer_models.raw_panel import RawPanelModel
from beer_models.q3_raw import resampled_records, uncalibrated_refit_predictions
from beer_models.explain import stratified_draw
from beer_models.style_models import make_style_model
from q3_stage2 import setup, evaluate, source_records
from r2_raw_q1 import PROTOCOL as BASE, RAW_MANIFEST


def run(args):
    p, d, b, g, classifiers, z, radius, center, target_indices, protected, dictionary, domains = setup(args.source, args.raw, args.q1)
    out = args.output
    parameters = pd.read_csv(out/'selected_parameters.csv', float_precision='round_trip')
    selected = pd.read_csv(out/'selected_profiles.csv', float_precision='round_trip')
    neighbors = pd.read_csv(out/'neighbors.csv')
    sources, _, _, _ = source_records(d, b, p)
    lookup = {int(beer): i for i, beer in enumerate(d['ids'].beer)}
    bounds = (g.x_[target_indices].min(0), g.x_[target_indices].max(0))
    jitter_tables = []
    from beer_models.local_transfer import defect_predictions
    for k, row in enumerate(parameters.itertuples()):
        rng = np.random.default_rng(p['bootstrap_seed']+10000+k)
        u = np.array([getattr(row, f'u{j}') for j in range(6)])
        perturbed = np.clip(u+rng.normal(0, .01, (32, 6)), 0, 1)
        ids = neighbors.loc[neighbors.beer.eq(row.beer)&neighbors.domain.eq(row.domain)].sort_values('slot').target_beer
        x = decode(perturbed, sources[row.beer], g.x_[[lookup[int(i)] for i in ids]], domains[row.domain])[0]
        before = defect_predictions(b, sources[row.beer][None], p['defects'])
        t = evaluate(x, perturbed, sources[row.beer], before, p, d, b, g, classifiers, z, radius, center, protected, bounds)
        t['beer'], t['domain'], t['draw'] = row.beer, row.domain, np.arange(32)
        jitter_tables.append(t)
    jitter = pd.concat(jitter_tables, ignore_index=True)
    save_csv(out, 'parameter_sensitivity.csv', jitter)
    summary = jitter.groupby(['beer', 'domain'], sort=False)[['main_computational_pass', 'with_stale_nonzero_pass', 'with_both_nonzero_pass', 'with_thirty_guard_pass']].mean().add_suffix('_fraction').reset_index()
    save_csv(out, 'parameter_sensitivity_summary.csv', summary)
    before_x = np.stack([sources[int(beer)] for beer in selected.beer])
    after_x = selected[d['features']].to_numpy(float)
    bp = json.loads(BASE.read_text(encoding='utf-8'))
    sc = json.loads((ROOT/'configs/q3_style_protocol.json').read_text(encoding='utf-8'))
    target_idx = [d['targets'].index(t) for t in p['defects']]
    rng = np.random.default_rng(p['bootstrap_seed'])
    rows, draws, defect_rows = [], [], []
    for repeat in range(p['bootstrap_repeats']):
        draw = stratified_draw(d['ids'].tasting_category_fine.to_numpy(), rng)
        positions, scores, weights = resampled_records(d, draw)
        model = RawPanelModel('extra_trees', bp, p['bootstrap_seed']+repeat, d['parents']).fit(d['x'][draw], positions, scores, weights, d['distribution'][draw])
        np.testing.assert_allclose(model.imputer.statistics_, np.nanmedian(d['x'][draw], axis=0), atol=0, rtol=0)
        np.testing.assert_allclose(np.bincount(positions, weights=weights, minlength=len(draw)), 1, atol=1e-12, rtol=0)
        a = uncalibrated_refit_predictions(model, before_x, d['parents'])
        c = uncalibrated_refit_predictions(model, after_x, d['parents'])
        grades = c['grade'][:, target_idx]-a['grade'][:, target_idx]
        delta = c['probability'][:, target_idx]-a['probability'][:, target_idx]
        style = {name: make_style_model(name, sc, p['bootstrap_seed']+repeat).fit(d['x'][draw], d['ids'].tasting_category_fine.iloc[draw]).predict(after_x) == p['target_style'] for name in classifiers}
        sampled = set(d['ids'].beer.iloc[draw])
        for i, item in enumerate(selected.itertuples()):
            grade_ok = bool((grades[i] <= 0).all())
            nonzero_ok = bool((delta[i, :4, 0] <= p['numeric_tolerance']).all())
            styles_ok = bool(all(values[i] for values in style.values()))
            rows.append(dict(repeat=repeat, beer=item.beer, domain=item.domain, source_in_bag=item.beer in sampled,
                grade_pass=grade_ok, supported_nonzero_pass=nonzero_ok, both_style_pass=styles_ok,
                joint_style_grade_four_nonzero=grade_ok and nonzero_ok and styles_ok,
                five_nonzero_pass=bool((delta[i, :, 0] <= p['numeric_tolerance']).all()), calibration_applied=False,
                logistic_style_pass=bool(style['logistic'][i]), extra_trees_style_pass=bool(style['extra_trees'][i])))
            for j, target in enumerate(p['defects']):
                defect_rows.append(dict(repeat=repeat, beer=item.beer, domain=item.domain, target=target,
                    grade_delta=int(grades[i, j]), nonzero_delta=float(delta[i, j, 0]), ge5_delta=float(delta[i, j, 1]), six_delta=float(delta[i, j, 2])))
        draws.extend(dict(repeat=repeat, occurrence=i, beer=int(d['ids'].beer.iloc[j])) for i, j in enumerate(draw))
        print(f'Paired fixed-plan bootstrap {repeat+1}/{p["bootstrap_repeats"]}', flush=True)
    rows = pd.DataFrame(rows)
    save_csv(out, 'bootstrap_checks.csv', rows)
    save_csv(out, 'bootstrap_defects.csv', defect_rows)
    save_csv(out, 'bootstrap_draws.csv', draws)
    summaries = []
    for (beer, domain), t in rows.groupby(['beer', 'domain'], sort=False):
        summaries.append(dict(beer=int(beer), domain=domain, refits=len(t), source_oob_refits=int((~t.source_in_bag).sum()),
            joint_fraction=float(t.joint_style_grade_four_nonzero.mean()),
            source_oob_joint_fraction=float(t.loc[~t.source_in_bag, 'joint_style_grade_four_nonzero'].mean()) if (~t.source_in_bag).any() else None,
            supported_nonzero_fraction=float(t.supported_nonzero_pass.mean()), both_style_fraction=float(t.both_style_pass.mean()),
            grade_fraction=float(t.grade_pass.mean()), safety_certified=False))
    save_csv(out, 'bootstrap_summary.csv', summaries)
    write_json(out/'stability_scope.json', dict(fixed_plan_refits=p['bootstrap_repeats'], raw_ET_fits=p['bootstrap_repeats'],
        style_classifier_fits=2*p['bootstrap_repeats'], parameter_perturbations=len(jitter), selection_repeated=False,
        original_sigmoids_attached_to_refit_forests=False, source_OOB_counts_reported=True,
        evidence='model_and_parameter_sensitivity_on_same_250_beers_not_intervention_or_external_validation'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_stage2_20261026')
    with threadpool_limits(limits=1):
        run(parser.parse_args())
