"""Choose sparse Q3 plans using design refits, then open sealed refit validation."""
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
from beer_audit.audit import save_csv, sha256, write_json
from beer_models.robust_transfer import sparse_decode, panel_five, style_margins, refit_metrics, design_order
from beer_models.raw_panel import RawPanelModel
from beer_models.q3_raw import resampled_records
from beer_models.explain import stratified_draw
from beer_models.style_models import make_style_model
from q3_stage2 import setup, source_records
from r2_raw_q1 import PROTOCOL as BASE, RAW_MANIFEST, content_hash

PROTOCOL = ROOT/'configs/q3_robust_protocol_20261027.json'
OLD = ROOT/'outputs/q3_stage2_20261026'


def fit_refits(d, p, partition, out):
    config = p[partition+'_ensemble']
    bp = json.loads(BASE.read_text(encoding='utf-8'))
    sc = json.loads((ROOT/'configs/q3_style_protocol.json').read_text(encoding='utf-8'))
    rng = np.random.default_rng(config['seed'])
    models, draw_rows = [], []
    for repeat in range(config['refits']):
        draw = stratified_draw(d['ids'].tasting_category_fine.to_numpy(), rng)
        pos, scores, weights = resampled_records(d, draw)
        model = RawPanelModel('extra_trees', bp, config['seed']+repeat, d['parents']).fit(d['x'][draw], pos, scores, weights, d['distribution'][draw])
        np.testing.assert_allclose(model.imputer.statistics_, np.nanmedian(d['x'][draw], axis=0), atol=0, rtol=0)
        np.testing.assert_allclose(np.bincount(pos, weights=weights, minlength=len(draw)), 1, atol=1e-12, rtol=0)
        classifiers = {name: make_style_model(name, sc, config['seed']+repeat).fit(d['x'][draw], d['ids'].tasting_category_fine.iloc[draw]) for name in ['logistic', 'extra_trees']}
        item = dict(panel=model, classifiers=classifiers, sampled_beers=set(d['ids'].beer.iloc[draw]))
        joblib.dump(item, out/'models'/f'{partition}_{repeat}.joblib', compress=3)
        draw_rows.extend(dict(partition=partition, repeat=repeat, occurrence=i, beer=int(d['ids'].beer.iloc[j])) for i, j in enumerate(draw))
        models.append(item)
        print(f'{partition} model {repeat+1}/{config["refits"]}', flush=True)
    save_csv(out, partition+'_draws.csv', draw_rows)
    return models


def source_domain(source, budget, d, g, classifiers, dictionary, protected, p):
    model = classifiers['logistic']
    compressed = model[:-1].transform(np.vstack([source, g.x_[g.indices_['Tripel']]]))
    ti, si = [list(model.classes_).index(name) for name in ['Tripel', 'Pils/Lager']]
    contribution = (compressed[1:].mean(0)-compressed[0])*(model[-1].coef_[ti]-model[-1].coef_[si])
    allowed = np.array([i for i in range(len(source)) if i not in protected and contribution[i] > 0])
    mutable = allowed[np.argsort(-contribution[allowed], kind='stable')[:budget-3]]
    zs = g.robust_.transform(source[None])[0]
    indices = g.indices_['Tripel']
    dist = np.square(g.robust_.transform(g.x_[indices])[:, mutable]-zs[mutable]).mean(1)
    neighbors = indices[np.argsort(dist, kind='stable')[:p['neighbors']]]
    groups = np.array([p['groups'].index(origin) if origin in p['groups'] else p['groups'].index('Other') for origin in dictionary.origin_paper])
    return mutable, neighbors, groups, contribution


def chemical_limits(source, d, reference, p, source_beer=None):
    lower, log = [], []
    for field in p['reducible_fields']:
        j = d['features'].index(field)
        values = d['x'][:, j]
        actual = reference.loc[d['ids'].beer, field].to_numpy(float)
        mapped = np.log10(np.where(actual > 0, actual, np.nan)) if field in ['DMS', 'X4VG'] else actual
        keep = np.isfinite(values)&np.isfinite(mapped)&np.isclose(values, mapped, atol=1e-6, rtol=0)
        valid = values[keep]
        floor = float(valid.min())
        if field not in ['DMS', 'X4VG']:
            floor = max(floor, 0.)
        iqr = float(np.diff(np.quantile(valid, [.25, .75]))[0])
        limit = min(source[j], max(floor, source[j]-.5*iqr))
        if field == 'X4VG':
            # A source zero/censored reference does not establish an invertible log scale.
            if source_beer is None:
                raise ValueError('A source identity is required for positive-reference4VG control')
            source_reference = float(reference.loc[int(source_beer), field])
            if not np.isfinite(source_reference) or source_reference <= 0 or not np.isclose(np.log10(source_reference), source[j], atol=1e-6, rtol=0):
                limit = source[j]
        lower.append(limit)
        log.append(dict(field=field, source=source[j], lower=limit, empirical_floor=floor, compatible_rows=int(keep.sum()), observed_iqr=iqr))
    return np.array(lower), log


def ensemble_arrays(models, x, before, indices):
    grades, deltas, margins, styles = [], [], [], []
    for model, baseline in zip(models, before):
        a = panel_five(model['panel'], x, indices)
        m, ok = style_margins(model['classifiers'], x)
        grades.append(a['grade']-baseline['grade'])
        deltas.append(a['nonzero']-baseline['nonzero'])
        margins.append(m); styles.append(ok)
    return np.stack(grades), np.stack(deltas), np.stack(margins), np.stack(styles)


def evaluate(x, source, b, indices, classifiers, g, z, center, radius, protected, lower, reducible, models, before, p, budget):
    nominal = panel_five(b['risk_panel_model'].models['extra_trees'], x, indices)
    base = panel_five(b['risk_panel_model'].models['extra_trees'], source[None], indices)
    delta = nominal['nonzero']-base['nonzero']
    grade = nominal['grade']-base['grade']
    margin, style = style_margins(classifiers, x)
    zz, zs = g.robust_.transform(x), g.robust_.transform(source[None])[0]
    initial = float(np.sqrt(np.square(zs-center).mean()))
    target = np.sqrt(np.square(zz-center).mean(1)); change = np.sqrt(np.square(zz-zs).mean(1))
    support = cdist(zz, z).min(1)/np.sqrt(len(source))
    count = (abs(x-source) > 1e-10).sum(1)
    chemical = (x[:, protected] <= source[protected]+1e-10).all(1)
    if not p.get('allow_positive_4vg_reduction', False):
        chemical &= np.isclose(x[:, protected[2]], source[protected[2]], atol=1e-10, rtol=0)
    requirements = p['design_requirements']
    arrays = ensemble_arrays(models, x, before, indices)
    metrics = refit_metrics(*arrays, requirements)
    q, mq, fraction = metrics['nonzero_upper'], metrics['style_lower'], metrics['joint_fraction']
    improve = 1-target/initial
    structural = (~chemical).astype(int)+(improve < .1)+(support > radius+1e-10)+(count > budget)+(margin < p['nominal']['minimum_style_probability_margin']).sum(1)+(~style)
    sensory = (grade > 0).sum(1)+(delta[:, :4] > 1e-10).sum(1)
    nominal_pass = (structural == 0)&(sensory == 0)
    design_violations = (q > requirements['maximum_four_nonzero_delta_quantile']+1e-10).sum(1)+(mq < requirements['minimum_style_margin_quantile']).sum(1)+(fraction < requirements['minimum_joint_refit_fraction'])
    slack = np.maximum(q-requirements['maximum_four_nonzero_delta_quantile'], 0).sum(1)/.002+np.maximum(-mq, 0).sum(1)/.01+8*np.maximum(.75-fraction, 0)
    table = pd.DataFrame(dict(structural_violations=structural, nominal_sensory_violations=sensory,
        design_violation_count=design_violations, design_slack=slack, nominal_pass=nominal_pass,
        design_pass=metrics['qualified'], qualified=nominal_pass&metrics['qualified'], design_joint_fraction=fraction,
        changed_fields=count, target_distance=target, source_distance=initial, distance_improvement=improve,
        change_distance=change, support_distance=support, objective=.5*target/initial+.35*change+.15*count/len(source)))
    for j in range(4):
        table[f'nonzero_q80_{j}'] = q[:, j]
        table[f'nominal_nonzero_delta_{j}'] = delta[:, j]
    for j, name in enumerate(classifiers):
        table[name+'_margin'] = margin[:, j]; table[name+'_margin_q20'] = mq[:, j]
    return table


def run(args):
    _, d, b, g, classifiers, z, radius, center, targets, protected, dictionary, domains = setup(args.source, args.raw, args.q1)
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    out = args.output; out.mkdir(parents=True, exist_ok=False); (out/'models').mkdir()
    write_json(out/'protocol.json', p)
    write_json(out/'provenance.json', dict(protocol_sha256=content_hash(p), q1_sha256=sha256(args.q1/'model.joblib'),
        input_sha256={f: sha256(args.source/f) for f in ['1.csv', '3.xlsx', '4.csv', '8.xlsx']}, raw_sha256=sha256(args.raw)))
    sources, source_meta, missing, _ = source_records(d, b, json.loads((ROOT/'configs/q3_stage2_protocol_20261026.json').read_text(encoding='utf-8')))
    save_csv(out, 'source_status.csv', source_meta)
    save_csv(out, 'source_profiles.csv', [dict(beer=k, **dict(zip(d['features'], v))) for k, v in sources.items()])
    reference = pd.read_excel(args.source/'8.xlsx', sheet_name='Calibrated chemical dataset').set_index('beer')
    reference.index = reference.index.astype(int)
    indices = [d['targets'].index(t) for t in ['dms', 'diacetyl', 'X4vg', 't2n', 'stale_hops']]
    reducible = np.array([d['features'].index(f) for f in p['reducible_fields']])
    models = fit_refits(d, p, 'design', out)
    selected, parameters, details, neighbor_rows, field_rows, bound_rows, all_tables = [], [], [], [], [], [], []
    for ordinal, (beer, source) in enumerate(sources.items()):
        before = [panel_five(model['panel'], source[None], indices) for model in models]
        lower, log = chemical_limits(source, d, reference, p)
        bound_rows.extend(dict(beer=beer, **row) for row in log)
        for budget in p['budgets']:
            mutable, nearest, groups, contribution = source_domain(source, budget, d, g, classifiers, dictionary, protected, p)
            neighbors = g.x_[nearest]
            neighbor_rows.extend(dict(beer=beer, budget=budget, slot=k, target_beer=int(d['ids'].beer.iloc[j])) for k, j in enumerate(nearest))
            field_rows.extend(dict(beer=beer, budget=budget, field=d['features'][j], group=p['groups'][groups[j]], logistic_contribution=float(contribution[j])) for j in mutable)
            seed = p['seed']+ordinal*1000+budget
            rng = np.random.default_rng(seed)
            dimension = p['neighbors']+len(p['groups'])+len(reducible)
            initial = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(p['search']['sobol_power'])
            vertices = np.hstack([np.eye(p['neighbors']), np.ones((p['neighbors'], len(p['groups'])+len(reducible)))])
            u = np.vstack([initial, vertices])
            tables, chunks = [], []
            for round_number in range(p['search']['rounds']+1):
                x = sparse_decode(u, source, neighbors, mutable, groups, reducible, lower)
                t = evaluate(x, source, b, indices, classifiers, g, z, center, radius, protected, lower, reducible, models, before, p, budget)
                for k in range(dimension): t[f'u{k}'] = u[:, k]
                t['round'] = round_number
                tables.append(t); chunks.append(u)
                table = pd.concat(tables, ignore_index=True); matrix = np.concatenate(chunks)
                if round_number < p['search']['rounds']:
                    elite = design_order(table)[:p['search']['elite']]
                    u = np.clip(matrix[rng.choice(elite, p['search']['batch_size'])]+rng.normal(0, p['search']['scales'][round_number], (p['search']['batch_size'], dimension)), 0, 1)
            table['beer'], table['budget'], table['candidate'] = beer, budget, np.arange(len(table))
            good = table.loc[table.qualified]
            chosen = int(good.sort_values(['objective', 'changed_fields', 'candidate']).iloc[0].candidate) if len(good) else int(design_order(table)[0])
            profile = sparse_decode(matrix[chosen:chosen+1], source, neighbors, mutable, groups, reducible, lower)[0]
            selected.append(dict(beer=beer, budget=budget, **dict(zip(d['features'], profile))))
            parameters.append(dict(beer=beer, budget=budget, candidate=chosen, **{f'u{k}': v for k, v in enumerate(matrix[chosen])}))
            details.append(dict(beer=beer, budget=budget, selected_candidate=chosen, design_qualified_count=len(good), **table.iloc[chosen].drop(['beer', 'budget', 'candidate']).to_dict()))
            all_tables.append(table)
            print(f'Design beer{beer}, budget{budget}: {len(good)}/{len(table)} qualified; best joint={table.iloc[chosen].design_joint_fraction:.3f}', flush=True)
    for name, rows in [('selected_profiles.csv', selected), ('selected_parameters.csv', parameters), ('design_choices.csv', details), ('neighbors.csv', neighbor_rows), ('mutable_fields.csv', field_rows), ('chemical_bounds.csv', bound_rows)]:
        save_csv(out, name, rows)
    pd.concat(all_tables, ignore_index=True).to_csv(out/'design_candidates.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    choices = pd.DataFrame(details)
    final = []
    for beer, group in choices.groupby('beer', sort=False):
        good = group.loc[group.qualified]
        chosen = good.sort_values(['objective', 'changed_fields', 'budget']).iloc[0] if len(good) else group.iloc[design_order(group)[0]]
        final.append(dict(beer=int(beer), budget=int(chosen.budget), design_qualified=bool(chosen.qualified)))
    save_csv(out, 'final_choices_before_validation.csv', final)
    write_json(out/'sealed_selection.json', dict(validation_opened=False, selection_sha256={name: sha256(out/name) for name in ['selected_profiles.csv', 'selected_parameters.csv', 'design_choices.csv', 'final_choices_before_validation.csv']}, protocol_sha256=content_hash(p)))
    print('All choices saved and hashed before validation. Run validate_q3_robust.py next.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_robust_20261027')
    with threadpool_limits(limits=1):
        run(parser.parse_args())
