"""Risk-aware sparse deletion after disclosure of the first robustness failure."""
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
from q3_stage2 import setup, source_records
from q3_robust_transfer import fit_refits, evaluate, chemical_limits, ensemble_arrays, OLD
from r2_raw_q1 import RAW_MANIFEST, content_hash

FIRST = ROOT/'outputs/q3_robust_20261027'
PROTOCOL = ROOT/'configs/q3_pruning_protocol_20261027.json'


def surplus(table):
    parts = [(table.design_joint_fraction.to_numpy()-.75)/.25]
    parts.extend((-table[f'nonzero_q80_{j}'].to_numpy()-.002)/.002 for j in range(4))
    parts.extend(table[name+'_margin_q20'].to_numpy()/.01 for name in ['logistic', 'extra_trees'])
    return np.stack(parts).min(0)


def parameter_draws(source, selected, n, seed):
    mutable = np.flatnonzero(abs(selected-source) > 1e-10)
    rng = np.random.default_rng(seed)
    progress = np.clip(1+rng.normal(0, .01, (n, len(mutable))), 0, 1)
    result = np.broadcast_to(selected, (n, len(source))).copy()
    result[:, mutable] = source[mutable]+progress*(selected[mutable]-source[mutable])
    return result


def run(args):
    r = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    p = json.loads((FIRST/'protocol.json').read_text(encoding='utf-8'))
    p['validation_ensemble'] = r['validation_ensemble']
    out = args.output; out.mkdir(parents=True, exist_ok=False); (out/'models').mkdir()
    write_json(out/'protocol.json', r)
    write_json(out/'base_protocol.json', p)
    _, d, b, g, classifiers, z, radius, center, target_indices, protected, dictionary, domains = setup(args.source, args.raw, args.q1)
    old_p = json.loads((ROOT/'configs/q3_stage2_protocol_20261026.json').read_text(encoding='utf-8'))
    sources, _, _, _ = source_records(d, b, old_p)
    indices = [d['targets'].index(t) for t in old_p['defects']]
    reducible = np.array([d['features'].index(f) for f in p['reducible_fields']])
    reference = pd.read_excel(args.source/'8.xlsx', sheet_name='Calibrated chemical dataset').set_index('beer')
    reference.index = reference.index.astype(int)
    development = [joblib.load(FIRST/'models'/f'{partition}_{i}.joblib') for partition, n in [('design', 8), ('validation', 12)] for i in range(n)]
    old = pd.read_csv(OLD/'selected_profiles.csv', float_precision='round_trip').loc[lambda t: t.domain.eq('expanded_sensitivity')].set_index('beer')
    prior = pd.read_csv(FIRST/'selected_profiles.csv', float_precision='round_trip').set_index(['beer', 'budget'])
    selected, assessments, histories, seed_metrics, stress_rows = [], [], [], [], []
    for ordinal, (beer, source) in enumerate(sources.items()):
        lower, _ = chemical_limits(source, d, reference, p)
        before = [panel_five(model['panel'], source[None], indices) for model in development]
        def check(x, budget=224):
            t = evaluate(x, source, b, indices, classifiers, g, z, center, radius, protected, lower, reducible, development, before, p, budget)
            t['guard_surplus'] = surplus(t)
            return t
        seeds, seed_names = [], []
        for target in g.indices_['Tripel']:
            base = g.x_[target].copy()
            base[protected[2]] = source[protected[2]]
            base[reducible] = np.minimum(np.maximum(base[reducible], lower), source[reducible])
            for level in r['chemical_seed_levels']:
                x = base.copy(); x[reducible] = source[reducible]-level*(source[reducible]-lower)
                seeds.append(x); seed_names.append(f'Tripel_{int(d["ids"].beer.iloc[target])}_reduction_{level}')
        for name, row in [('previous_broad', old.loc[beer]), ('robust96', prior.loc[(beer, 96)]), ('robust144', prior.loc[(beer, 144)])]:
            x = row[d['features']].to_numpy(float)
            seeds.append(x); seed_names.append(name)
        seeds = np.stack(seeds)
        table = check(seeds)
        table['beer'], table['seed'] = beer, seed_names
        seed_metrics.append(table)
        good = table.loc[table.qualified].sort_values(['guard_surplus', 'objective'], ascending=[False, True])
        completed = []
        for index in good.index[:3]:
            current = seeds[index].copy()
            for iteration in range(80):
                state = check(current[None]).iloc[0]
                histories.append(dict(beer=beer, seed=seed_names[index], iteration=iteration, **state.to_dict()))
                if state.changed_fields <= r['maximum_changed_fields']: break
                changed = np.flatnonzero((abs(current-source) > 1e-10)&~np.isin(np.arange(len(source)), protected))
                zs, zc = g.robust_.transform(source[None])[0], g.robust_.transform(current[None])[0]
                ordered = changed[np.argsort(abs(zc[changed]-zs[changed]), kind='stable')]
                accepted = False
                for size in [16, 8, 4, 1]:
                    chunks = [ordered[j:j+size] for j in range(0, len(ordered), size)]
                    if not chunks: continue
                    proposals = np.broadcast_to(current, (len(chunks), len(source))).copy()
                    for j, fields in enumerate(chunks): proposals[j, fields] = source[fields]
                    t = check(proposals)
                    acceptable = t.loc[t.qualified].sort_values(['changed_fields', 'guard_surplus', 'objective'], ascending=[True, False, True])
                    if len(acceptable):
                        current = proposals[int(acceptable.index[0])]
                        accepted = True; break
                if not accepted: break
            state = check(current[None], r['maximum_changed_fields']).iloc[0]
            if state.qualified:
                jitter = parameter_draws(source, current, 8, r['seed']+ordinal*100+int(index))
                stress = check(jitter, r['maximum_changed_fields'])
                hits = int(stress.qualified.sum())
                stress_rows.extend(dict(beer=beer, seed=seed_names[index], draw=k, **part.to_dict()) for k, part in stress.iterrows())
                if hits >= r['parameter_design_stress']['required_passes']:
                    completed.append((current, state, hits, seed_names[index]))
        if completed:
            chosen = sorted(completed, key=lambda item: (-item[1].guard_surplus, item[1].objective, item[1].changed_fields))[0]
            x, state, hits, name = chosen
            qualified = True
        else:
            # Preserve previous sparse failure rather than quietly present a dense seed as adopted.
            final = pd.read_csv(FIRST/'final_choices_before_validation.csv').set_index('beer')
            budget = int(final.loc[beer, 'budget'])
            x = prior.loc[(beer, budget), d['features']].to_numpy(float)
            state = check(x[None], r['maximum_changed_fields']).iloc[0]
            hits, name, qualified = 0, 'previous_sparse_failed_fallback', False
        selected.append(dict(beer=beer, **dict(zip(d['features'], x))))
        assessments.append(dict(beer=beer, selected_seed=name, design_stress_passes=hits,
            design_selected=qualified, qualified_seed_count=len(good), **state.to_dict()))
        print(f'Pruning beer{beer}: {len(good)} qualified dense seeds, sparse+stress selected={qualified}', flush=True)
    save_csv(out, 'seed_checks.csv', pd.concat(seed_metrics, ignore_index=True))
    save_csv(out, 'pruning_history.csv', histories)
    save_csv(out, 'design_parameter_checks.csv', stress_rows)
    save_csv(out, 'selected_profiles.csv', selected); save_csv(out, 'design_assessment.csv', assessments)
    seal = {name: sha256(out/name) for name in ['selected_profiles.csv', 'design_assessment.csv']}
    write_json(out/'sealed_selection.json', dict(selection_sha256=seal, protocol_sha256=content_hash(r),
        viewed_previous_validation_models_reclassified_development=12, fresh_validation_models_opened=False))
    fresh = fit_refits(d, p, 'validation', out)
    records, defect_records, results = [], [], []
    previous_final = pd.read_csv(FIRST/'final_choices_before_validation.csv').set_index('beer')
    for ordinal, (beer, source) in enumerate(sources.items()):
        chosen = selected[ordinal]
        sparse = np.array([chosen[f] for f in d['features']])
        first_sparse = prior.loc[(beer, int(previous_final.loc[beer, 'budget'])), d['features']].to_numpy(float)
        x = np.vstack([sparse, first_sparse, old.loc[beer, d['features']].to_numpy(float)])
        baseline = [panel_five(model['panel'], source[None], indices) for model in fresh]
        arrays = ensemble_arrays(fresh, x, baseline, indices)
        metrics = refit_metrics(*arrays, p['validation_adoption'])
        before_dev = [panel_five(model['panel'], source[None], indices) for model in development]
        lower, _ = chemical_limits(source, d, reference, p)
        jitter = parameter_draws(source, sparse, 32, r['seed']+90000+ordinal)
        stress = evaluate(jitter, source, b, indices, classifiers, g, z, center, radius, protected, lower, reducible, development, before_dev, p, 144)
        stress_fraction = float(stress.qualified.mean())
        for i, method in enumerate(['risk_aware_pruned', 'previous_sparse', 'previous_broad']):
            for repeat, model in enumerate(fresh):
                joint = bool(arrays[3][repeat, i] and (arrays[0][repeat, i] <= 0).all() and (arrays[1][repeat, i, :4] <= 1e-10).all())
                records.append(dict(beer=beer, method=method, repeat=repeat, source_oob=beer not in model['sampled_beers'], joint_pass=joint))
                for j, target in enumerate(old_p['defects']):
                    defect_records.append(dict(beer=beer, method=method, repeat=repeat, target=target,
                        grade_delta=int(arrays[0][repeat, i, j]), nonzero_delta=float(arrays[1][repeat, i, j])))
            results.append(dict(beer=beer, method=method, validation_joint_fraction=metrics['joint_fraction'][i],
                validation_qualified=bool(metrics['qualified'][i]), changed_fields=int((abs(x[i]-source) > 1e-10).sum()),
                design_selected=bool(assessments[ordinal]['design_selected']) if i == 0 else False,
                parameter_qualified_fraction=stress_fraction if i == 0 else np.nan,
                stable_four_proxy_conditional=bool(i == 0 and assessments[ordinal]['design_selected'] and metrics['qualified'][i] and stress_fraction >= .8),
                all_five_true_nonworsening_certified=False, actual_after_measured=False))
    save_csv(out, 'validation_checks.csv', records); save_csv(out, 'validation_defects.csv', defect_records)
    save_csv(out, 'plan_assessment.csv', results)
    for name, digest in seal.items(): assert sha256(out/name) == digest
    write_json(out/'summary.json', dict(status='completed_adaptive_development_and_fresh_sealed_checks',
        design_models=20, newly_sealed_validation_refits=12, design_selected_plans=sum(item['design_selected'] for item in assessments),
        stable_four_proxy_conditional_plans=sum(item['stable_four_proxy_conditional'] for item in results),
        all_five_true_nonworsening_certified=0, stale_evidence_sufficient=False, q1_model_changed=False, phase3_started=False,
        selection_sha256_unchanged=seal, raw_sha256=sha256(args.raw), q1_sha256=sha256(args.q1/'model.joblib')))
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_pruning_20261027')
    with threadpool_limits(limits=1): run(parser.parse_args())
