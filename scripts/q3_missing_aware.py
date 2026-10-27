"""Fixed223-input Q3 model for a genuinely missing source coordinate."""
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
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.tail_risk import base_probabilities, binary_loss
from beer_models.raw_panel import RawPanelModel
from beer_models.style_models import make_style_model
from beer_models.robust_transfer import panel_five, style_margins, refit_metrics
from beer_models.q3_raw import resampled_records
from beer_models.explain import stratified_draw
from beer_models.generation import StyleGenerator
from beer_models.uncertain_coordinate import preserve_unknown
from q3_risk_aware_pruning import FIRST, parameter_draws
from q3_stage2 import setup, source_records
from r2_raw_q1 import RAW_MANIFEST, PROTOCOL as BASE, content_hash
from q1_target_experiment import fit_shared

PROTOCOL = ROOT/'configs/q3_missing_aware_protocol_20261027.json'
STALE = ROOT/'outputs/q3_stale_risk_20261027'
CURRENT = ROOT/'outputs/q3_pruning_20261027'


def run(args):
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    _, d, official, _, _, _, _, _, _, _, _, _ = setup(args.source, args.raw, args.q1)
    out = args.output; out.mkdir(parents=True, exist_ok=False); (out/'models').mkdir()
    write_json(out/'protocol.json', p)
    omitted = d['features'].index(p['omitted_input']); keep = np.array([j for j in range(len(d['features'])) if j != omitted])
    small = dict(d, x=d['x'][:, keep], features=[d['features'][j] for j in keep])
    bp = json.loads(BASE.read_text(encoding='utf-8')); sc = json.loads((ROOT/'configs/q3_style_protocol.json').read_text(encoding='utf-8'))
    indices = [d['targets'].index(t) for t in p['sensory_targets']]
    five = indices+[d['targets'].index('stale_hops')]
    y = 1-d['distribution'][:, indices, 0]
    predictions = {name: np.empty((3, len(y), 4)) for name in ['prior', 'full', 'masked']}
    style_rows, split_rows = [], []
    for repeat in range(3):
        for fold in range(3):
            old = joblib.load(STALE/'models'/f'r{repeat}_fold{fold}.joblib')
            train, test = old['train'], old['test']; seed = old['models']['shared_et'].seed
            model = fit_shared(small, train, bp, seed)
            np.testing.assert_allclose(model.imputer.statistics_, np.nanmedian(small['x'][train], axis=0), atol=0, rtol=0)
            predictions['prior'][repeat, test] = y[train].mean(0)
            predictions['full'][repeat, test] = 1-base_probabilities(old['models']['shared_et'], d['x'][test])[:, indices, 0]
            predictions['masked'][repeat, test] = 1-base_probabilities(model, small['x'][test])[:, indices, 0]
            styles = {}
            for method in ['logistic', 'extra_trees']:
                for domain, data in [('full', d), ('masked', small)]:
                    style = make_style_model(method, sc, seed).fit(data['x'][train], data['ids'].tasting_category_fine.iloc[train])
                    pred = style.predict(data['x'][test])
                    styles[domain+'_'+method] = style
                    style_rows.extend(dict(repeat=repeat, fold=fold, beer=int(beer), classifier=method, domain=domain,
                        observed=str(actual), predicted=str(guess), correct=bool(actual == guess)) for beer, actual, guess in zip(d['ids'].beer.iloc[test], d['ids'].tasting_category_fine.iloc[test], pred))
            joblib.dump(dict(panel=model, classifiers=styles, train=train, test=test), out/'models'/f'outer_{repeat}_{fold}.joblib', compress=3)
            split_rows.extend(dict(repeat=repeat, fold=fold, role=role, beer=int(d['ids'].beer.iloc[i])) for role, positions in [('train', train), ('test', test)] for i in positions)
            print(f'Missing-aware outer {repeat}/{fold}', flush=True)
    metric_rows, loss_rows, intervals = [], [], []
    for method, prob in predictions.items():
        errors = binary_loss(prob, y[None])
        for repeat in range(3):
            for j, target in enumerate(p['sensory_targets']):
                metric_rows.append(dict(method=method, repeat=repeat, target=target, Brier=float(errors[repeat, :, j].mean())))
                loss_rows.extend(dict(method=method, repeat=repeat, beer=int(beer), target=target, observed_fraction=float(q), probability=float(v), Brier=float(e)) for beer, q, v, e in zip(d['ids'].beer, y[:, j], prob[repeat, :, j], errors[repeat, :, j]))
    rng = np.random.default_rng(20261207); styles = d['ids'].tasting_category_fine.to_numpy()
    draws = np.stack([stratified_draw(styles, rng) for _ in range(2000)])
    target_gates = []
    for j, target in enumerate(p['sensory_targets']):
        masked = binary_loss(predictions['masked'][:, :, j], y[None, :, j]).mean(0)
        prior = binary_loss(predictions['prior'][:, :, j], y[None, :, j]).mean(0)
        full = binary_loss(predictions['full'][:, :, j], y[None, :, j]).mean(0)
        gain, increase = prior-masked, masked-full
        lower99, upper99 = np.quantile(gain[draws].mean(1), [.005, .995])
        lower95, upper95 = np.quantile(increase[draws].mean(1), [.025, .975])
        intervals.append(dict(target=target, prior_gain=float(gain.mean()), prior_lower99=float(lower99), prior_upper99=float(upper99),
            full_Brier_increase=float(increase.mean()), full_lower95=float(lower95), full_upper95=float(upper95)))
        target_gates.append(bool(lower99 > 0 and increase.mean() <= .0005 and upper95 <= .001))
    style_frame = pd.DataFrame(style_rows)
    accuracy = style_frame.groupby(['classifier','domain']).correct.mean().unstack('domain')
    style_ok = bool(((accuracy.full-accuracy.masked) <= .03).all())
    predictive_ok = all(target_gates) and style_ok
    for name, rows in [('metrics.csv', metric_rows), ('oof_predictions.csv', loss_rows), ('paired_intervals.csv', intervals), ('style_oof.csv', style_frame), ('split_membership.csv', split_rows)]: save_csv(out, name, rows)
    save_csv(out, 'style_accuracy.csv', accuracy.reset_index())
    write_json(out/'predictive_qualification.json', dict(passed=predictive_ok, four_target_gates=target_gates, style_gate=style_ok,
        raw_sha256=sha256(args.raw), protocol_sha256=content_hash(p)))
    print(pd.DataFrame(intervals).to_string(index=False), flush=True)
    if not predictive_ok:
        write_json(out/'summary.json', dict(status='predictive_gate_failed_not_adopted', predictive_qualified=False,
            zero_assumption_removed_for_qualified_Q3=False, true_all_five_safety_certified=False, q1_model_changed=False, phase3_started=False))
        return
    old_p = json.loads((ROOT/'configs/q3_stage2_protocol_20261026.json').read_text(encoding='utf-8'))
    source = source_records(d, official, old_p)[0][119]
    proposed = pd.read_csv(CURRENT/'selected_profiles.csv', float_precision='round_trip').set_index('beer').loc[119, d['features']].to_numpy(float)
    missing_source = source.copy(); missing_source[omitted] = np.nan
    missing_after = preserve_unknown(missing_source, proposed, omitted)
    generator = StyleGenerator().fit(small['x'], styles)
    z = generator.robust_.transform(generator.x_); distances = cdist(z,z)/np.sqrt(len(keep)); np.fill_diagonal(distances, np.inf)
    radius = float(np.quantile(distances.min(1), .95)); center = z[generator.indices_['Tripel']].mean(0)
    a, c = missing_source[keep], missing_after[keep]
    za, zc = generator.robust_.transform(np.vstack([a,c]))
    source_distance, target_distance = np.sqrt(np.square(za-center).mean()), np.sqrt(np.square(zc-center).mean())
    improvement = float(1-target_distance/source_distance)
    support = float(np.sqrt(np.square(z-zc).mean(1)).min())
    param_full = parameter_draws(source, proposed, 32, 20261209)
    param_full[:, omitted] = np.nan
    params = param_full[:, keep]
    models, draw_log = [], []
    for partition, count, parent in [('development',20,FIRST), ('validation',12,CURRENT)]:
        for i in range(count):
            if partition == 'development':
                role, number = ('design', i) if i < 8 else ('validation', i-8)
                saved = joblib.load(FIRST/'models'/f'{role}_{number}.joblib')
                old_draws = pd.read_csv(FIRST/(role+'_draws.csv'))
            else:
                number=i; saved=joblib.load(CURRENT/'models'/f'validation_{i}.joblib'); old_draws=pd.read_csv(CURRENT/'validation_draws.csv')
            order = old_draws.loc[old_draws.repeat.eq(number)].sort_values('occurrence').beer.to_numpy(int)
            lookup = {int(beer): n for n,beer in enumerate(d['ids'].beer)}; draw = np.array([lookup[int(beer)] for beer in order])
            positions, scores, weights = resampled_records(small, draw)
            model = RawPanelModel('extra_trees', bp, saved['panel'].seed, d['parents']).fit(small['x'][draw], positions, scores, weights, d['distribution'][draw])
            seed = saved['classifiers']['extra_trees'][-1].random_state
            classifiers = {name: make_style_model(name,sc,seed).fit(small['x'][draw],d['ids'].tasting_category_fine.iloc[draw]) for name in ['logistic','extra_trees']}
            item=dict(panel=model,classifiers=classifiers,source_oob=119 not in set(order))
            joblib.dump(item,out/'models'/f'{partition}_{i}.joblib',compress=3); models.append((partition,item))
            draw_log.extend(dict(partition=partition,repeat=i,occurrence=n,beer=int(beer)) for n,beer in enumerate(order))
            print(f'Matched missing-aware {partition} {i+1}/{count}',flush=True)
    req = json.loads((CURRENT/'base_protocol.json').read_text(encoding='utf-8'))
    records, role_metrics, parameter_ensemble = [], {}, {}
    for partition, rkey in [('development','design_requirements'),('validation','validation_adoption')]:
        members = [item for role,item in models if role == partition]
        grades, risks, margins, passes = [], [], [], []
        x = np.vstack([c,params])
        for repeat,item in enumerate(members):
            before=panel_five(item['panel'],a[None],five); after=panel_five(item['panel'],x,five)
            margin,passed=style_margins(item['classifiers'],x)
            grades.append(after['grade']-before['grade']);risks.append(after['nonzero']-before['nonzero']);margins.append(margin);passes.append(passed)
            records.append(dict(role=partition,repeat=repeat,source_oob=item['source_oob'],joint_pass=bool(passed[0] and (grades[-1][0] <= 0).all() and (risks[-1][0,:4] <= 1e-10).all())))
        metrics=refit_metrics(np.stack(grades),np.stack(risks),np.stack(margins),np.stack(passes),req[rkey])
        parameter_ensemble[partition]=metrics['qualified'][1:]
        role_metrics[partition]=dict(qualified=bool(metrics['qualified'][0]),joint_fraction=float(metrics['joint_fraction'][0]), risk_delta_q80=metrics['nonzero_upper'][0].tolist(),
            parameter_fraction=float(metrics['qualified'][1:].mean()))
    full=fit_shared(small,np.arange(len(styles)),bp,202612099)
    class_full={name:make_style_model(name,sc,sc['seed']).fit(small['x'],d['ids'].tasting_category_fine) for name in ['logistic','extra_trees']}
    all_targets=np.vstack([c,params])
    before=panel_five(full,a[None],five);after=panel_five(full,all_targets,five);margin,_=style_margins(class_full,all_targets)
    official_before=official['raw_panel_model'].predict(missing_source[None])['grade'][:,five]
    official_after=official['raw_panel_model'].predict(np.vstack([missing_after,param_full]))['grade'][:,five]
    nominal=(after['grade'] <= before['grade']).all(1)&(after['nonzero'][:,:4] <= before['nonzero'][:,:4]+1e-10).all(1)&(margin >= .01).all(1)&(official_after <= official_before).all(1)
    target_z=generator.robust_.transform(all_targets)
    parameter_distances=np.sqrt(np.square(target_z-center).mean(1))
    nearest=cdist(target_z,z).min(1)/np.sqrt(len(keep))
    nominal&=(1-parameter_distances/source_distance >= .1)&(nearest <= radius+1e-10)
    nominal_ok=bool(nominal[0])
    parameter_rows=[]
    for k in range(32):
        parameter_rows.append(dict(draw=k,nominal_qualified=bool(nominal[k+1]),development_qualified=bool(parameter_ensemble['development'][k]),combined_qualified=bool(nominal[k+1] and parameter_ensemble['development'][k])))
    role_metrics['development']['parameter_fraction']=float((nominal[1:]&parameter_ensemble['development']).mean())
    role_metrics['validation']['parameter_fraction']=float((nominal[1:]&parameter_ensemble['validation']).mean())
    known_protected=[d['features'].index(f) for f in ['DMS','X4VG','X.E..2.nonenal']]
    symbolic_guard=bool(np.isnan(missing_source[omitted]) and np.isnan(missing_after[omitted]) and (missing_after[known_protected] <= missing_source[known_protected]+1e-10).all())
    changed_count=int((abs(c-a)>1e-10).sum())
    adopted=bool(nominal_ok and symbolic_guard and changed_count <= 144 and improvement >= .1 and support <= radius and role_metrics['development']['qualified'] and role_metrics['validation']['qualified'] and role_metrics['development']['parameter_fraction'] >= .8)
    save_csv(out,'matched_draws.csv',draw_log);save_csv(out,'transfer_checks.csv',records)
    save_csv(out,'parameter_checks.csv',parameter_rows)
    save_csv(out,'source119_profile.csv',[dict(beer=119,**dict(zip(d['features'],missing_source)))])
    save_csv(out,'transfer119_profile.csv',[dict(beer='transfer119',**dict(zip(d['features'],missing_after)))])
    joblib.dump(dict(panel=full,classifiers=class_full,keep_indices=keep,features=small['features'],omitted_input=p['omitted_input']),out/'model.joblib',compress=3)
    write_json(out/'transfer_rule.json',dict(operation='preserve_unknown_source_Diacetyl',source_coordinate=None,target_coordinate=None,
        exact_zero_assumption_required=False,missing_input_not_converted_to_measurement=True,chemical_nonincrease_is_conditional_on_preserving_actual_unknown=True,
        omitted_coordinate_cannot_affect_auxiliary_model_by_construction=True))
    write_json(out/'summary.json',dict(status='completed',predictive_qualified=True,adopted_four_proxy_missing_aware_plan=adopted,
        zero_assumption_removed_for_qualified_Q3=adopted,role_metrics=role_metrics,nominal_qualified=nominal_ok,
        distance_improvement_223_available_coordinates=improvement,nearest_support_distance=support,support_radius=radius,
        q1_model_sha256=sha256(args.q1/'model.joblib'),q1_model_changed=False,stale_evidence_sufficient=False,
        true_all_five_safety_certified=False,phase3_started=False,development_only_matched_sensitivity=True))
    print(f'MISSING-AWARE ADOPTED={adopted}; roles={role_metrics}',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--raw',type=Path,default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1',type=Path,default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/q3_missing_aware_20261027')
    with threadpool_limits(limits=1):run(parser.parse_args())
