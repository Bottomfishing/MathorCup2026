"""Finite high-grade decisions and training-only risk calibration, grouped by beer."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.baseline import make_splits
from beer_models.explain import stratified_draw
from beer_models.raw_panel import RawPanelModel, median_grades, rating_loss, predict_raw_frame
from beer_models.tail_risk import (RawOrdinalModel, base_probabilities, cost_decision, tail_metrics,
    feasible, fit_risk, TailPointModel, survival, binary_loss, risk_frame)
from r2_raw_q1 import load_inputs, content_hash, RAW_MANIFEST

PROTOCOL = ROOT / 'configs/r2b_tail_risk_protocol_20261026.json'
BASE_PROTOCOL = ROOT / 'configs/r2_raw_panel_protocol_20261026.json'


def select_point(probabilities, observed, parents, protocol):
    candidates, rows = {}, []
    for name in protocol['base_models']:
        for cost, weights in protocol['cost_weights'].items():
            key = name + '__' + cost
            candidates[key] = cost_decision(probabilities[name], weights, parents)[0]
    baseline = tail_metrics(observed, candidates['extra_trees__median'])['raw_rating_mae']
    for name, grades in candidates.items():
        metrics = tail_metrics(observed, grades)
        rows.append(dict(candidate=name, **metrics, feasible=feasible(metrics, baseline, protocol['point_limits'])))
    eligible = [r for r in rows if r['feasible']]
    if eligible:
        winner = min(eligible, key=lambda r: (-(r['raw_recall_5']+r['raw_recall_6'])/2, r['raw_rating_mae'], list(candidates).index(r['candidate'])))['candidate']
    else:
        winner = 'extra_trees__median'
    for row in rows:
        row['selected'] = row['candidate'] == winner
        row['selection_had_feasible_candidate'] = bool(eligible)
    return winner, rows


def risk_tables(probabilities, observed, targets, protocol):
    metrics, curves = [], []
    actual = survival(observed)
    edges = np.array(protocol['risk_curve_edges'])
    for method, p in probabilities.items():
        cumulative = survival(p)
        for j, target in enumerate(targets):
            for cut in protocol['risk_events']:
                pred, truth = cumulative[:, j, cut-1], actual[:, j, cut-1]
                bins = np.clip(np.searchsorted(edges, pred, side='right') - 1, 0, len(edges)-2)
                ece = 0.
                for k in range(len(edges)-1):
                    mask = bins == k
                    if not mask.any():
                        continue
                    mean_p, mean_y = float(pred[mask].mean()), float(truth[mask].mean())
                    ece += mask.mean() * abs(mean_p-mean_y)
                    curves.append(dict(method=method, target=target, rating_ge=cut, bin_lower=edges[k],
                        bin_upper=edges[k+1], n_beers=int(mask.sum()), mean_probability=mean_p, observed_fraction=mean_y))
                pp = np.clip(pred, 1e-12, 1-1e-12)
                metrics.append(dict(method=method, target=target, rating_ge=cut,
                    brier=float(binary_loss(pred, truth).mean()),
                    log_loss=float(-(truth*np.log(pp)+(1-truth)*np.log(1-pp)).mean()),
                    calibration_ece=float(ece), mean_probability=float(pred.mean()), observed_fraction=float(truth.mean())))
    return pd.DataFrame(metrics), pd.DataFrame(curves)


def paired_intervals(difference, styles, protocol, quantiles):
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    draws = [difference[stratified_draw(styles, rng)].mean() for _ in range(protocol['bootstrap_repeats'])]
    return float(difference.mean()), *map(float, np.quantile(draws, quantiles))


def calibration_records(model, context, targets):
    records = []
    for j, target in enumerate(targets):
        for k, fitted in enumerate(model.calibrators[j]):
            records.append(dict(context=context, target=target, rating_ge=k+1, source=model.choices[j],
                kind='constant' if isinstance(fitted, float) else 'sigmoid',
                constant=fitted if isinstance(fitted, float) else np.nan,
                intercept=np.nan if isinstance(fitted, float) else float(fitted.intercept_[0]),
                slope=np.nan if isinstance(fitted, float) else float(fitted.coef_[0, 0])))
    return records


def export_probabilities(output, name, probabilities, ids, targets):
    frames = []
    for method, values in probabilities.items():
        frames.append(pd.DataFrame({'method': method, 'beer': np.repeat(ids, len(targets)),
            'target': np.tile(targets, len(ids)), **{f'p{k}': values[:, :, k].ravel() for k in range(7)}}))
    pd.concat(frames, ignore_index=True).to_csv(output/name, index=False, compression={'method':'gzip','mtime':0})


def run(args):
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    base_protocol = json.loads(BASE_PROTOCOL.read_text(encoding='utf-8'))
    d = load_inputs(args.source, args.raw, base_protocol)
    x, q, ids, targets = d['x'], d['distribution'], d['ids'], d['targets']
    styles = ids.tasting_category_fine.to_numpy()
    splits = make_splits(styles, base_protocol['outer_folds'], base_protocol['outer_seed'])
    output = args.output
    output.mkdir(parents=True, exist_ok=False); (output/'models').mkdir()
    write_json(output/'protocol.json', protocol)
    fit_log, split_log, point_choices, risk_choices, cal_log, inner_frames = [], [], [], [], [], []
    base_oof = {name:np.empty_like(q) for name in protocol['base_models']}
    risk_oof = {name:np.empty_like(q) for name in ['training_prior','selected_uncalibrated','selected_calibrated']}
    point_oof = {name+'__'+cost:np.empty(q.shape[:2],int) for name in protocol['base_models'] for cost in protocol['cost_weights']}
    point_oof['inner_selected'] = np.empty(q.shape[:2],int)
    folds = np.empty(len(x),int)
    sources_oof = np.empty(q.shape[:2],object)

    def fit(name, train, seed, context):
        tick = time.perf_counter(); print(f'Fit {context} {name}: {len(train)} beers', flush=True)
        if name == 'raw_ordinal':
            model = RawOrdinalModel(protocol).fit(x[train], q[train])
        else:
            lookup = np.full(len(x),-1,int); lookup[train] = np.arange(len(train))
            rows = lookup[d['positions']] >= 0
            model = RawPanelModel('extra_trees',base_protocol,seed,d['parents']).fit(x[train],
                lookup[d['positions'][rows]],d['scores'][rows],d['weights'][rows],q[train])
        np.testing.assert_allclose(model.imputer.statistics_, np.nanmedian(x[train],axis=0),rtol=0,atol=0)
        fit_log.append(dict(context=context,source=name,seed=seed,n_beers=len(train),seconds=time.perf_counter()-tick))
        return model

    full_models = full_risk = full_choice = None
    for number in range(len(splits)+1):
        full = number == len(splits)
        train, test = (np.arange(len(x)),np.array([],int)) if full else splits[number]
        context = 'full_training' if full else f'outer_{number}'
        inner = make_splits(styles[train],base_protocol['inner_folds'],base_protocol['inner_seed']+number)
        probs = {name:np.empty((len(train),len(targets),7)) for name in ['prior',*protocol['base_models']]}
        for fold,(it,iv) in enumerate(inner):
            for role,positions in [('train',train[it]),('validation',train[iv])]:
                split_log.extend(dict(context=context,inner_fold=fold,role=role,beer=int(ids.beer.iloc[p])) for p in positions)
            probs['prior'][iv] = q[train[it]].mean(axis=0)
            for name in protocol['base_models']:
                model = fit(name,train[it],base_protocol['model_seed']+number*100+fold,f'{context}_inner_{fold}')
                probs[name][iv] = base_probabilities(model,x[train[iv]])
        for name,values in probs.items():
            inner_frames.append(pd.DataFrame({'context':context,'source':name,
                'beer':np.repeat(ids.beer.iloc[train].to_numpy(),len(targets)), 'target':np.tile(targets,len(train)),
                **{f'p{k}':values[:,:,k].ravel() for k in range(7)}}))
        chosen, rows = select_point(probs,q[train],d['parents'],protocol)
        point_choices.extend(dict(context=context,**row) for row in rows)
        print(f'{context}: point={chosen}, feasible={rows[0]["selection_had_feasible_candidate"]}',flush=True)
        # Full-data fallback must reproduce the original R2 forest, including its seed.
        final_seed = base_protocol['model_seed'] + (999 if full else number*100+99)
        models = {name:fit(name,train,final_seed,f'{context}_final') for name in protocol['base_models']}
        risk, rows = fit_risk(models,probs,q[train],protocol)
        risk_choices.extend(dict(context=context,target=targets[row.pop('target_index')],**row) for row in rows)
        cal_log.extend(calibration_records(risk,context,targets))
        if full:
            full_models,full_risk,full_choice = models,risk,chosen
            break
        folds[test] = number
        for name,model in models.items():
            p = base_probabilities(model,x[test]);base_oof[name][test] = p
            for cost,weights in protocol['cost_weights'].items():
                point_oof[name+'__'+cost][test] = cost_decision(p,weights,d['parents'])[0]
        point_oof['inner_selected'][test] = point_oof[chosen][test]
        risk_oof['training_prior'][test] = risk.prior
        for j,source in enumerate(risk.choices):
            sources_oof[test,j] = source
            risk_oof['selected_uncalibrated'][test,j] = risk.prior[j] if source=='prior' else base_oof[source][test,j]
        risk_oof['selected_calibrated'][test] = risk.predict_distribution(x[test])
        joblib.dump(dict(models=models,risk=risk,point_choice=chosen),output/'models'/f'outer{number}.joblib',compress=3)

    point_rows = [dict(candidate=name,**tail_metrics(q,grades)) for name,grades in point_oof.items()]
    baseline = rating_loss(q,point_oof['extra_trees__median']).mean(axis=1)
    delta = rating_loss(q,point_oof['inner_selected']).mean(axis=1)-baseline
    mean,low,high = paired_intervals(delta,styles,protocol,[.025,.975])
    selected_metrics = tail_metrics(q,point_oof['inner_selected'])
    point_qualified = feasible(selected_metrics,float(baseline.mean()),protocol['point_limits']) and high <= protocol['point_limits']['maximum_raw_mae_increase']
    all_risks = dict(training_prior=risk_oof['training_prior'],extra_trees_uncalibrated=base_oof['extra_trees'],
        raw_ordinal_uncalibrated=base_oof['raw_ordinal'],selected_uncalibrated=risk_oof['selected_uncalibrated'],
        selected_calibrated=risk_oof['selected_calibrated'])
    metrics,curves = risk_tables(all_risks,q,targets,protocol)
    risk_status = ['exploratory_not_safety_validation']*len(targets)
    defect_rows = []
    for target in protocol['defect_targets']:
        j=targets.index(target);truth=1-q[:,j,0]
        improvement=binary_loss(1-risk_oof['training_prior'][:,j,0],truth)-binary_loss(1-risk_oof['selected_calibrated'][:,j,0],truth)
        gain,lower,upper=paired_intervals(improvement,styles,protocol,protocol['defect_interval_quantiles'])
        status='historical_nonzero_probability_support_not_safety' if lower>0 else 'insufficient_heldout_probability_support'
        risk_status[j]=status
        defect_rows.append(dict(target=target,brier_improvement=gain,lower99=lower,upper99=upper,
            status=status,safety_certified=False,interval_scope='fixed_OOF_beer_loss_bootstrap'))
    save_csv(output,'point_comparison.csv',point_rows)
    save_csv(output,'point_inner_selection.csv',point_choices);save_csv(output,'risk_inner_selection.csv',risk_choices)
    save_csv(output,'risk_metrics.csv',metrics);save_csv(output,'risk_reliability.csv',curves)
    save_csv(output,'defect_risk_validation.csv',defect_rows)
    save_csv(output,'calibration_parameters.csv',cal_log);save_csv(output,'inner_splits.csv',split_log)
    save_csv(output,'fit_log.csv',fit_log);save_csv(output,'outer_folds.csv',pd.DataFrame({'beer':ids.beer,'outer_fold':folds}))
    export_probabilities(output,'oof_base_probabilities.csv.gz',base_oof,ids.beer.to_numpy(),targets)
    export_probabilities(output,'oof_risk_probabilities.csv.gz',risk_oof,ids.beer.to_numpy(),targets)
    pd.concat(inner_frames,ignore_index=True).to_csv(output/'inner_probabilities.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    point_frames=[pd.DataFrame({'candidate':name,'beer':np.repeat(ids.beer.to_numpy(),len(targets)),
        'target':np.tile(targets,len(x)),'predicted_grade':g.ravel()}) for name,g in point_oof.items()]
    pd.concat(point_frames).to_csv(output/'oof_point_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    # Risk descriptions cannot be interpreted as chemical-threshold or safety certificates.
    p=risk_oof['selected_calibrated'];s=survival(p)
    risk_long=pd.DataFrame({'beer':np.repeat(ids.beer.to_numpy(),len(targets)),'target':np.tile(targets,len(x)),
        'outer_fold':np.repeat(folds,len(targets)), 'probability_rating_ge1':s[:,:,0].ravel(),
        'probability_rating_ge5':s[:,:,4].ravel(),'probability_rating_6':s[:,:,5].ravel(),
        'risk_source':sources_oof.ravel(),'risk_validation_status':np.tile(risk_status,len(x)),
        'safety_certified':False})
    before = survival(risk_oof['selected_uncalibrated'])
    for name, k in [('ge1', 0), ('ge5', 4), ('6', 5)]:
        risk_long['uncalibrated_probability_rating_'+name] = before[:, :, k].ravel()
    risk_long['calibration_quality_status'] = 'fitted_method_not_assumed_to_improve_heldout_calibration'
    save_csv(output,'oof_rating_risks.csv',risk_long)
    deployed_choice=full_choice if point_qualified else 'extra_trees__median'
    deployed_source,deployed_cost=deployed_choice.split('__')
    point=TailPointModel(full_models[deployed_source],protocol['cost_weights'][deployed_cost],d['parents'])
    semantics='integer_balanced_panel_conditional_median_with_parent_cap' if deployed_cost=='median' else 'integer_tail_cost_decision_not_population_median'
    bundle=dict(bundle_kind='r2b_raw_panel',raw_panel_model=point,risk_panel_model=full_risk,risk_status=risk_status,
        selected_model=deployed_choice,features=d['features'],targets=targets,primary=d['primary'],
        scorecard_scope=d['scope'].to_dict('records'),protocol=base_protocol,r2b_protocol=protocol,
        point_semantics=semantics,score_status='raw_panel_balanced_median_prediction' if deployed_cost=='median' else 'tail_cost_grade_prediction',
        raw_source_sha256=d['manifest']['sha256'],qualified_on_historical_grouped_folds=True,
        tail_point_qualified_on_historical_folds=bool(point_qualified),
        point_deployment_status='tail_qualified' if point_qualified else 'provisional_baseline_tail_not_qualified')
    joblib.dump(bundle,output/'model.joblib',compress=3)
    frame=pd.read_csv(args.source/'1.csv',dtype=str)
    long,primary,wide,quality=predict_raw_frame(frame,bundle)
    risks=risk_frame(frame,bundle,long.beer.drop_duplicates().to_numpy())
    long=long.merge(risks,on=['beer','target'],how='left',validate='one_to_one',sort=False)
    long['safety_certified']=False
    save_csv(output,'full_fit_scores_all52.csv',long);save_csv(output,'full_fit_grades_all52.csv',wide)
    save_csv(output,'sample_quality.csv',quality)
    deployed_oof=point_oof['inner_selected'] if point_qualified else point_oof['extra_trees__median']
    complete=pd.DataFrame({'beer':np.repeat(ids.beer.to_numpy(),len(targets)),'target':np.tile(targets,len(x)),
        'estimated_grade_0_6':deployed_oof.ravel(),'score_status':'OOF_'+bundle['point_deployment_status']})
    extras=pd.DataFrame([dict(beer=int(b),target=t,estimated_grade_0_6=pd.NA,score_status='no_raw_panel_label') for b in ids.beer for t in base_protocol['unlabelled_card_targets']])
    complete=pd.concat([complete,extras],ignore_index=True).merge(risk_long.drop(columns='outer_fold'),on=['beer','target'],how='left',validate='one_to_one')
    complete['estimated_grade_0_6'] = complete.estimated_grade_0_6.astype('Int64')
    complete['safety_certified']=False
    save_csv(output,'oof_scores_all52.csv',complete)
    wide=complete.pivot(index='beer',columns='target',values='estimated_grade_0_6').loc[ids.beer,d['scope'].target_id].reset_index()
    save_csv(output,'estimated_grades_all52.csv',wide)
    save_csv(output,'estimated_grades_primary47.csv',wide[['beer',*d['scope'].loc[d['scope'].acceptance_scope.eq('main_content'),'target_id']]])
    save_csv(output,'point_uncertainty.csv',[dict(mean_raw_mae_increase=mean,lower95=low,upper95=high,qualified=bool(point_qualified))])
    summary=dict(status='completed',baseline_commit=protocol['baseline_commit'],protocol_content_sha256=content_hash(protocol),
        raw_source_sha256=d['manifest']['sha256'],source_sha256={n:sha256(args.source/n) for n in ['1.csv','4.csv','8.xlsx']},
        n_beers=len(x),n_targets=len(targets),n_raw_ratings=len(d['raw']),base_model_fits=len(fit_log),
        full_training_point_choice=full_choice,deployed_point_choice=deployed_choice,
        tail_point_qualified=bool(point_qualified),selected_point_metrics=selected_metrics,
        selected_raw_mae_increase=mean,increase_upper95=high,risk_status={t:risk_status[targets.index(t)] for t in protocol['defect_targets']},
        grade_semantics=semantics,model_sha256=sha256(output/'model.joblib'),
        safety_certified=False,unlabelled_targets=base_protocol['unlabelled_card_targets'],task_R3_started=False)
    write_json(output/'summary.json',summary)
    plot_results(output)
    print(pd.DataFrame(point_rows).to_string(index=False),flush=True)
    print(pd.DataFrame(defect_rows).to_string(index=False),flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points=pd.read_csv(output/'point_comparison.csv');risk=pd.read_csv(output/'risk_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    baseline=points.loc[points.candidate.eq('extra_trees__median')].iloc[0]
    selected=points.loc[points.candidate.eq('inner_selected')].iloc[0]
    for _,r in points.loc[~points.candidate.eq('inner_selected')].iterrows():
        label=r.candidate
        if label=='extra_trees__median' and np.isclose(r.raw_rating_mae,selected.raw_rating_mae):
            label+=' (= inner_selected)'
        axes[0].scatter(r.raw_rating_mae,(r.raw_recall_5+r.raw_recall_6)/2,s=45,label=label)
    axes[0].axvline(baseline.raw_rating_mae+.02,color='grey',linestyle=':',label='MAE limit (+0.02)')
    axes[0].set(xlabel='Original-rating MAE',ylabel='Mean exact recall of grades 5 and 6',title='Tail recall versus average error')
    axes[0].legend(fontsize=7,loc='upper left',bbox_to_anchor=(0,-.15))
    names=['diacetyl','dms','X4vg','t2n','stale_hops']
    for method,shift in [('training_prior',-.18),('selected_uncalibrated',0),('selected_calibrated',.18)]:
        values=risk[risk.method.eq(method)&risk.rating_ge.eq(1)].set_index('target').loc[names,'brier']
        axes[1].bar(np.arange(5)+shift,values,width=.18,label=method)
    axes[1].set(xticks=np.arange(5),xticklabels=names,ylabel='Expected binary Brier loss',title='Nonzero-rating probability; not safety')
    axes[1].legend(fontsize=8)
    fig.savefig(output/'tail_risk_diagnostics.png',dpi=160);fig.savefig(output/'tail_risk_diagnostics.svg')
    svg=output/'tail_risk_diagnostics.svg';svg.write_text('\n'.join(v.rstrip() for v in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--raw',type=Path,default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    p.add_argument('--output',type=Path,required=True)
    with threadpool_limits(limits=1):run(p.parse_args())
