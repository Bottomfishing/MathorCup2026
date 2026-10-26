"""Task5: three frozen repeated nested beer-level validation runs."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import write_json,sha256
from beer_models.baseline import make_splits
from beer_models.raw_panel import median_grades,enforce_hierarchy,rating_loss
from beer_models.raw_explain import fit_raw_ridge,explain_quantile,check_additivity,decision,nonzero_top
from beer_models.tail_risk import RawOrdinalModel,base_probabilities,fit_risk,survival,binary_loss,tail_metrics
from beer_models.generation import StyleGenerator
from beer_models.q3_raw import ShrinkageCopula,spectrum_diagnostics,raw_predictions,defect_guard
from beer_models.style_models import make_style_model
from beer_models.repeated_validation import beer_averaged_interval,repeat_summary,rank_tables
from r2_raw_q1 import load_inputs,content_hash,RAW_MANIFEST,PROTOCOL as BASE_PROTOCOL,summary_metrics
from r2b_tail_risk import PROTOCOL as RISK_PROTOCOL,risk_tables,calibration_records
from r3_raw_explanation import PROTOCOL as EXPLAIN_PROTOCOL,ELIGIBILITY,fit_gbr,select_gbr
from task4_raw_q3 import PROTOCOL as GENERATION_PROTOCOL,LEGACY,STYLE_CONFIG,read,text_hash,save_tables,sensory_validation,generation_gate

PROTOCOL=ROOT/'configs/task5_repeated_nested_protocol_20261026.json'


def fit_model(data,train,method,seed,bp,rp,ep):
    if method=='raw_ordinal':return RawOrdinalModel(rp).fit(data['x'][train],data['distribution'][train])
    if method=='ridge_raw_mean':return fit_raw_ridge(data['x'][train],data['distribution'][train],ep['ridge_alpha'],seed)
    return fit_gbr(data,train,method,seed,bp)


def grade_from_probability(probability,parents):return enforce_hierarchy(median_grades(probability),parents)


def generation_comparison(data,train,test,point_model,gp,seed):
    x,styles=data['x'],data['ids'].tasting_category_fine.to_numpy()
    a=StyleGenerator().fit(x[train],styles[train])
    b=ShrinkageCopula().fit(x[train],styles[train],requested_styles=gp['styles'])
    real_probability=base_probabilities(point_model,x[test]);real_grade=grade_from_probability(real_probability,data['parents'])
    metrics=[];sensory=[];samples={}
    for si,style in enumerate(gp['styles']):
        it=np.flatnonzero(styles[train]==style);iv=np.flatnonzero(styles[test]==style)
        ztrain=a.robust_.transform(a.x_[it]);ztest=a.robust_.transform(a.impute_.transform(x[test[iv]]))
        for method in gp['generation_methods']:
            values=a.sample(style,gp['generation_count'],'interpolate',seed+si)[0] if method=='interpolate' else b.sample(style,gp['generation_count'],seed+si)
            row=dict(style=style,method=method,seed=seed+si,**spectrum_diagnostics(ztrain,ztest,a.robust_.transform(values),gp))
            probability=base_probabilities(point_model,values);grade=grade_from_probability(probability,data['parents'])
            rows=sensory_validation(data['distribution'][test[iv]],real_probability[iv],real_grade[iv],probability,grade,data['targets'])
            for field in ['model_median_w1','observed_median_w1','model_rating_distribution_w1','observed_rating_distribution_w1']:
                row[field]=float(np.mean([r[field] for r in rows]))
            metrics.append(row);sensory.extend(dict(style=style,method=method,**r) for r in rows)
            samples[(method,style)]=values
    return pd.DataFrame(metrics),pd.DataFrame(sensory),samples


def fixed_path_sensitivity(data,test,models,risk,repeat,fold,gp):
    before=read(LEGACY/'transfer_source_profiles.csv',dtype={'beer':str}).set_index('beer')
    after=read(LEGACY/'transfer_profiles.csv',dtype={'beer':str}).set_index('beer')
    held=set(data['ids'].beer.iloc[test].astype(str));beers=[beer for beer in before.index if beer in held]
    if not beers:return []
    bundle=dict(raw_panel_model=models['extra_trees'],risk_panel_model=risk)
    a=raw_predictions(bundle,before.loc[beers,data['features']].to_numpy(float),data['parents'])
    b=raw_predictions(bundle,after.loc[beers,data['features']].to_numpy(float),data['parents'])
    indices=[data['targets'].index(t) for t in gp['defect_targets']]
    joint,deltas,_=defect_guard(a,b,indices,gp['nonincrease_tolerance']);rows=[]
    for i,beer in enumerate(beers):
        for j,target in enumerate(gp['defect_targets']):
            row=dict(repeat=repeat,outer_fold=fold,beer=beer,target=target,source_held_out=True,
                grade_before=int(a['grade'][i,indices[j]]),grade_after=int(b['grade'][i,indices[j]]),
                grade_nonincrease=bool(deltas['grade'][i,j]<=0),all_five_sensory_guards_pass=bool(joint[i]),
                scope='fixed_historical_prepared_path_sensitivity_not_nested_migration_validation',
                actual_after_measured=False,safety_certified=False)
            for version in ['calibrated','uncalibrated']:
                for k,event in enumerate(['ge1','ge5','6']):row[version+'_probability_delta_'+event]=float(deltas[version][i,j,k])
            rows.append(row)
    return rows


def derive_tables(data,p,rp,ep,folds,grades,probabilities,shap_values,coefficients,selected_methods,generation_outer):
    q=data['distribution'];targets=data['targets'];styles=data['ids'].tasting_category_fine.to_numpy()
    metric_rows=[];target_rows=[];risk_rows=[];reliability=[];long=[];loss_rows=[]
    for repeat,seed in enumerate(p['repeat_outer_seeds']):
        for method,matrix in grades.items():
            pred=matrix[repeat];loss=rating_loss(q,pred)
            metric_rows.append(dict(repeat=repeat,outer_seed=seed,method=method,**(summary_metrics(q,pred)|tail_metrics(q,pred))))
            for j,target in enumerate(targets):
                target_rows.append(dict(repeat=repeat,method=method,target=target,raw_rating_mae=float(loss[:,j].mean()),
                    panel_median_mae=float(abs(pred[:,j]-median_grades(q)[:,j]).mean())))
            loss_rows.extend(dict(repeat=repeat,method=method,beer=int(beer),raw_rating_mae=float(value)) for beer,value in zip(data['ids'].beer,loss.mean(1)))
        pm={name:values[repeat] for name,values in probabilities.items()}
        metrics,curves=risk_tables(pm,q,targets,rp)
        risk_rows.extend(metrics.assign(repeat=repeat).to_dict('records'));reliability.extend(curves.assign(repeat=repeat).to_dict('records'))
        cal=survival(pm['selected_calibrated']);uncal=survival(pm['selected_uncalibrated'])
        for i,beer in enumerate(data['ids'].beer):
            for target in data['scope'].target_id:
                row=dict(repeat=repeat,beer=int(beer),outer_fold=int(folds[repeat,i]),target=target,
                    estimated_grade_0_6=None,score_status='no_raw_panel_label',safety_certified=False)
                if target in targets:
                    j=targets.index(target);row.update(estimated_grade_0_6=int(grades['extra_trees'][repeat,i,j]),score_status='repeated_OOF_raw_median')
                    for event,k in [('ge1',0),('ge5',4),('6',5)]:
                        row['probability_rating_'+event]=float(cal[i,j,k]);row['uncalibrated_probability_rating_'+event]=float(uncal[i,j,k])
                long.append(row)
    all_metrics=pd.DataFrame(metric_rows);target_metrics=pd.DataFrame(target_rows);risk_metrics=pd.DataFrame(risk_rows)
    intervals=[]
    baseline=np.stack([rating_loss(q,g) for g in grades['training_median']])
    for method in ['extra_trees','gbr_selected','ridge_raw_mean']:
        difference=baseline-np.stack([rating_loss(q,g) for g in grades[method]])
        gain,lo,hi=beer_averaged_interval(difference.mean(2),styles,p['bootstrap_repeats'],p['bootstrap_seed'],[.025,.975])
        intervals.append(dict(metric='raw_MAE_gain',method=method,target='all50',mean_gain=float(gain),lower=float(lo),upper=float(hi),level=.95,
            n_independent_beer_units=len(q),n_repeated_predictions=len(q)*len(folds),scope='beer_averaged_fixed_OOF_bootstrap_not_full_refit_CI'))
    truth=1-q[:,:,0]
    for target in rp['defect_targets']:
        j=targets.index(target)
        for method in ['extra_trees_uncalibrated','selected_uncalibrated','selected_calibrated']:
            prior=1-probabilities['training_prior'][:,:,j,0];prediction=1-probabilities[method][:,:,j,0]
            delta=binary_loss(prior,truth[:,j])-binary_loss(prediction,truth[:,j])
            gain,lo,hi=beer_averaged_interval(delta,styles,p['bootstrap_repeats'],p['bootstrap_seed'],[.005,.995])
            intervals.append(dict(metric='nonzero_Brier_gain',method=method,target=target,mean_gain=float(gain),lower=float(lo),upper=float(hi),level=.99,
                n_independent_beer_units=len(q),n_repeated_predictions=len(q)*len(folds),scope='beer_averaged_fixed_OOF_bootstrap_not_intervention_CI'))
    eligible_config=json.loads(ELIGIBILITY.read_text(encoding='utf-8'))
    excluded=eligible_config['exclude_from_compound_top3'];features=data['features']
    eligible=np.array([i for i,f in enumerate(features) if f not in excluded]);family=np.array([i for i,f in enumerate(features) if f not in excluded or f=='bitterness'])
    assert len(eligible)==219 and len(family)==220
    tables=rank_tables(shap_values,coefficients,folds,features,targets,data['scope'],eligible,family)
    long=pd.DataFrame(long);long['estimated_grade_0_6']=long.estimated_grade_0_6.astype('Int64')
    tables.update({'q1_metrics_by_repeat.csv':all_metrics,'predictive_metrics_by_target.csv':target_metrics,
        'q1_repeat_summary.csv':repeat_summary(all_metrics,['method'],['raw_rating_mae','raw_rating_exact','panel_median_mae','raw_recall_5','raw_recall_6']),
        'risk_metrics_by_repeat.csv':risk_metrics,'risk_reliability.csv':pd.DataFrame(reliability),
        'risk_repeat_summary.csv':repeat_summary(risk_metrics,['method','target','rating_ge'],['brier','log_loss','calibration_ece']),
        'paired_beer_averaged_intervals.csv':pd.DataFrame(intervals),'q1_oof_scores52.csv.gz':long,
        'loss_by_repeat_beer.csv':pd.DataFrame(loss_rows)})
    generation_outer=pd.DataFrame(generation_outer)
    gen_metrics=['marginal_w1','energy','variance_ratio','rank_correlation_rmse','heldout_coverage','internal_near_duplicate_rate',
        'model_median_w1','observed_median_w1']
    repeated=generation_outer.groupby(['repeat','method'],sort=False)[gen_metrics].mean().reset_index()
    selected=[]
    for repeat in range(len(folds)):
        for fold in range(p['outer_folds']):
            choice=selected_methods[(repeat,fold)]
            selected.extend(generation_outer[(generation_outer.repeat==repeat)&(generation_outer.outer_fold==fold)&generation_outer.method.eq(choice)].assign(method='inner_selected_workflow').to_dict('records'))
    selected=pd.DataFrame(selected)
    selected_repeated=selected.groupby(['repeat','method'],sort=False)[gen_metrics].mean().reset_index()
    repeated=pd.concat([repeated,selected_repeated],ignore_index=True)
    tables.update({'generation_selected_outer_validation.csv':selected,'generation_metrics_by_repeat.csv':repeated,
        'generation_repeat_summary.csv':repeat_summary(repeated,['method'],gen_metrics)})
    return tables


def run(args):
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'));bp=json.loads(BASE_PROTOCOL.read_text(encoding='utf-8'))
    rp=json.loads(RISK_PROTOCOL.read_text(encoding='utf-8'));ep=json.loads(EXPLAIN_PROTOCOL.read_text(encoding='utf-8'))
    gp=json.loads(GENERATION_PROTOCOL.read_text(encoding='utf-8'));sc=json.loads(STYLE_CONFIG.read_text(encoding='utf-8'))
    data=load_inputs(args.source,args.raw,bp);x,q,ids,targets=data['x'],data['distribution'],data['ids'],data['targets']
    styles=ids.tasting_category_fine.to_numpy();r=len(p['repeat_outer_seeds']);n,t,f=len(x),len(targets),x.shape[1]
    out=args.output;out.mkdir(parents=True,exist_ok=False);(out/'models').mkdir();(out/'inner').mkdir()
    write_json(out/'protocol.json',p)
    grades={name:np.empty((r,n,t),int) for name in ['training_median','extra_trees','gbr_selected','ridge_raw_mean']}
    probabilities={name:np.empty((r,n,t,7)) for name in ['training_prior','extra_trees_uncalibrated','raw_ordinal_uncalibrated','selected_uncalibrated','selected_calibrated']}
    shap_values=np.empty((r,n,t,f));shap_base=np.empty((r,n,t));shap_output=np.empty((r,n,t))
    coefficients=np.empty((r,p['outer_folds'],t,f));folds=np.empty((r,n),int)
    fit_log=[];splits_log=[];backgrounds=[];gbr_choices=[];risk_choices=[];calibration=[];gen_inner=[];gen_gates=[]
    gen_outer=[];gen_sensory=[];selected_gen={};gen_spectra={};style_oof=[];fixed_paths=[];additivity=[]
    def fit(train,method,seed,context):
        tick=time.perf_counter();model=fit_model(data,train,method,seed,bp,rp,ep)
        imputer=model.named_steps['impute'] if method=='ridge_raw_mean' else model.imputer
        np.testing.assert_allclose(imputer.statistics_,np.nanmedian(x[train],axis=0),atol=0,rtol=0)
        fit_log.append(dict(context=context,method=method,seed=seed,n_training_beers=len(train)))
        return model
    for repeat,outer_seed in enumerate(p['repeat_outer_seeds']):
        for fold,(train,test) in enumerate(make_splits(styles,p['outer_folds'],outer_seed)):
            context=f'r{repeat}_f{fold}';print(f'{context}: nested training on {len(train)} beers, test {len(test)}',flush=True)
            folds[repeat,test]=fold
            for role,index in [('outer_train',train),('outer_test',test)]:
                splits_log.extend(dict(repeat=repeat,outer_fold=fold,inner_fold=-1,role=role,beer=int(ids.beer.iloc[i])) for i in index)
            inner_p={name:np.empty((len(train),t,7)) for name in ['prior','extra_trees','raw_ordinal']}
            inner_g={name:np.empty((len(train),t),int) for name in ep['gbr_candidates']};inner_fold=np.empty(len(train),int)
            gen_local=[]
            for number,(it,iv) in enumerate(make_splits(styles[train],p['inner_folds'],outer_seed+100+fold)):
                inner_fold[iv]=number
                for role,index in [('inner_train',train[it]),('inner_validation',train[iv])]:
                    splits_log.extend(dict(repeat=repeat,outer_fold=fold,inner_fold=number,role=role,beer=int(ids.beer.iloc[i])) for i in index)
                seed=p['seed']+repeat*1000+fold*100+number
                models={name:fit(train[it],name,seed,f'{context}_inner{number}') for name in ['extra_trees','raw_ordinal',*ep['gbr_candidates']]}
                inner_p['prior'][iv]=q[train[it]].mean(0)
                for name in ['extra_trees','raw_ordinal']:inner_p[name][iv]=base_probabilities(models[name],x[train[iv]])
                for name in ep['gbr_candidates']:inner_g[name][iv]=models[name].predict(x[train[iv]])['grade']
                gm,_,_=generation_comparison(data,train[it],train[iv],models['extra_trees'],gp,p['seed']+repeat*10000+fold*1000+(number+1)*100)
                gm=gm.assign(repeat=repeat,outer_fold=fold,inner_fold=number);gen_inner.extend(gm.to_dict('records'));gen_local.extend(gm.to_dict('records'))
                del models
            chosen,losses=select_gbr(inner_g,q[train],ep['gbr_candidates'])
            gbr_choices.extend(dict(repeat=repeat,outer_fold=fold,candidate=name,inner_raw_mae=value,selected=name==chosen) for name,value in losses.items())
            generation_choice,gates=generation_gate(pd.DataFrame(gen_local));selected_gen[(repeat,fold)]=generation_choice
            gen_gates.extend(gates.assign(repeat=repeat,outer_fold=fold,selected_generation_method=generation_choice).to_dict('records'))
            seed=p['seed']+repeat*1000+fold*100+99
            models={name:fit(train,name,seed,f'{context}_outer') for name in ['extra_trees','raw_ordinal',chosen,'ridge_raw_mean']}
            risk,choices=fit_risk({name:models[name] for name in ['extra_trees','raw_ordinal']},inner_p,q[train],rp)
            risk_choices.extend(dict(repeat=repeat,outer_fold=fold,target=targets[row.pop('target_index')],**row) for row in choices)
            calibration.extend(calibration_records(risk,context,targets))
            prior=np.broadcast_to(q[train].mean(0),(len(test),t,7))
            grades['training_median'][repeat,test]=grade_from_probability(prior,data['parents'])
            et=base_probabilities(models['extra_trees'],x[test]);ordinal=base_probabilities(models['raw_ordinal'],x[test])
            grades['extra_trees'][repeat,test]=grade_from_probability(et,data['parents'])
            grades['gbr_selected'][repeat,test]=models[chosen].predict(x[test])['grade']
            grades['ridge_raw_mean'][repeat,test]=decision(models['ridge_raw_mean'].predict(x[test]),data['parents'])
            for name,value in [('training_prior',prior),('extra_trees_uncalibrated',et),('raw_ordinal_uncalibrated',ordinal),
                ('selected_uncalibrated',risk.predict_uncalibrated_distribution(x[test])),('selected_calibrated',risk.predict_distribution(x[test]))]:probabilities[name][repeat,test]=value
            bg=np.sort(np.random.default_rng(p['seed']+repeat*10000+fold*1000+500).choice(train,min(ep['background_size'],len(train)),replace=False))
            backgrounds.extend(dict(repeat=repeat,outer_fold=fold,beer=int(ids.beer.iloc[i])) for i in bg)
            v,b,c=explain_quantile(models[chosen],x[test],x[bg]);error=check_additivity(v,b,c)
            shap_values[repeat,test]=v;shap_base[repeat,test]=b;shap_output[repeat,test]=c
            coefficients[repeat,fold]=models['ridge_raw_mean'].named_steps['model'].coef_
            additivity.append(dict(repeat=repeat,outer_fold=fold,max_error=error))
            gm,gs,spectra=generation_comparison(data,train,test,models['extra_trees'],gp,p['seed']+repeat*10000+fold*1000+900)
            gen_outer.extend(gm.assign(repeat=repeat,outer_fold=fold,inner_selected_method=generation_choice).to_dict('records'))
            gen_sensory.extend(gs.assign(repeat=repeat,outer_fold=fold).to_dict('records'))
            for (method,style),values in spectra.items():gen_spectra[f'{context}_{method}_{style}']=values
            classifiers={}
            for name in ['logistic','extra_trees']:
                tick=time.perf_counter();classifier=make_style_model(name,sc,seed).fit(x[train],styles[train]);classifiers[name]=classifier
                fit_log.append(dict(context=context+'_style',method=name,seed=seed,n_training_beers=len(train)))
                prediction=classifier.predict(x[test]);style_oof.extend(dict(repeat=repeat,outer_fold=fold,beer=int(ids.beer.iloc[i]),model=name,
                    observed_style=styles[i],predicted_style=value,correct=bool(value==styles[i])) for i,value in zip(test,prediction))
            fixed_paths.extend(fixed_path_sensitivity(data,test,models,risk,repeat,fold,gp))
            joblib.dump(dict(models=models,risk=risk,style_models=classifiers,train=train,test=test,background=bg,
                gbr_choice=chosen,generation_choice=generation_choice),out/'models'/(context+'.joblib'),compress=3)
            np.savez_compressed(out/'inner'/(context+'.npz'),**inner_p,**inner_g,beer=ids.beer.iloc[train].to_numpy(),inner_fold=inner_fold)
            print(f'{context}: GBR={chosen}, inner generator={generation_choice}, SHAP max error={error:.2g}',flush=True)
            del models,risk,classifiers
    tables=derive_tables(data,p,rp,ep,folds,grades,probabilities,shap_values,coefficients,selected_gen,gen_outer)
    tables.update({'model_fit_log.csv':fit_log,'grouped_splits.csv':splits_log,'background_beers.csv':backgrounds,
        'gbr_inner_selection.csv':gbr_choices,'risk_inner_selection.csv':risk_choices,'calibration_parameters.csv':calibration,
        'generation_inner_validation.csv':gen_inner,'generation_inner_gate_checks.csv':gen_gates,
        'generation_outer_validation.csv':gen_outer,'generation_outer_sensory_by_target.csv.gz':gen_sensory,
        'style_oof.csv':style_oof,'fixed_historical_path_sensitivity.csv':fixed_paths,'shap_additivity.csv':additivity})
    paths=pd.DataFrame(fixed_paths)
    tables['fixed_historical_path_joint.csv']=paths.groupby(['repeat','beer'],sort=False).agg(
        source_held_out=('source_held_out','all'),all_five_sensory_guards_pass=('all_five_sensory_guards_pass','first')).reset_index()
    save_tables(out,tables)
    np.savez_compressed(out/'oof_arrays.npz',**{f'grade_{name}':value for name,value in grades.items()},
        **{f'probability_{name}':value for name,value in probabilities.items()},folds=folds,coefficients=coefficients,
        beer=ids.beer.to_numpy(),targets=np.array(targets),features=np.array(data['features']))
    np.savez_compressed(out/'gbr_OOF_SHAP.npz',values=shap_values,base=shap_base,output=shap_output)
    np.savez_compressed(out/'outer_generated_spectra.npz',**gen_spectra)
    sources=[BASE_PROTOCOL,RISK_PROTOCOL,EXPLAIN_PROTOCOL,GENERATION_PROTOCOL,STYLE_CONFIG,ELIGIBILITY,
        LEGACY/'transfer_source_profiles.csv',LEGACY/'transfer_profiles.csv']
    summary=dict(status='completed',baseline_commit=p['baseline_commit'],protocol_content_sha256=content_hash(p),
        source_sha256={name:sha256(args.source/name) for name in ['1.csv','4.csv','8.xlsx']},raw_source_sha256=data['manifest']['sha256'],
        input_lf_normalized_sha256={path.relative_to(ROOT).as_posix():text_hash(path) for path in sources},
        packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','shap']},
        repeats=r,outer_folds_per_repeat=p['outer_folds'],unique_beers=n,repeated_beer_predictions=r*n,
        model_fits=len(fit_log),generator_fits=2*(p['inner_folds']+1)*r*p['outer_folds'],
        q1_raw_mae_by_repeat=tables['q1_metrics_by_repeat.csv'].query("method == 'extra_trees'").raw_rating_mae.tolist(),
        GBR_choices=[row['candidate'] for row in gbr_choices if row['selected']],
        nested_generator_choices=[selected_gen[(rep,fold)] for rep in range(r) for fold in range(p['outer_folds'])],
        max_SHAP_additivity_error=max(row['max_error'] for row in additivity),
        generated_outer_spectra=r*p['outer_folds']*2*3*p['generation_count'],
        fixed_historical_path_repeated_cases=len(tables['fixed_historical_path_joint.csv']),
        deployed_Q1_changed=False,deployed_Q2_changed=False,deployed_Q3_changed=False,
        task4_zero_candidate_result_preserved=True,safety_certified=False,actual_intervention_validated=False,task6_started=False,
        array_sha256={name:sha256(out/name) for name in ['oof_arrays.npz','gbr_OOF_SHAP.npz','outer_generated_spectra.npz']})
    write_json(out/'summary.json',summary);plot_results(out)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=read(output/'q1_metrics_by_repeat.csv');generation=read(output/'generation_metrics_by_repeat.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for method in ['training_median','extra_trees','gbr_selected','ridge_raw_mean']:
        g=metrics[metrics.method.eq(method)];axes[0].plot(g.repeat,g.raw_rating_mae,'o-',label=method)
    fixed=generation[generation.method.eq('interpolate')].sort_values('repeat')
    selected=generation[generation.method.eq('inner_selected_workflow')].sort_values('repeat')
    same=np.array_equal(fixed.energy.to_numpy(),selected.energy.to_numpy())
    for method in (['interpolate','shrinkage_copula'] if same else ['interpolate','shrinkage_copula','inner_selected_workflow']):
        g=generation[generation.method.eq(method)]
        label='interpolate (= inner-selected)' if method=='interpolate' and same else method
        axes[1].plot(g.repeat,g.energy,'o-',label=label)
    axes[0].set(title='Repeated grouped OOF raw-rating MAE',ylabel='MAE')
    axes[1].set(title='Outer held-out generation energy',ylabel='Energy distance')
    for ax in axes:ax.set_xticks([0,1,2],['20261027','20261028','20261029']);ax.set_xlabel('Outer split seed');ax.legend(fontsize=8)
    fig.suptitle('Same 250 beers: repeated validation, not an external test')
    fig.savefig(output/'repeated_validation.png',dpi=160);fig.savefig(output/'repeated_validation.svg')
    svg=output/'repeated_validation.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8');plt.close(fig)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--raw',type=Path,default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    return p


if __name__=='__main__':
    with threadpool_limits(limits=1):run(parser().parse_args())
