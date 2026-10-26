"""Rebuild nested choices, refit all learners and replay Task5 held-out evidence."""
import io
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256,write_json
from beer_models.baseline import make_splits
from beer_models.raw_explain import explain_quantile,check_additivity,decision
from beer_models.tail_risk import fit_risk,base_probabilities
from beer_models.style_models import make_style_model
from r2_raw_q1 import load_inputs,content_hash
from r2b_tail_risk import calibration_records
from r3_raw_explanation import select_gbr
from task4_raw_q3 import generation_gate,read,text_hash
from task5_repeated_nested import (PROTOCOL,BASE_PROTOCOL,RISK_PROTOCOL,EXPLAIN_PROTOCOL,GENERATION_PROTOCOL,STYLE_CONFIG,
    fit_model,grade_from_probability,generation_comparison,fixed_path_sensitivity,derive_tables,parser)


def replay(args,refit_models=True):
    """The false branch recovers a completed training cache; CLI verification always refits."""
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'));bp=json.loads(BASE_PROTOCOL.read_text(encoding='utf-8'))
    rp=json.loads(RISK_PROTOCOL.read_text(encoding='utf-8'));ep=json.loads(EXPLAIN_PROTOCOL.read_text(encoding='utf-8'))
    gp=json.loads(GENERATION_PROTOCOL.read_text(encoding='utf-8'));sc=json.loads(STYLE_CONFIG.read_text(encoding='utf-8'))
    data=load_inputs(args.source,args.raw,bp);x,q,ids,targets=data['x'],data['distribution'],data['ids'],data['targets']
    styles=ids.tasting_category_fine.to_numpy();r=len(p['repeat_outer_seeds']);n,t,f=len(x),len(targets),x.shape[1]
    grades={name:np.empty((r,n,t),int) for name in ['training_median','extra_trees','gbr_selected','ridge_raw_mean']}
    probabilities={name:np.empty((r,n,t,7)) for name in ['training_prior','extra_trees_uncalibrated','raw_ordinal_uncalibrated','selected_uncalibrated','selected_calibrated']}
    shap_values=np.empty((r,n,t,f));shap_base=np.empty((r,n,t));shap_output=np.empty((r,n,t))
    coefficients=np.empty((r,p['outer_folds'],t,f));folds=np.empty((r,n),int)
    fit_log=[];splits_log=[];backgrounds=[];gbr_choices=[];risk_choices=[];calibration=[];gen_inner=[];gen_gates=[]
    gen_outer=[];gen_sensory=[];selected_gen={};gen_spectra={};style_oof=[];fixed_paths=[];additivity=[]
    def validate_fit(model,train,method):
        imputer=model.named_steps['impute'] if method=='ridge_raw_mean' else model.imputer
        np.testing.assert_allclose(imputer.statistics_,np.nanmedian(x[train],axis=0),atol=0,rtol=0)
    for repeat,outer_seed in enumerate(p['repeat_outer_seeds']):
        for fold,(train,test) in enumerate(make_splits(styles,p['outer_folds'],outer_seed)):
            context=f'r{repeat}_f{fold}';print(f'Replay {context}, independent model refits={refit_models}',flush=True)
            saved=joblib.load(args.output/'models'/(context+'.joblib'))
            np.testing.assert_array_equal(saved['train'],train);np.testing.assert_array_equal(saved['test'],test)
            assert np.all(np.isin(data['positions'],train)^np.isin(data['positions'],test))
            inner_saved=np.load(args.output/'inner'/(context+'.npz'),allow_pickle=False)
            np.testing.assert_array_equal(inner_saved['beer'],ids.beer.iloc[train])
            folds[repeat,test]=fold
            for role,index in [('outer_train',train),('outer_test',test)]:
                splits_log.extend(dict(repeat=repeat,outer_fold=fold,inner_fold=-1,role=role,beer=int(ids.beer.iloc[i])) for i in index)
            ip={name:np.empty((len(train),t,7)) for name in ['prior','extra_trees','raw_ordinal']}
            ig={name:np.empty((len(train),t),int) for name in ep['gbr_candidates']};gl=[]
            for number,(it,iv) in enumerate(make_splits(styles[train],p['inner_folds'],outer_seed+100+fold)):
                np.testing.assert_array_equal(inner_saved['inner_fold'][iv],np.full(len(iv),number))
                for role,index in [('inner_train',train[it]),('inner_validation',train[iv])]:
                    splits_log.extend(dict(repeat=repeat,outer_fold=fold,inner_fold=number,role=role,beer=int(ids.beer.iloc[i])) for i in index)
                seed=p['seed']+repeat*1000+fold*100+number;models={}
                for method in ['extra_trees','raw_ordinal',*ep['gbr_candidates']]:
                    fit_log.append(dict(context=f'{context}_inner{number}',method=method,seed=seed,n_training_beers=len(it)))
                    if refit_models or method=='extra_trees':
                        models[method]=fit_model(data,train[it],method,seed,bp,rp,ep);validate_fit(models[method],train[it],method)
                ip['prior'][iv]=q[train[it]].mean(0)
                for name in ['extra_trees','raw_ordinal']:
                    ip[name][iv]=base_probabilities(models[name],x[train[iv]]) if refit_models or name=='extra_trees' else inner_saved[name][iv]
                    np.testing.assert_allclose(ip[name][iv],inner_saved[name][iv],atol=1e-12,rtol=0)
                np.testing.assert_allclose(ip['prior'][iv],inner_saved['prior'][iv],atol=1e-12,rtol=0)
                for name in ep['gbr_candidates']:
                    ig[name][iv]=models[name].predict(x[train[iv]])['grade'] if refit_models else inner_saved[name][iv]
                    np.testing.assert_array_equal(ig[name][iv],inner_saved[name][iv])
                gm,_,_=generation_comparison(data,train[it],train[iv],models['extra_trees'],gp,p['seed']+repeat*10000+fold*1000+(number+1)*100)
                gm=gm.assign(repeat=repeat,outer_fold=fold,inner_fold=number);gen_inner.extend(gm.to_dict('records'));gl.extend(gm.to_dict('records'))
                del models
            chosen,losses=select_gbr(ig,q[train],ep['gbr_candidates']);assert chosen==saved['gbr_choice']
            gbr_choices.extend(dict(repeat=repeat,outer_fold=fold,candidate=name,inner_raw_mae=value,selected=name==chosen) for name,value in losses.items())
            generation_choice,gates=generation_gate(pd.DataFrame(gl));assert generation_choice==saved['generation_choice']
            selected_gen[(repeat,fold)]=generation_choice
            gen_gates.extend(gates.assign(repeat=repeat,outer_fold=fold,selected_generation_method=generation_choice).to_dict('records'))
            seed=p['seed']+repeat*1000+fold*100+99;models={}
            for name in ['extra_trees','raw_ordinal',chosen,'ridge_raw_mean']:
                fit_log.append(dict(context=context+'_outer',method=name,seed=seed,n_training_beers=len(train)))
                models[name]=fit_model(data,train,name,seed,bp,rp,ep) if refit_models else saved['models'][name]
                validate_fit(models[name],train,name)
                if name in ['extra_trees','raw_ordinal']:
                    np.testing.assert_allclose(base_probabilities(models[name],x[test]),base_probabilities(saved['models'][name],x[test]),atol=1e-12,rtol=0)
                elif name=='ridge_raw_mean':np.testing.assert_allclose(models[name].predict(x[test]),saved['models'][name].predict(x[test]),atol=1e-12,rtol=0)
                else:np.testing.assert_array_equal(models[name].predict(x[test])['grade'],saved['models'][name].predict(x[test])['grade'])
            risk,choices=fit_risk({name:models[name] for name in ['extra_trees','raw_ordinal']},ip,q[train],rp)
            assert risk.choices==saved['risk'].choices
            np.testing.assert_allclose(risk.predict_distribution(x[test]),saved['risk'].predict_distribution(x[test]),atol=1e-12,rtol=0)
            risk_choices.extend(dict(repeat=repeat,outer_fold=fold,target=targets[row.pop('target_index')],**row) for row in choices)
            calibration.extend(calibration_records(risk,context,targets))
            prior=np.broadcast_to(q[train].mean(0),(len(test),t,7));et=base_probabilities(models['extra_trees'],x[test]);ordinal=base_probabilities(models['raw_ordinal'],x[test])
            grades['training_median'][repeat,test]=grade_from_probability(prior,data['parents'])
            grades['extra_trees'][repeat,test]=grade_from_probability(et,data['parents'])
            grades['gbr_selected'][repeat,test]=models[chosen].predict(x[test])['grade']
            grades['ridge_raw_mean'][repeat,test]=decision(models['ridge_raw_mean'].predict(x[test]),data['parents'])
            for name,value in [('training_prior',prior),('extra_trees_uncalibrated',et),('raw_ordinal_uncalibrated',ordinal),
                ('selected_uncalibrated',risk.predict_uncalibrated_distribution(x[test])),('selected_calibrated',risk.predict_distribution(x[test]))]:probabilities[name][repeat,test]=value
            bg=np.sort(np.random.default_rng(p['seed']+repeat*10000+fold*1000+500).choice(train,min(ep['background_size'],len(train)),replace=False))
            np.testing.assert_array_equal(saved['background'],bg);assert not np.intersect1d(bg,test).size
            backgrounds.extend(dict(repeat=repeat,outer_fold=fold,beer=int(ids.beer.iloc[i])) for i in bg)
            v,b,c=explain_quantile(models[chosen],x[test],x[bg]);error=check_additivity(v,b,c)
            shap_values[repeat,test]=v;shap_base[repeat,test]=b;shap_output[repeat,test]=c
            coefficients[repeat,fold]=models['ridge_raw_mean'].named_steps['model'].coef_
            additivity.append(dict(repeat=repeat,outer_fold=fold,max_error=error))
            gm,gs,spectra=generation_comparison(data,train,test,models['extra_trees'],gp,p['seed']+repeat*10000+fold*1000+900)
            gen_outer.extend(gm.assign(repeat=repeat,outer_fold=fold,inner_selected_method=generation_choice).to_dict('records'))
            gen_sensory.extend(gs.assign(repeat=repeat,outer_fold=fold).to_dict('records'))
            for (method,style),values in spectra.items():gen_spectra[f'{context}_{method}_{style}']=values
            for name in ['logistic','extra_trees']:
                fit_log.append(dict(context=context+'_style',method=name,seed=seed,n_training_beers=len(train)))
                classifier=make_style_model(name,sc,seed).fit(x[train],styles[train]) if refit_models else saved['style_models'][name]
                prediction=classifier.predict(x[test]);np.testing.assert_array_equal(prediction,saved['style_models'][name].predict(x[test]))
                style_oof.extend(dict(repeat=repeat,outer_fold=fold,beer=int(ids.beer.iloc[i]),model=name,observed_style=styles[i],
                    predicted_style=value,correct=bool(value==styles[i])) for i,value in zip(test,prediction))
            fixed_paths.extend(fixed_path_sensitivity(data,test,models,risk,repeat,fold,gp))
            del models,risk,saved
    tables=derive_tables(data,p,rp,ep,folds,grades,probabilities,shap_values,coefficients,selected_gen,gen_outer)
    tables.update({'model_fit_log.csv':fit_log,'grouped_splits.csv':splits_log,'background_beers.csv':backgrounds,
        'gbr_inner_selection.csv':gbr_choices,'risk_inner_selection.csv':risk_choices,'calibration_parameters.csv':calibration,
        'generation_inner_validation.csv':gen_inner,'generation_inner_gate_checks.csv':gen_gates,
        'generation_outer_validation.csv':gen_outer,'generation_outer_sensory_by_target.csv.gz':gen_sensory,
        'style_oof.csv':style_oof,'fixed_historical_path_sensitivity.csv':fixed_paths,'shap_additivity.csv':additivity})
    paths=pd.DataFrame(fixed_paths)
    tables['fixed_historical_path_joint.csv']=paths.groupby(['repeat','beer'],sort=False).agg(
        source_held_out=('source_held_out','all'),all_five_sensory_guards_pass=('all_five_sensory_guards_pass','first')).reset_index()
    return dict(data=data,p=p,rp=rp,ep=ep,tables=tables,grades=grades,probabilities=probabilities,shap_values=shap_values,
        shap_base=shap_base,shap_output=shap_output,coefficients=coefficients,folds=folds,gen_spectra=gen_spectra,selected_gen=selected_gen)


def verify(args):
    out=args.output;s=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'));assert s['protocol_content_sha256']==content_hash(p)
    assert json.loads((out/'protocol.json').read_text(encoding='utf-8'))==p
    for name,digest in s['source_sha256'].items():assert sha256(args.source/name)==digest
    for name,digest in s['input_lf_normalized_sha256'].items():assert text_hash(ROOT/name)==digest
    for name,digest in s['array_sha256'].items():assert sha256(out/name)==digest
    assert sha256(args.raw)==s['raw_source_sha256']
    state=replay(args,refit_models=True)
    for name,table in state['tables'].items():
        expected=pd.read_csv(io.StringIO(pd.DataFrame(table).to_csv(index=False)),float_precision='round_trip')
        pd.testing.assert_frame_equal(read(out/name),expected,check_dtype=False,check_exact=False,rtol=0,atol=1e-10,obj=name)
    saved=np.load(out/'oof_arrays.npz',allow_pickle=False)
    np.testing.assert_array_equal(saved['beer'],state['data']['ids'].beer)
    np.testing.assert_array_equal(saved['targets'],state['data']['targets']);np.testing.assert_array_equal(saved['features'],state['data']['features'])
    for name,matrix in state['grades'].items():np.testing.assert_array_equal(saved['grade_'+name],matrix)
    for name,matrix in state['probabilities'].items():np.testing.assert_allclose(saved['probability_'+name],matrix,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(saved['folds'],state['folds']);np.testing.assert_allclose(saved['coefficients'],state['coefficients'],atol=1e-12,rtol=0)
    shap=np.load(out/'gbr_OOF_SHAP.npz',allow_pickle=False)
    for name,field in [('values','shap_values'),('base','shap_base'),('output','shap_output')]:np.testing.assert_allclose(shap[name],state[field],atol=1e-12,rtol=0)
    generated=np.load(out/'outer_generated_spectra.npz',allow_pickle=False);assert set(generated.files)==set(state['gen_spectra'])
    for name,values in state['gen_spectra'].items():np.testing.assert_allclose(generated[name],values,atol=1e-12,rtol=0)
    long=read(out/'q1_oof_scores52.csv.gz');missing=long.target.isin(['A_acetaldehyde','F_acetaldehyde'])
    assert len(long)==39000 and missing.sum()==1500 and long.loc[missing,'estimated_grade_0_6'].isna().all()
    assert long.loc[missing,'probability_rating_ge1'].isna().all() and not long.safety_certified.any()
    slots=read(out/'q2_all52_rank_slots.csv');assert len(slots)==936
    assert slots[slots.target.isin(['A_acetaldehyde','F_acetaldehyde'])].field.isna().all()
    assert len(state['tables']['model_fit_log.csv'])==126
    result=dict(status='passed',independently_refitted_prediction_models=108,independently_refitted_style_models=18,
        generator_refits=54,nested_outer_contexts=9,nested_inner_contexts=18,compared_tables=len(state['tables']),
        unique_beers=250,repeated_beer_predictions=750,raw_scorecard_rows=len(long),blank_unlabelled_rows=int(missing.sum()),
        inner_probability_and_GBR_grade_arrays_recomputed=9,SHAP_cells_recomputed=int(state['shap_values'].size),
        generated_outer_spectra=sum(len(v) for v in state['gen_spectra'].values()),
        bootstrap_unit='250 beers after averaging repeated losses',all_selection_training_only=True,
        historical_path_sensitivity_not_nested_migration_validation=True,deployment_changed=False,safety_certified=False,
        scope='Same 250 development beers; numerical replay of frozen nested workflows, not an external test or intervention verification.')
    write_json(out/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):verify(parser().parse_args())
