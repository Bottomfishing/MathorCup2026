"""Task4: finite generation comparison and complete raw-scale migration scenarios."""
import argparse
import hashlib
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
from beer_audit.audit import save_csv,write_json,sha256
from beer_models.baseline import make_splits
from beer_models.generation import StyleGenerator
from beer_models.style_models import make_style_model
from beer_models.raw_panel import RawPanelModel,median_grades,rating_loss,predict_raw_frame
from beer_models.tail_risk import base_probabilities,survival,binary_loss,risk_frame
from beer_models.q3_raw import (ShrinkageCopula,spectrum_diagnostics,discrete_w1,
    raw_predictions,defect_guard,resampled_records,uncalibrated_refit_predictions,profile_diagnostics)
from beer_models.defect_constraints import project_defects,defect_aware_box
from beer_models.explain import stratified_draw
from beer_models.review_diagnostics import pareto_mask
from r2_raw_q1 import load_inputs,content_hash,RAW_MANIFEST,PROTOCOL as BASE_PROTOCOL

PROTOCOL=ROOT/'configs/task4_raw_q3_protocol_20261026.json'
LEGACY=ROOT/'outputs/q3_task3_20261025'
POOL=ROOT/'outputs/review_path_tradeoff_20261025'
ANCHORS=ROOT/'outputs/q3_resolution_20261024/anchors.csv'
STABILITY=ROOT/'outputs/review_q3_fixed_paths_20261025/joint_pass_fraction.csv'
R3=ROOT/'outputs/r3_raw_explanation_20261026/predictive_metrics.csv'
STYLE_CONFIG=ROOT/'configs/q3_style_protocol.json'


def text_hash(path):return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def read(path,**kwargs):return pd.read_csv(path,float_precision='round_trip',**kwargs)


def save_tables(output,tables):
    for name,table in tables.items():
        if name.endswith('.gz'):pd.DataFrame(table).to_csv(output/name,index=False,compression={'method':'gzip','mtime':0})
        else:save_csv(output,name,table)


def portrait(frame,bundle):
    long,primary,wide,quality=predict_raw_frame(frame,bundle)
    long=long.merge(risk_frame(frame,bundle,long.beer.drop_duplicates().to_numpy()),on=['beer','target'],how='left',validate='one_to_one')
    long['safety_certified']=False;long['actual_after_measured']=False
    return long,primary,wide,quality


def sensory_validation(q_observed,p_real,g_real,p_gen,g_gen,targets):
    observed_median=median_grades(q_observed)
    empirical=lambda grades:np.eye(7)[grades].mean(0)
    gm,rm,om=map(empirical,[g_gen,g_real,observed_median])
    predicted,real,observed=p_gen.mean(0),p_real.mean(0),q_observed.mean(0)
    rows=[]
    for j,target in enumerate(targets):
        rows.append(dict(target=target,model_median_w1=float(discrete_w1(gm[j],rm[j])),
            observed_median_w1=float(discrete_w1(gm[j],om[j])),
            model_rating_distribution_w1=float(discrete_w1(predicted[j],real[j])),
            observed_rating_distribution_w1=float(discrete_w1(predicted[j],observed[j])),
            generated_mean_predicted_grade=float(g_gen[:,j].mean()),heldout_mean_predicted_grade=float(g_real[:,j].mean()),
            heldout_mean_observed_median=float(observed_median[:,j].mean()),
            generated_expected_rating=float(predicted[j]@np.arange(7)),heldout_observed_expected_rating=float(observed[j]@np.arange(7))))
    return rows


def generation_gate(validation):
    means=validation.groupby('method',sort=False).mean(numeric_only=True)
    a,b=means.loc['interpolate'],means.loc['shrinkage_copula']
    rows=[]
    def add(rule,left,right,passed):rows.append(dict(rule=rule,copula_value=float(left),limit=float(right),passed=bool(passed)))
    for metric in ['marginal_w1','energy','exact_training_copy_rate']:
        add(metric,b[metric],a[metric],b[metric]<=a[metric]+1e-12)
    add('variance_lower',b.variance_ratio,.8,b.variance_ratio>=.8)
    add('variance_upper',b.variance_ratio,1.2,b.variance_ratio<=1.2)
    add('rank_correlation_rmse',b.rank_correlation_rmse,a.rank_correlation_rmse+.05,b.rank_correlation_rmse<=a.rank_correlation_rmse+.05)
    add('model_median_w1',b.model_median_w1,a.model_median_w1+.02,b.model_median_w1<=a.model_median_w1+.02)
    for style,group in validation.groupby('style',sort=False):
        m=group.groupby('method').mean(numeric_only=True);aa,bb=m.loc['interpolate'],m.loc['shrinkage_copula']
        add(style+'_energy',bb.energy,aa.energy*1.1,bb.energy<=aa.energy*1.1)
        add(style+'_coverage',bb.heldout_coverage,aa.heldout_coverage-.05,bb.heldout_coverage>=aa.heldout_coverage-.05)
    return ('shrinkage_copula' if all(row['passed'] for row in rows) else 'interpolate'),pd.DataFrame(rows)


def style_metrics(oof):
    rows=[]
    for name,g in pd.DataFrame(oof).groupby('model',sort=False):
        rows.append(dict(model=name,scope='all22styles_OOF',n=len(g),accuracy=float(g.correct.mean())))
        for style,h in g.groupby('observed_style',sort=False):
            rows.append(dict(model=name,scope=style,n=len(h),accuracy=float(h.correct.mean())))
    return pd.DataFrame(rows)


def run_generation(d,p,bundle,args):
    x,q,styles=d['x'],d['distribution'],d['ids'].tasting_category_fine.to_numpy()
    sc=json.loads(STYLE_CONFIG.read_text(encoding='utf-8'));validation=[];sensory=[];styles_oof=[];copula_log=[]
    for fold,(train,test) in enumerate(make_splits(styles,p['validation_folds'],p['validation_seed'])):
        print(f'Generation held-out comparison fold {fold}',flush=True)
        interpolation=StyleGenerator().fit(x[train],styles[train]);copula=ShrinkageCopula().fit(x[train],styles[train])
        outer=joblib.load(args.q1/'models'/f'outer{fold}.joblib')
        assert outer['point_choice']=='extra_trees__median'
        model=outer['models']['extra_trees'];preal=base_probabilities(model,x[test]);greal=model.predict(x[test])['grade']
        classifiers={name:make_style_model(name,sc,sc['seed']+fold).fit(x[train],styles[train]) for name in ['logistic','extra_trees']}
        for name,classifier in classifiers.items():
            predictions=classifier.predict(x[test])
            styles_oof.extend(dict(beer=int(d['ids'].beer.iloc[i]),outer_fold=fold,model=name,observed_style=styles[i],
                predicted_style=pred,correct=bool(pred==styles[i])) for i,pred in zip(test,predictions))
        for si,style in enumerate(p['styles']):
            tr=np.flatnonzero(styles[train]==style);te=np.flatnonzero(styles[test]==style)
            ztrain=interpolation.robust_.transform(interpolation.x_[tr])
            ztest=interpolation.robust_.transform(interpolation.impute_.transform(x[test[te]]))
            copula_log.append(dict(outer_fold=fold,style=style,n_training_beers=len(tr),shrinkage=copula.shrinkage[style],
                minimum_correlation_eigenvalue=float(np.linalg.eigvalsh(copula.correlations[style]).min())))
            for method in p['generation_methods']:
                seed=p['validation_generation_seed']+fold*100+si
                gen=interpolation.sample(style,p['generation_count'],'interpolate',seed)[0] if method=='interpolate' else copula.sample(style,p['generation_count'],seed)
                row=dict(outer_fold=fold,style=style,method=method,seed=seed,
                    **spectrum_diagnostics(ztrain,ztest,interpolation.robust_.transform(gen),p))
                sg=sensory_validation(q[test[te]],preal[te],greal[te],base_probabilities(model,gen),model.predict(gen)['grade'],d['targets'])
                sensory.extend(dict(outer_fold=fold,style=style,method=method,**r) for r in sg)
                for key in ['model_median_w1','observed_median_w1','model_rating_distribution_w1','observed_rating_distribution_w1']:
                    row[key]=float(np.mean([r[key] for r in sg]))
                validation.append(row)
        del outer,model
    validation=pd.DataFrame(validation);selected,gate=generation_gate(validation)
    interpolation=StyleGenerator().fit(x,styles);copula=ShrinkageCopula().fit(x,styles)
    classifiers={name:make_style_model(name,sc,sc['seed']).fit(x,styles) for name in ['logistic','extra_trees']}
    for name,model in classifiers.items():joblib.dump(model,args.output/'models'/(name+'.joblib'),compress=3)
    candidates=[];metadata=[];full_diagnostics=[]
    old=read(LEGACY/'generated_profiles.csv').set_index('generated_id')
    for si,style in enumerate(p['styles']):
        idx=interpolation.indices_[style];real=interpolation.robust_.transform(interpolation.x_[idx])
        copula_log.append(dict(outer_fold=-1,style=style,n_training_beers=len(idx),shrinkage=copula.shrinkage[style],
            minimum_correlation_eigenvalue=float(np.linalg.eigvalsh(copula.correlations[style]).min())))
        for method in p['generation_methods']:
            seed=p['generation_seed']+si
            if method=='interpolate':
                gen,meta=interpolation.sample(style,p['generation_count'],method,seed)
                np.testing.assert_allclose(gen,old.loc[[f'{style}_{i:03d}' for i in range(len(gen))],d['features']].to_numpy(float),atol=1e-10,rtol=0)
            else:gen=copula.sample(style,p['generation_count'],seed);meta=None
            for i,row in enumerate(gen):
                gid=f'{method}_{style}_{i:03d}'
                candidates.append(dict(generated_id=gid,style=style,method=method,**dict(zip(d['features'],row))))
                metadata.append(dict(generated_id=gid,style=style,method=method,seed=seed,
                    parent_1=int(d['ids'].beer.iloc[meta['parents'][i,0]]) if meta else None,
                    parent_2=int(d['ids'].beer.iloc[meta['parents'][i,1]]) if meta else None,
                    parent_weight=float(meta['weights'][i]) if meta else None,
                    physical_brewing_validated=False,coordinate_system='supplied_attachment1_not_concentration_blend'))
            full_diagnostics.append(dict(style=style,method=method,scope='full_training_descriptive_not_holdout',
                **spectrum_diagnostics(real,real,interpolation.robust_.transform(gen),p)))
    candidates=pd.DataFrame(candidates);metadata=pd.DataFrame(metadata)
    main=candidates[candidates.method.eq(selected)].copy();main.generated_id=main.generated_id.str.removeprefix(selected+'_')
    main_meta=metadata[metadata.method.eq(selected)].copy();main_meta.generated_id=main_meta.generated_id.str.removeprefix(selected+'_')
    frames=candidates[['generated_id',*d['features']]].rename(columns={'generated_id':'beer'})
    long,_,_,_=portrait(frames,bundle);long=long.rename(columns={'beer':'generated_id'})
    long=long.merge(candidates[['generated_id','style','method']],on='generated_id',validate='many_to_one')
    long['prediction_role']='synthetic_full_fit_inference_not_measured_or_OOF'
    main_long=long[long.method.eq(selected)].copy();main_long.generated_id=main_long.generated_id.str.removeprefix(selected+'_')
    wide=main_long.pivot(index='generated_id',columns='target',values='estimated_grade_0_6').loc[main.generated_id,d['scope'].target_id].reset_index()
    style_diag=candidates[['generated_id','style','method']].copy()
    for name,model in classifiers.items():style_diag[name+'_prediction']=model.predict(candidates[d['features']].to_numpy(float))
    style_diag['both_match_requested_style']=(style_diag.logistic_prediction==style_diag['style'])&(style_diag.extra_trees_prediction==style_diag['style'])
    fingerprints=main_long.groupby(['style','target'],sort=False).agg(n=('generated_id','size'),
        mean_grade=('estimated_grade_0_6','mean'),p10_grade=('estimated_grade_0_6',lambda a:a.quantile(.1)),
        p90_grade=('estimated_grade_0_6',lambda a:a.quantile(.9)),mean_probability_rating_ge1=('probability_rating_ge1','mean')).reset_index()
    tables={'generation_validation.csv':validation,'generation_validation_by_target.csv.gz':sensory,
        'generation_replacement_checks.csv':gate,'style_oof.csv':styles_oof,'copula_fit_parameters.csv':copula_log,
        'generation_candidate_profiles.csv':candidates,'generation_candidate_metadata.csv':metadata,
        'generation_full_training_diagnostics.csv':full_diagnostics,'generation_style_predictions.csv':style_diag,
        'generation_profile_diagnostics.csv':profile_diagnostics(interpolation,candidates,d['ids'],d['features'],p),
        'style_classifier_metrics.csv':style_metrics(styles_oof),
        'generation_candidate_sensory52.csv.gz':long,'generated_profiles.csv':main,'generated_metadata.csv':main_meta,
        'generated_sensory52.csv.gz':main_long,'generated_grades_all52.csv':wide,'sensory_fingerprints52.csv':fingerprints}
    return tables,selected,classifiers,interpolation


def reconstruct_paths(rows,source,anchors,features,protected):
    values=source+rows.fraction.to_numpy(float)[:,None]*(anchors[rows.anchor.to_numpy(int)]-source)
    for policy in ['preserve','reduce']:
        mask=rows.policy.eq(policy).to_numpy()
        values[mask]=project_defects(values[mask],source,protected,policy)
    if not rows.policy.isin(['preserve','reduce']).all():raise ValueError('Unknown frozen path policy')
    return values


def five_defect_rows(key,before,after,bpred,apred,d,p,bundle):
    rows=[];indices=[d['targets'].index(t) for t in p['defect_targets']]
    _,deltas,passes=defect_guard(bpred,apred,indices,p['nonincrease_tolerance'])
    for i,target in enumerate(p['defect_targets']):
        field=p['protected_features'][i] if i<4 else ''
        # Defect order is DMS, Diacetyl, 4VG, t2n, Stale.
        chemical=(after[d['features'].index(field)]<=before[d['features'].index(field)]+p['nonincrease_tolerance']) if field else None
        row=dict(**key,target=target,chemical_field=field,coordinate_nonincrease=chemical,
            grade_before=int(bpred['grade'][0,indices[i]]),grade_after=int(apred['grade'][0,indices[i]]),
            raw_grade_nonincrease=bool(deltas['grade'][0,i]<=0),risk_status=bundle['risk_status'][indices[i]],
            R3_median_factor_status='weak_predictor_exploratory_only',safety_certified=False,actual_after_measured=False)
        for name in ['calibrated','uncalibrated']:
            for event,k in [('ge1',0),('ge5',1),('6',2)]:row[name+'_probability_delta_'+event]=float(deltas[name][0,i,k])
            row[name+'_risk_nonincrease']=bool((deltas[name][0,i]<=p['nonincrease_tolerance']).all())
        rows.append(row)
    return rows


def run_migration(d,p,bundle,classifiers,generator,args):
    before=read(LEGACY/'transfer_source_profiles.csv',dtype={'beer':str}).set_index('beer')
    old_after=read(LEGACY/'transfer_profiles.csv',dtype={'beer':str}).set_index('beer').loc[before.index]
    original=read(LEGACY/'transfer_results.csv',dtype={'beer':str}).set_index('beer').loc[before.index]
    pool=read(POOL/'feasible_candidates.csv',dtype={'beer':str});old_weights=read(POOL/'weight_choices.csv',dtype={'beer':str})
    anchors=read(ANCHORS).set_index('anchor')[d['features']].to_numpy(float)
    assert len(pool)==8293 and len(before)==11 and list(before)==d['features']
    evidence=read(LEGACY/'source_defect_evidence.csv',dtype={'beer':str})
    raw_source=read(args.source/'1.csv',dtype={'beer':str}).set_index('beer')
    calibration=pd.read_excel(args.source/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer')
    protected=[d['features'].index(f) for f in p['protected_features']]
    for e in evidence.itertuples():
        value=float(raw_source.loc[e.beer,e.field])
        if e.basis=='observed_coordinate':np.testing.assert_allclose(value,e.constraint_cap_coordinate,atol=0,rtol=0)
        else:
            assert e.field=='X2.3.butanedione' and np.isnan(value) and e.constraint_cap_coordinate==0
            assert float(calibration.loc[int(e.beer),e.field])==0
        np.testing.assert_allclose(before.loc[e.beer,e.field],e.constraint_cap_coordinate,atol=0,rtol=0)
    tripel=generator.x_[generator.indices_['Tripel']];lower,upper=tripel.min(0),tripel.max(0)
    center=generator.robust_.transform(tripel).mean(0);indices=[d['targets'].index(t) for t in p['defect_targets']]
    grid=[];choices=[];selected_profiles=[];checks=[];old_checks=[];old_results=[];weighted_profiles=[];weight_checks=[]
    for beer in before.index:
        print(f'Complete migration constraints: beer {beer}',flush=True)
        source=before.loc[beer].to_numpy(float);old_point=old_after.loc[beer].to_numpy(float)
        subset=pool.loc[pool.beer.eq(beer)].copy().reset_index(drop=True)
        values=reconstruct_paths(subset,source,anchors,d['features'],protected)
        bpred=raw_predictions(bundle,source[None],d['parents']);apred=raw_predictions(bundle,values,d['parents'])
        okay,deltas,individual=defect_guard(bpred,apred,indices,p['nonincrease_tolerance'])
        chemical=(values[:,protected]<=source[protected]+p['nonincrease_tolerance']).all(1)
        box=defect_aware_box(values,lower,upper,protected)
        style={name:model.predict(values) for name,model in classifiers.items()}
        style_ok=(style['logistic']=='Tripel')&(style['extra_trees']=='Tripel')
        z=generator.robust_.transform(values);zs=generator.robust_.transform(source[None])[0]
        distances=np.sqrt(np.mean((z-center)**2,axis=1));changes=np.sqrt(np.mean((z-zs)**2,axis=1))
        initial=float(np.sqrt(np.mean((zs-center)**2)))
        np.testing.assert_allclose(distances,subset.target_distance,atol=1e-12,rtol=0)
        np.testing.assert_allclose(changes,subset.change_distance,atol=1e-12,rtol=0)
        good=okay&chemical&box&style_ok&(distances<initial-1e-8)
        front=np.zeros(len(subset),bool)
        if good.any():front[good]=pareto_mask(np.column_stack([distances[good],changes[good]]))
        for k,r in subset.iterrows():
            row=dict(beer=beer,candidate=int(r.candidate),anchor=int(r.anchor),fraction=float(r.fraction),policy=r.policy,
                target_distance=distances[k],change_distance=changes[k],historical_continuous_guard_eligibility_reused=True,
                range_pass=bool(box[k]),chemical_pass=bool(chemical[k]),style_pass=bool(style_ok[k]),
                all_raw_grades_pass=bool(individual['grade'][k]),all_calibrated_risks_pass=bool(individual['calibrated'][k]),
                all_uncalibrated_risks_pass=bool(individual['uncalibrated'][k]),all_current_constraints_pass=bool(good[k]),pareto=bool(front[k]))
            for j,target in enumerate(p['defect_targets']):
                row[target+'_grade_delta']=int(deltas['grade'][k,j])
                for name in ['calibrated','uncalibrated']:
                    for cut,ci in [('ge1',0),('ge5',1),('6',2)]:row[target+'_'+name+'_delta_'+cut]=float(deltas[name][k,j,ci])
            grid.append(row)
        historical_pred=raw_predictions(bundle,old_point[None],d['parents'])
        check=five_defect_rows(dict(beer=beer,scenario='original_fixed_path'),source,old_point,bpred,historical_pred,d,p,bundle)
        old_checks.extend(check)
        old_ok=all(r['raw_grade_nonincrease'] and r['calibrated_risk_nonincrease'] and r['uncalibrated_risk_nonincrease'] for r in check)
        old_results.append(dict(beer=beer,passes_current_raw_scenario=old_ok,
            uses_reference_zero_cap=bool(original.loc[beer,'uses_reference_zero_cap']),safety_certified=False,
            status='historical_path_current_model_scenario_only' if old_ok else 'historical_path_rejected_by_current_raw_checks'))
        for weight in p['weights']:
            previous=old_weights.loc[old_weights.beer.eq(beer)&np.isclose(old_weights.weight,weight)].iloc[0]
            old_k=int(np.flatnonzero(subset.candidate.eq(int(previous.candidate)))[0])
            old_w_checks=five_defect_rows(dict(beer=beer,weight=weight,scenario='original_weight_choice'),source,values[old_k],bpred,
                {name:v[old_k:old_k+1] for name,v in apred.items()},d,p,bundle)
            weight_checks.extend(old_w_checks)
            row=dict(beer=beer,weight=weight,source_target_distance=initial,n_historically_eligible=len(subset),
                n_pass_current=int(good.sum()),original_weight_candidate=int(previous.candidate),
                original_weight_pass_current=bool(good[old_k]),selected=False,
                candidate=None,anchor=None,fraction=None,policy=None,target_distance=None,change_distance=None,
                objective=None,distance_reduction_fraction=None,pareto=False,
                uses_reference_zero_cap=bool(original.loc[beer,'uses_reference_zero_cap']),
                status='no_candidate_passed_complete_current_checks',safety_certified=False,actual_after_measured=False)
            if good.any():
                eligible=np.flatnonzero(good);k=int(eligible[np.argmin(weight*distances[eligible]+(1-weight)*changes[eligible])])
                r=subset.iloc[k]
                row.update(selected=True,status='unvalidated_model_scenario_candidate',candidate=int(r.candidate),
                    anchor=int(r.anchor),fraction=float(r.fraction),policy=r.policy,target_distance=distances[k],
                    change_distance=changes[k],objective=weight*distances[k]+(1-weight)*changes[k],
                    distance_reduction_fraction=1-distances[k]/initial,pareto=bool(front[k]),
                    uses_reference_zero_cap=bool(original.loc[beer,'uses_reference_zero_cap']))
                weighted_profiles.append(dict(beer=beer,weight=weight,**dict(zip(d['features'],values[k]))))
                if np.isclose(weight,p['main_weight']):
                    selected_profiles.append(dict(beer=beer,**dict(zip(d['features'],values[k]))))
                    checks.extend(five_defect_rows(dict(beer=beer,scenario='main_weight_selected'),source,values[k],bpred,
                        {name:v[k:k+1] for name,v in apred.items()},d,p,bundle))
            choices.append(row)
    choices=pd.DataFrame(choices);selected_profiles=pd.DataFrame(selected_profiles,columns=['beer',*d['features']])
    main=choices[np.isclose(choices.weight,p['main_weight'])].copy()
    full_before=before.copy().reset_index();full_after=old_after.copy().reset_index()
    b_long,_,_,_=portrait(full_before,bundle);a_long,_,_,_=portrait(full_after,bundle)
    old_sensory=b_long.merge(a_long,on=['beer','target'],suffixes=('_before','_after'),validate='one_to_one')
    if len(selected_profiles):
        selected_long,_,selected_wide,_=portrait(selected_profiles,bundle)
    else:
        selected_long=b_long.iloc[:0].copy();selected_wide=pd.DataFrame(columns=['beer',*d['scope'].target_id])
    tables={'migration_grid_checks.csv.gz':grid,'weight_choices_complete.csv':choices,'transfer_results.csv':main,
        'transfer_profiles.csv':selected_profiles,'all_weight_selected_profiles.csv':pd.DataFrame(weighted_profiles,columns=['beer','weight',*d['features']]),
        'source_defect_evidence.csv':evidence,'transfer_source_profiles.csv':full_before,
        'original_fixed_path_results.csv':old_results,'original_fixed_path_five_defect_checks.csv':old_checks,
        'original_weight_five_defect_checks.csv':weight_checks,'selected_five_defect_checks.csv':pd.DataFrame(checks,columns=pd.DataFrame(old_checks).columns),
        'original_fixed_path_sensory52.csv.gz':old_sensory,'transfer_sensory52.csv.gz':selected_long,'transfer_grades_all52.csv':selected_wide}
    tables.update(constraint_attrition(pd.DataFrame(grid),p))
    return tables,before,old_after


def constraint_attrition(grid,p):
    rows=[];defects=[];tol=p['nonincrease_tolerance']
    for beer,g in grid.groupby('beer',sort=False):
        keep=np.ones(len(g),bool)
        rows.append(dict(beer=beer,stage='historically_eligible',n_pass=len(g)))
        for stage,field in [('coordinate_caps','chemical_pass'),('target_ranges','range_pass'),('two_style_models','style_pass'),
            ('all_five_raw_grades','all_raw_grades_pass'),('all_five_calibrated_probabilities','all_calibrated_risks_pass'),
            ('all_five_uncalibrated_probabilities','all_uncalibrated_risks_pass')]:
            keep &= g[field].to_numpy(bool);rows.append(dict(beer=beer,stage=stage,n_pass=int(keep.sum())))
    for scope,g in [('all_sources',grid),*list(grid.groupby('beer',sort=False))]:
        for target in p['defect_targets']:
            row=dict(beer=scope,target=target,n_candidates=len(g),grade_nonincrease_fraction=float((g[target+'_grade_delta']<=0).mean()))
            for version in ['calibrated','uncalibrated']:
                for cut in ['ge1','ge5','6']:
                    row[version+'_pass_fraction_'+cut]=float((g[target+'_'+version+'_delta_'+cut]<=tol).mean())
            defects.append(row)
    return {'migration_constraint_attrition.csv':pd.DataFrame(rows),'defect_guard_pass_fractions.csv':pd.DataFrame(defects)}


def summarize_refits(changes):
    summary=[]
    for (beer,target),g in changes.groupby(['beer','target'],sort=False):
        held=g[~g.source_in_training_sample]
        summary.append(dict(beer=beer,target=target,refits=len(g),grade_pass_fraction=float(g.grade_pass.mean()),
            probability_pass_fraction=float(g.all_probability_pass.mean()),n_source_oob_refits=len(held),
            probability_pass_fraction_source_OOB=float(held.all_probability_pass.mean()) if len(held) else np.nan,
            nonzero_delta_q025=float(g.probability_delta_ge1.quantile(.025)),nonzero_delta_median=float(g.probability_delta_ge1.median()),
            nonzero_delta_q975=float(g.probability_delta_ge1.quantile(.975)),interval_scope='refit_percentiles_not_intervention_CI'))
    joint=changes.assign(joint=changes.grade_pass&changes.all_probability_pass).groupby(['repeat','beer']).joint.all().unstack('beer')
    inbag=changes.groupby(['repeat','beer']).source_in_training_sample.first().unstack('beer')
    old=read(STABILITY,dtype={'beer':str}).set_index('beer');combined=[]
    for beer in changes.beer.drop_duplicates():
        mask=~inbag[beer]
        combined.append(dict(beer=beer,raw_ET_all_five_joint_fraction=float(joint[beer].mean()),raw_refits=len(joint),
            source_oob_refits=int(mask.sum()),raw_joint_fraction_source_OOB=float(joint.loc[mask,beer].mean()) if mask.any() else np.nan,
            raw_joint_fraction_source_in_bag=float(joint.loc[~mask,beer].mean()) if (~mask).any() else np.nan,
            legacy_three_standardized_proxy_fraction=float(old.loc[beer,'all_three_pass_fraction']),legacy_refits=int(old.loc[beer,'refits']),
            selection_scope='fixed_original_path_no_reselection',safety_certified=False))
    return summary,combined


def refit_fixed_paths(d,p,bp,before,after,args):
    indices=[d['targets'].index(t) for t in p['defect_targets']];rng=np.random.default_rng(p['seed'])
    changes=[];oob=[];draw_rows=[];fits=[]
    for repeat in range(p['refit_repeats']):
        tick=time.perf_counter();draw=stratified_draw(d['ids'].tasting_category_fine.to_numpy(),rng)
        positions,scores,weights=resampled_records(d,draw)
        model=RawPanelModel('extra_trees',bp,p['seed']+repeat,d['parents']).fit(d['x'][draw],positions,scores,weights,d['distribution'][draw])
        np.testing.assert_allclose(model.imputer.statistics_,np.nanmedian(d['x'][draw],axis=0),atol=0,rtol=0)
        draw_rows.extend(dict(repeat=repeat,occurrence=i,beer=int(d['ids'].beer.iloc[j])) for i,j in enumerate(draw))
        a=uncalibrated_refit_predictions(model,before.to_numpy(float),d['parents'])
        b=uncalibrated_refit_predictions(model,after.to_numpy(float),d['parents'])
        sampled_beers=set(d['ids'].beer.iloc[draw].astype(str))
        for i,beer in enumerate(before.index):
            for j,target in enumerate(p['defect_targets']):
                delta=b['probability'][i,indices[j]]-a['probability'][i,indices[j]]
                changes.append(dict(repeat=repeat,beer=beer,target=target,source_in_training_sample=beer in sampled_beers,grade_before=int(a['grade'][i,indices[j]]),
                    grade_after=int(b['grade'][i,indices[j]]),grade_pass=bool(b['grade'][i,indices[j]]<=a['grade'][i,indices[j]]),
                    probability_delta_ge1=float(delta[0]),probability_delta_ge5=float(delta[1]),probability_delta_6=float(delta[2]),
                    all_probability_pass=bool((delta<=p['nonincrease_tolerance']).all()),calibration_applied=False,safety_certified=False))
        held=np.setdiff1d(np.arange(len(d['x'])),draw);qp=d['distribution'][held]
        prediction=uncalibrated_refit_predictions(model,d['x'][held],d['parents']);prior=d['distribution'][draw].mean(0)
        prior_grade=median_grades(np.broadcast_to(prior,(len(held),*prior.shape)))
        oob.append(dict(repeat=repeat,target='all50',n_oob=len(held),metric='raw_rating_MAE',
            model_loss=float(rating_loss(qp,prediction['grade']).mean()),training_prior_loss=float(rating_loss(qp,prior_grade).mean())))
        for j,target in enumerate(p['defect_targets']):
            actual=1-qp[:,indices[j],0]
            oob.append(dict(repeat=repeat,target=target,n_oob=len(held),metric='nonzero_uncalibrated_Brier',
                model_loss=float(binary_loss(prediction['probability'][:,indices[j],0],actual).mean()),
                training_prior_loss=float(binary_loss(1-prior[indices[j],0],actual).mean())))
        fits.append(dict(repeat=repeat,seed=p['seed']+repeat,n_resampled_beer_occurrences=len(draw),n_rating_records=len(positions),seconds=time.perf_counter()-tick))
        print(f'Fixed original paths raw refits {repeat+1}/{p["refit_repeats"]}',flush=True)
    changes=pd.DataFrame(changes);summary,combined=summarize_refits(changes)
    return {'raw_fixed_path_refit_changes.csv':changes,'raw_fixed_path_refit_summary.csv':summary,
        'fixed_path_joint_refit_comparison.csv':combined,'raw_refit_oob_metrics.csv':oob,
        'raw_refit_draws.csv':draw_rows,'raw_refit_fit_log.csv':fits}


def run(args):
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'));bp=json.loads(BASE_PROTOCOL.read_text(encoding='utf-8'))
    d=load_inputs(args.source,args.raw,bp)
    parent=json.loads((args.q1/'summary.json').read_text(encoding='utf-8'))
    assert sha256(args.q1/'model.joblib')==parent['model_sha256']
    assert parent['source_sha256']=={n:sha256(args.source/n) for n in ['1.csv','4.csv','8.xlsx']}
    bundle=joblib.load(args.q1/'model.joblib')
    assert bundle['features']==d['features'] and bundle['targets']==d['targets']
    assert not bundle['tail_point_qualified_on_historical_folds']
    r3=read(R3);assert r3[r3.method.eq('gbr_selected')&r3.target.isin(p['defect_targets'])].support_status.eq('weak_predictor_exploratory_only').all()
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'models').mkdir()
    write_json(args.output/'protocol.json',p)
    gt,method,classifiers,generator=run_generation(d,p,bundle,args);save_tables(args.output,gt)
    mt,before,after=run_migration(d,p,bundle,classifiers,generator,args);save_tables(args.output,mt)
    rt=refit_fixed_paths(d,p,bp,before,after,args);save_tables(args.output,rt)
    provenance=[BASE_PROTOCOL,STYLE_CONFIG,R3,ANCHORS,STABILITY,POOL/'feasible_candidates.csv',POOL/'weight_choices.csv',
        *[LEGACY/name for name in ['generated_profiles.csv','transfer_source_profiles.csv','transfer_profiles.csv','transfer_results.csv','source_defect_evidence.csv']]]
    summary=dict(status='completed',baseline_commit=p['baseline_commit'],protocol_content_sha256=content_hash(p),
        source_sha256=parent['source_sha256'],raw_source_sha256=d['manifest']['sha256'],
        input_lf_normalized_sha256={f.relative_to(ROOT).as_posix():text_hash(f) for f in provenance},
        q1_model_sha256=parent['model_sha256'],q1_outer_model_sha256={f'outer{i}.joblib':sha256(args.q1/'models'/f'outer{i}.joblib') for i in range(3)},
        selected_generation_method=method,generation_validation_cases=len(gt['generation_validation.csv']),
        generated_profiles=len(gt['generated_profiles.csv']),generation_candidate_profiles=len(gt['generation_candidate_profiles.csv']),
        main_scenario_selected=int(mt['transfer_results.csv'].selected.sum()),n_sources=len(before),
        frozen_historical_pool=8293,fully_checked_weight_choices=len(mt['weight_choices_complete.csv']),
        original_paths_passing_current_raw=int(pd.DataFrame(mt['original_fixed_path_results.csv']).passes_current_raw_scenario.sum()),
        raw_fixed_path_refits=p['refit_repeats'],legacy_fixed_path_refits_reused=200,
        stale_probability_support='insufficient_heldout_probability_support',
        safety_certified=False,physical_feasibility_certified=False,actual_after_measured=False,task5_started=False)
    write_json(args.output/'summary.json',summary);plot_results(args.output)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    validation=read(output/'generation_validation.csv');joint=read(output/'fixed_path_joint_refit_comparison.csv',dtype={'beer':str})
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for method,offset in [('interpolate',-.18),('shrinkage_copula',.18)]:
        part=validation[validation.method.eq(method)].groupby('style').mean(numeric_only=True).reindex(['Tripel','Kriek','Lambic'])
        axes[0].bar(np.arange(3)+offset,part.variance_ratio,width=.36,label=method)
        axes[1].bar(np.arange(3)+offset,part.energy,width=.36,label=method)
    for ax in axes[:2]:ax.set_xticks(np.arange(3),['Tripel','Kriek','Lambic']);ax.legend(fontsize=7)
    axes[0].axhline(1.,color='grey',linestyle=':');axes[0].set_title('Within-style variance ratio')
    axes[1].set_title('Held-out energy distance (lower is better)')
    axes[2].bar(np.arange(len(joint)),joint.raw_ET_all_five_joint_fraction)
    axes[2].set(xticks=np.arange(len(joint)),xticklabels=joint.beer,ylim=(0,1),title='Fixed paths: 20 raw ET refits',ylabel='Joint model-guard pass fraction')
    if joint.raw_ET_all_five_joint_fraction.eq(0).all():
        axes[2].text(.5,.55,'All 11 fixed paths: 0 / 20 joint passes',ha='center',va='center',transform=axes[2].transAxes,fontsize=9)
    axes[2].tick_params(axis='x',labelsize=7)
    fig.suptitle('Q3 model diagnostics; generated spectra and migration are unvalidated scenarios')
    fig.savefig(output/'q3_raw_diagnostics.png',dpi=160);fig.savefig(output/'q3_raw_diagnostics.svg')
    svg=output/'q3_raw_diagnostics.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--q1',type=Path,default=ROOT/'outputs/r2b_tail_risk_20261026')
    p.add_argument('--raw',type=Path,default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    return p


if __name__=='__main__':
    with threadpool_limits(limits=1):run(parser().parse_args())
