"""R3: nested finite GBR and held-out raw-scale explanations, without Q1/Q3 changes."""
import argparse
import hashlib
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
from beer_audit.audit import save_csv,write_json,sha256
from beer_audit.checks import repair_utf8_mojibake
from beer_models.baseline import make_splits
from beer_models.raw_panel import RawPanelModel,median_grades,rating_loss,round_grade
from beer_models.raw_explain import (fit_raw_ridge,explain_linear,explain_quantile,check_additivity,
    decision,nonzero_top,training_modules,module_importance,attribution_direction,predictive_support)
from r2_raw_q1 import load_inputs,content_hash,RAW_MANIFEST,PROTOCOL as BASE_PROTOCOL

PROTOCOL=ROOT/'configs/r3_raw_explanation_protocol_20261026.json'
OLD=ROOT/'outputs/q2_task4_20261025/top3_panel_all50.csv'
LITERATURE=ROOT/'outputs/task3_q2_literature_20261025/top3_all50_literature.csv'
META=ROOT/'outputs/p0_audit_20261023_final/feature_dictionary.csv'
ELIGIBILITY=ROOT/'configs/q2_exploratory_v1.json'


def text_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


def fit_gbr(d,train,method,seed,bp):
    lookup=np.full(len(d['x']),-1,int);lookup[train]=np.arange(len(train))
    keep=lookup[d['positions']]>=0
    return RawPanelModel(method,bp,seed,d['parents']).fit(d['x'][train],lookup[d['positions'][keep]],
        d['scores'][keep],d['weights'][keep],d['distribution'][train])


def select_gbr(predictions,q,candidates):
    losses={name:float(rating_loss(q,predictions[name]).mean()) for name in candidates}
    winner=next(name for name in candidates if losses[name]<=min(losses.values())+1e-12)
    return winner,losses


def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b) if a|b else np.nan


def assemble(d,p,folds,grades,values,bases,continuous,coefficients,groups_by_fold,edges_by_fold):
    x,q,targets,features=d['x'],d['distribution'],d['targets'],d['features']
    metadata=pd.read_csv(META).set_index('field_name_R')
    exclusion=json.loads(ELIGIBILITY.read_text(encoding='utf-8'))['exclude_from_compound_top3']
    eligible=np.array([i for i,f in enumerate(features) if f not in exclusion])
    family=np.array([i for i,f in enumerate(features) if f not in exclusion or f=='bitterness'])
    assert len(eligible)==219 and len(family)==220
    old=pd.read_csv(OLD);lit=pd.read_csv(LITERATURE)
    tables={};metric_rows=[];pred_rows=[];add_rows=[]
    for method,g in grades.items():
        loss=rating_loss(q,g).mean(0)
        if method=='training_median':gain=lo=hi=np.zeros(len(targets))
        else:gain,lo,hi=predictive_support(q,g,grades['training_median'],d['ids'].tasting_category_fine.to_numpy(),p)
        for j,target in enumerate(targets):
            supported=method!='training_median' and gain[j]>=.01 and lo[j]>0
            metric_rows.append(dict(method=method,target=target,raw_rating_mae=loss[j],
                improvement_vs_training_median=gain[j],lower95=lo[j],upper95=hi[j],
                support_status='historical_predictive_support_not_causal' if supported else 'weak_predictor_exploratory_only',
                interval_scope='fixed_OOF_loss_not_refit_uncertainty',multiplicity_adjusted=False,migration_control_validated=False))
        for i,beer in enumerate(d['ids'].beer):
            for j,target in enumerate(targets):
                pred_rows.append(dict(method=method,beer=int(beer),outer_fold=int(folds[i]),target=target,estimated_grade=int(g[i,j])))
                if method in values:
                    add_rows.append(dict(method=method,beer=int(beer),target=target,outer_fold=int(folds[i]),
                        base_value=bases[method][i,j],sum_attributions=values[method][i,j].sum(),
                        continuous_unprocessed=continuous[method][i,j],clipped_0_6=float(np.clip(continuous[method][i,j],0,6)),
                        rounded_before_parent_cap=int(round_grade(continuous[method][i:i+1,j:j+1])[0,0]),
                        final_grade=int(g[i,j]),postprocessing_delta=float(g[i,j]-continuous[method][i,j])))
    metrics=pd.DataFrame(metric_rows);tables['predictive_metrics.csv']=metrics
    tables['oof_predictions.csv.gz']=pd.DataFrame(pred_rows);tables['additivity_and_postprocessing.csv.gz']=pd.DataFrame(add_rows)
    fold_scores=[];fold_top=[];stability=[];all_scores=[];ranks=[];family_ranks=[];module_rows=[];module_tops=[]
    neighbors={f:{} for f in features}
    for edges in edges_by_fold:
        for i,j,rho in edges:
            for a,b in [(i,j),(j,i)]:neighbors[features[a]][features[b]]=neighbors[features[a]].get(features[b],0)+1
    def enrich(method,j,indices,score,frequency,direction,scope):
        result=[];target=targets[j]
        status=metrics.loc[metrics.method.eq(method)&metrics.target.eq(target),'support_status'].iloc[0]
        for rank,k in enumerate(indices,1):
            field=features[k];meta=metadata.loc[field]
            evidence=lit.loc[lit.target.eq(target)&lit.field.eq(field)]
            row=dict(method=method,target=target,rank=rank,field=field,
                chemical_name=repair_utf8_mojibake(meta.chemical_name),ranking_score=float(score[k]),
                ranking_scope=scope,ranking_measure='mean_absolute_OOF_SHAP' if method=='gbr_selected' else 'mean_absolute_standardized_training_coefficient',
                attribution_mean_absolute=float(abs(values[method][:,j,k]).mean()),
                outer_top3_frequency=float(frequency[k]),stability_denominator=p['outer_folds'],
                input_attribution_spearman=float(direction[k]),predictive_support_status=status,
                correlated_neighbors_ge2folds=';'.join(other for other,count in neighbors[field].items() if count>=2),
                chemical_evidence_status='unreviewed_new_target_field_pair',reference_unit=meta.unit,
                unit_scope='reference_only_not_certified_input_unit',
                measurement_type='chemical_family_IBU_indicator' if field=='bitterness' else 'individual_analyte_candidate',
                causal_effect_established=False,migration_control_validated=False,
                label_source='raw_balanced_individual_ratings',old_target_statistics_transferred=False)
            if len(evidence):
                row['chemical_evidence_status']='existing_task3_exact_pair_annotation_not_raw_target_validation'
                for col in lit.columns:
                    if col.startswith('lit_'):row[col]=evidence.iloc[0][col]
            result.append(row)
        return result
    for method,v in values.items():
        direction=attribution_direction(x,v);scores=[];chosen=[]
        for fold in range(p['outer_folds']):
            mask=folds==fold;absolute=abs(v[mask]).mean(0)
            score=absolute if method=='gbr_selected' else abs(coefficients[fold])
            scores.append(score)
            choices=[]
            for j,target in enumerate(targets):
                indices=nonzero_top(score[j],eligible);choices.append(set(indices))
                for k,f in enumerate(features):
                    fold_scores.append(dict(method=method,outer_fold=fold,target=target,field=f,ranking_score=score[j,k],
                        mean_absolute_attribution=absolute[j,k],standardized_coefficient=coefficients[fold,j,k] if method=='ridge_raw_mean' else np.nan))
                fold_top.extend(dict(method=method,outer_fold=fold,target=target,rank=r,field=features[k]) for r,k in enumerate(indices,1))
            chosen.append(choices)
            groups=groups_by_fold[fold];importance=module_importance(v[mask],groups)
            candidates=[g for g,members in enumerate(groups) if set(members)&set(eligible)]
            for j,target in enumerate(targets):
                top=nonzero_top(importance[j],candidates)
                for g,members in enumerate(groups):
                    module_rows.append(dict(method=method,outer_fold=fold,target=target,module_id=g,
                        mean_absolute_signed_sum=importance[j,g],n_features=len(members),
                        eligible_analytes=len(set(members)&set(eligible))))
                module_tops.extend(dict(method=method,outer_fold=fold,target=target,rank=r,module_id=int(g),
                    mean_absolute_signed_sum=importance[j,g],members=';'.join(features[k] for k in groups[g]),
                    interpretation='correlated_component_not_a_single_causal_compound') for r,g in enumerate(top,1))
        global_score=abs(v).mean(0) if method=='gbr_selected' else abs(coefficients).mean(0)
        for j,target in enumerate(targets):
            freq=np.array([sum(k in chosen[f][j] for f in range(p['outer_folds']))/p['outer_folds'] for k in range(len(features))])
            for k,field in enumerate(features):
                all_scores.append(dict(method=method,target=target,field=field,ranking_score=global_score[j,k],
                    mean_absolute_OOF_attribution=float(abs(v[:,j,k]).mean()),eligible_strict219=k in eligible,
                    outer_top3_frequency=freq[k],input_attribution_spearman=direction[j,k]))
            ranks.extend(enrich(method,j,nonzero_top(global_score[j],eligible),global_score[j],freq,direction[j],'strict_219'))
            family_ranks.extend(enrich(method,j,nonzero_top(global_score[j],family),global_score[j],freq,direction[j],'family_220'))
            for first in range(p['outer_folds']):
                for second in range(first+1,p['outer_folds']):
                    a,b=scores[first][j,eligible],scores[second][j,eligible]
                    rho=float(pd.Series(a).corr(pd.Series(b),method='spearman')) if np.ptp(a)>0 and np.ptp(b)>0 else np.nan
                    stability.append(dict(method=method,target=target,fold_a=first,fold_b=second,
                        top3_jaccard=jaccard(chosen[first][j],chosen[second][j]),importance_rank_spearman=rho,
                        identifiable_in_both=bool(chosen[first][j] and chosen[second][j])))
    ranks=pd.DataFrame(ranks);tables['top3_raw_all50.csv']=ranks
    tables['family_top3_supplement.csv']=pd.DataFrame(family_ranks)
    slots=[]
    for method in values:
        for target in d['scope'].target_id:
            for rank in [1,2,3]:
                hit=ranks.loc[ranks.method.eq(method)&ranks.target.eq(target)&ranks['rank'].eq(rank)]
                row=dict(method=method,target=target,rank=rank,slot_status='unidentified_no_raw_panel_label' if target not in targets else 'unidentified_zero_model_attribution')
                if len(hit):row.update(hit.iloc[0].to_dict());row['slot_status']='ranked_predictive_association'
                slots.append(row)
    tables['top3_raw_all52_slots.csv']=pd.DataFrame(slots)
    tables['all_factor_scores.csv.gz']=pd.DataFrame(all_scores);tables['fold_factor_scores.csv.gz']=pd.DataFrame(fold_scores)
    tables['fold_top3.csv']=pd.DataFrame(fold_top);tables['rank_stability.csv']=pd.DataFrame(stability)
    tables['fold_module_scores.csv.gz']=pd.DataFrame(module_rows);tables['fold_module_top3.csv']=pd.DataFrame(module_tops)
    comparison=[]
    for target in targets:
        previous=old.loc[old.target.eq(target),'field'].tolist()
        gbr=ranks.loc[ranks.target.eq(target)&ranks.method.eq('gbr_selected'),'field'].tolist()
        ridge=ranks.loc[ranks.target.eq(target)&ranks.method.eq('ridge_raw_mean'),'field'].tolist()
        comparison.append(dict(target=target,old_standardized_ridge_top3=';'.join(previous),raw_quantile_GBR_top3=';'.join(gbr),
            raw_mean_ridge_top3=';'.join(ridge),gbr_vs_raw_ridge_jaccard=jaccard(gbr,ridge),
            gbr_vs_old_jaccard=jaccard(gbr,previous),raw_ridge_vs_old_jaccard=jaccard(ridge,previous),
            comparison_scope='different_responses_and_estimands_not_accuracy_ranking',old_statistics_transferred=False))
    tables['historical_rank_comparison.csv']=pd.DataFrame(comparison)
    return tables


def run(args):
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'));bp=json.loads(BASE_PROTOCOL.read_text(encoding='utf-8'))
    d=load_inputs(args.source,args.raw,bp);x,q=d['x'],d['distribution'];styles=d['ids'].tasting_category_fine.to_numpy()
    output=args.output;output.mkdir(parents=True,exist_ok=False);(output/'models').mkdir()
    write_json(output/'protocol.json',p)
    n,t,f=len(x),len(d['targets']),x.shape[1]
    values={name:np.empty((n,t,f)) for name in ['gbr_selected','ridge_raw_mean']}
    bases={name:np.empty((n,t)) for name in values};continuous={name:np.empty((n,t)) for name in values}
    grades={name:np.empty((n,t),int) for name in ['training_median',*values]}
    coefficients=np.empty((p['outer_folds'],t,f));folds=np.empty(n,int)
    inner_rows=[];selection=[];fitlog=[];background_log=[];splitlog=[];edge_rows=[];member_rows=[]
    all_groups=[];all_edges=[];errors=[]
    def fit(train,name,seed,context):
        tick=time.perf_counter();print(f'Fit {context} {name} ({len(train)} beers)',flush=True)
        model=fit_raw_ridge(x[train],q[train],p['ridge_alpha'],seed) if name=='ridge_raw_mean' else fit_gbr(d,train,name,seed,bp)
        fitlog.append(dict(context=context,method=name,seed=seed,n_training_beers=len(train),seconds=time.perf_counter()-tick))
        return model
    for fold,(train,test) in enumerate(make_splits(styles,p['outer_folds'],p['outer_seed'])):
        folds[test]=fold
        inner_pred={name:np.empty((len(train),t),int) for name in p['gbr_candidates']}
        for inner,(it,iv) in enumerate(make_splits(styles[train],p['inner_folds'],p['inner_seed']+fold)):
            for role,index in [('train',train[it]),('validation',train[iv])]:
                splitlog.extend(dict(outer_fold=fold,inner_fold=inner,role=role,beer=int(d['ids'].beer.iloc[i])) for i in index)
            for name in p['gbr_candidates']:
                model=fit(train[it],name,p['model_seed']+fold*100+inner,f'outer{fold}_inner{inner}')
                pred=model.predict(x[train[iv]])['grade'];inner_pred[name][iv]=pred
                for row,i in enumerate(train[iv]):
                    inner_rows.extend(dict(outer_fold=fold,inner_fold=inner,method=name,beer=int(d['ids'].beer.iloc[i]),
                        target=target,estimated_grade=int(pred[row,j])) for j,target in enumerate(d['targets']))
        chosen,losses=select_gbr(inner_pred,q[train],p['gbr_candidates'])
        selection.extend(dict(outer_fold=fold,candidate=name,inner_raw_mae=loss,selected=name==chosen) for name,loss in losses.items())
        gbr=fit(train,chosen,p['model_seed']+fold*100+99,f'outer{fold}_final')
        ridge=fit(train,'ridge_raw_mean',p['model_seed']+fold*100+99,f'outer{fold}_final')
        rng=np.random.default_rng(p['background_seed']+fold)
        background=np.sort(rng.choice(train,size=min(p['background_size'],len(train)),replace=False))
        background_log.extend(dict(outer_fold=fold,beer=int(d['ids'].beer.iloc[i])) for i in background)
        for name,model,explainer in [('gbr_selected',gbr,explain_quantile),('ridge_raw_mean',ridge,explain_linear)]:
            print(f'Explain fold {fold}: {name}',flush=True)
            v,b,c=explainer(model,x[test],x[background]);error=check_additivity(v,b,c)
            values[name][test]=v;bases[name][test]=b;continuous[name][test]=c
            grades[name][test]=decision(c,d['parents'])
            errors.append(dict(outer_fold=fold,method=name,max_additivity_error=error))
        coefficients[fold]=ridge.named_steps['model'].coef_
        reference=np.broadcast_to(q[train].mean(0),(len(test),t,7))
        grades['training_median'][test]=decision(median_grades(reference),d['parents'])
        groups,edges=training_modules(gbr.imputer.transform(x[train]),p['correlation_threshold'])
        all_groups.append(groups);all_edges.append(edges)
        edge_rows.extend(dict(outer_fold=fold,source=d['features'][i],target=d['features'][j],spearman_rho=r) for i,j,r in edges)
        member_rows.extend(dict(outer_fold=fold,module_id=g,field=d['features'][i]) for g,members in enumerate(groups) for i in members)
        joblib.dump(dict(gbr=gbr,ridge=ridge,chosen=chosen,train=train,test=test,background=background),output/'models'/f'outer{fold}.joblib',compress=3)
    tables=assemble(d,p,folds,grades,values,bases,continuous,coefficients,all_groups,all_edges)
    for name,table in tables.items():
        if name.endswith('.gz'):table.to_csv(output/name,index=False,compression={'method':'gzip','mtime':0})
        else:save_csv(output,name,table)
    for name,rows in [('inner_predictions.csv.gz',inner_rows),('inner_selection.csv',selection),('fit_log.csv',fitlog),
        ('inner_splits.csv',splitlog),('background_beers.csv',background_log),('training_correlation_edges.csv',edge_rows),
        ('training_module_members.csv',member_rows),('additivity_checks.csv',errors)]:
        if name.endswith('.gz'):pd.DataFrame(rows).to_csv(output/name,index=False,compression={'method':'gzip','mtime':0})
        else:save_csv(output,name,rows)
    save_csv(output,'outer_folds.csv',pd.DataFrame(dict(beer=d['ids'].beer,outer_fold=folds)))
    for name in values:
        np.savez_compressed(output/f'{name}_OOF_attributions.npz',values=values[name],base_values=bases[name],
            continuous=continuous[name],beer=d['ids'].beer.to_numpy(),targets=np.array(d['targets']),features=np.array(d['features']))
    summary=dict(status='completed',protocol_content_sha256=content_hash(p),baseline_commit=p['baseline_commit'],
        source_sha256={name:sha256(args.source/name) for name in ['1.csv','4.csv','8.xlsx']},raw_source_sha256=d['manifest']['sha256'],
        input_lf_normalized_sha256={path.relative_to(ROOT).as_posix():text_hash(path) for path in [BASE_PROTOCOL,OLD,LITERATURE,META,ELIGIBILITY]},
        packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','shap','numba','llvmlite']},
        n_beers=n,n_targets=t,n_features=f,n_eligible=219,base_fits=len(fitlog),
        selected_GBR_depths=[row['candidate'] for row in selection if row['selected']],
        overall_raw_mae={name:float(rating_loss(q,g).mean()) for name,g in grades.items()},
        supported_targets={name:int(((tables['predictive_metrics.csv'].method==name)&tables['predictive_metrics.csv'].support_status.eq('historical_predictive_support_not_causal')).sum()) for name in values},
        max_additivity_error=max(row['max_additivity_error'] for row in errors),
        rank_rows=len(tables['top3_raw_all50.csv']),causal_effect_established=False,migration_control_validated=False,
        historical_Q2_replaced=False,Q1_model_replaced=False,task4_started=False,
        attribution_sha256={f'{name}_OOF_attributions.npz':sha256(output/f'{name}_OOF_attributions.npz') for name in values})
    write_json(output/'summary.json',summary);plot_results(output)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table=pd.read_csv(output/'top3_raw_all50.csv');table=table[table.method.eq('gbr_selected')]
    targets=['bitternes','A_malt_all','A_esters_all','diacetyl','X4vg','stale_hops']
    fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
    for ax,target in zip(axes.ravel(),targets):
        rows=table[table.target.eq(target)].sort_values('rank',ascending=False)
        ax.set_title(target)
        if len(rows):
            ax.barh(np.arange(len(rows)),rows.ranking_score)
            ax.set_yticks(np.arange(len(rows)),rows.chemical_name,fontsize=8)
        else:ax.text(.5,.5,'No nonzero attribution rank',ha='center',transform=ax.transAxes)
        ax.set_xlabel('Mean |held-out SHAP| (raw score units)')
    fig.suptitle('Raw quantile GBR explanations; associations, not causal effects')
    fig.savefig(output/'raw_explanation_examples.png',dpi=160);fig.savefig(output/'raw_explanation_examples.svg')
    svg=output/'raw_explanation_examples.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--source',type=Path,required=True);result.add_argument('--output',type=Path,required=True)
    result.add_argument('--raw',type=Path,default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):run(parser().parse_args())
