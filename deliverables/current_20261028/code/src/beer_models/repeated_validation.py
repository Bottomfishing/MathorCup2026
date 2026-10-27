"""Summaries for repeated beer-grouped validation, without pseudo-replication."""
import numpy as np
import pandas as pd
from .explain import stratified_draw
from .raw_explain import nonzero_top


def beer_averaged_interval(repeated_differences,styles,repeats,seed,quantiles):
    values=np.asarray(repeated_differences,float)
    if values.ndim<2:raise ValueError('Expected repeat by beer differences')
    per_beer=values.mean(axis=0)
    if len(per_beer)!=len(styles):raise ValueError('Beer and style axes differ')
    rng=np.random.default_rng(seed)
    draws=np.array([per_beer[stratified_draw(styles,rng)].mean(axis=0) for _ in range(repeats)])
    return per_beer.mean(axis=0),*np.quantile(draws,quantiles,axis=0)


def repeat_summary(table,keys,metrics):
    rows=[]
    for names,group in table.groupby(keys,sort=False,dropna=False):
        if not isinstance(names,tuple):names=(names,)
        for metric in metrics:
            values=group[metric].to_numpy(float)
            rows.append(dict(zip(keys,names))|dict(metric=metric,n_repeats=len(values),mean=float(values.mean()),
                sample_sd=float(values.std(ddof=1)) if len(values)>1 else np.nan,minimum=float(values.min()),maximum=float(values.max())))
    return pd.DataFrame(rows)


def overlap(left,right):
    left,right=set(left),set(right)
    return len(left&right)/len(left|right) if left|right else np.nan


def rank_tables(values,coefficients,folds,features,targets,scope,eligible,family):
    rows=[];family_rows=[];slots=[];fold_scores=[];fold_top=[];stability=[];counts=[];global_scores=[]
    repeat_top={};n_repeats=len(values)
    for repeat in range(n_repeats):
        for method in ['gbr_selected','ridge_raw_mean']:
            score=abs(values[repeat]).mean(0) if method=='gbr_selected' else abs(coefficients[repeat]).mean(0)
            fold_sets=[]
            for fold in range(coefficients.shape[1]):
                fs=abs(values[repeat,folds[repeat]==fold]).mean(0) if method=='gbr_selected' else abs(coefficients[repeat,fold])
                chosen=[]
                for j,target in enumerate(targets):
                    chosen.append(set(nonzero_top(fs[j],eligible)))
                    for k,field in enumerate(features):fold_scores.append(dict(repeat=repeat,outer_fold=fold,method=method,target=target,field=field,importance=float(fs[j,k])))
                    fold_top.extend(dict(repeat=repeat,outer_fold=fold,method=method,target=target,rank=r,field=features[k]) for r,k in enumerate(nonzero_top(fs[j],eligible),1))
                fold_sets.append(chosen)
            top={}
            for j,target in enumerate(targets):
                indices=nonzero_top(score[j],eligible);top[target]=set(indices)
                for k,field in enumerate(features):
                    global_scores.append(dict(repeat=repeat,method=method,target=target,field=field,importance=float(score[j,k]),eligible_strict219=k in eligible))
                def records(pool,label):
                    return [dict(repeat=repeat,method=method,target=target,rank=r,field=features[k],importance=float(score[j,k]),
                        rank_measure='mean_absolute_OOF_SHAP' if method=='gbr_selected' else 'mean_absolute_standardized_training_coefficient',
                        fold_top3_frequency=sum(k in g[j] for g in fold_sets)/len(fold_sets),ranking_scope=label,
                        causal_effect_established=False,migration_control_validated=False) for r,k in enumerate(nonzero_top(score[j],pool),1)]
                rows.extend(records(eligible,'strict219'));family_rows.extend(records(family,'family220'))
                for a in range(len(fold_sets)):
                    for b in range(a+1,len(fold_sets)):
                        stability.append(dict(scope='within_repeat_outer_refits',repeat_a=repeat,repeat_b=repeat,fold_a=a,fold_b=b,
                            method=method,target=target,top3_jaccard=overlap(fold_sets[a][j],fold_sets[b][j]),
                            nonempty_both=bool(fold_sets[a][j] and fold_sets[b][j])))
            repeat_top[(repeat,method)]=top
            method_rows=[r for r in rows if r['repeat']==repeat and r['method']==method]
            counts.append(dict(repeat=repeat,method=method,nonzero_rank_targets=len({r['target'] for r in method_rows}),
                identified_rank_rows=len(method_rows),zero_effect_slots=3*len(targets)-len(method_rows),unlabelled_slots=6))
            lookup={(r['target'],r['rank']):r for r in method_rows}
            for target in scope.target_id:
                for rank in [1,2,3]:
                    row=dict(repeat=repeat,method=method,target=target,rank=rank,field=None,importance=np.nan,
                        slot_status='no_raw_panel_label' if target not in targets else 'no_nonzero_model_effect')
                    if (target,rank) in lookup:row.update(lookup[(target,rank)]);row['slot_status']='ranked_exploratory_association'
                    slots.append(row)
    for a in range(n_repeats):
        for b in range(a+1,n_repeats):
            for method in ['gbr_selected','ridge_raw_mean']:
                for target in targets:
                    left,right=repeat_top[(a,method)][target],repeat_top[(b,method)][target]
                    stability.append(dict(scope='between_repeat_global_ranks',repeat_a=a,repeat_b=b,fold_a=None,fold_b=None,
                        method=method,target=target,top3_jaccard=overlap(left,right),nonempty_both=bool(left and right)))
    return {'q2_repeat_top3.csv':pd.DataFrame(rows),'q2_family_top3_supplement.csv':pd.DataFrame(family_rows),
        'q2_all52_rank_slots.csv':pd.DataFrame(slots),'q2_fold_importance.csv.gz':pd.DataFrame(fold_scores),
        'q2_repeat_importance.csv.gz':pd.DataFrame(global_scores),'q2_fold_top3.csv':pd.DataFrame(fold_top),
        'q2_rank_stability.csv':pd.DataFrame(stability),'q2_rank_coverage.csv':pd.DataFrame(counts)}
