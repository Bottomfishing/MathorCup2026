"""Q2 held-out feature/group permutation. AI-assisted: OpenAI Codex, 2026-10-24.

All feature/group ranking here measures the saved model's reliance, not causality.
Validation bootstrap keeps fitted models fixed; do not mix it with refit bootstrap.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
from threadpoolctl import threadpool_limits

from beer_audit.audit import save_csv,write_json,sha256
from beer_audit.checks import pairwise_spearman
from .baseline import load_data,make_pipeline,metric_vectors
from .explain import stratified_draw


def correlation_groups(training,threshold):
    """Complete linkage ensures every pair within a group exceeds the cutoff."""
    if not 0<threshold<1:raise ValueError('Invalid grouping threshold')
    rho=pairwise_spearman(pd.DataFrame(training)).fillna(0).to_numpy()
    rho=np.clip((rho+rho.T)/2,-1,1);np.fill_diagonal(rho,1.)
    distance=np.clip(1-abs(rho),0,1);np.fill_diagonal(distance,0.)
    tree=linkage(squareform(distance,checks=True),method='complete')
    labels=fcluster(tree,t=1-threshold,criterion='distance')
    groups=[np.flatnonzero(labels==g).tolist() for g in sorted(set(labels))]
    return sorted(groups,key=lambda group:group[0]),rho


def permuted_copy(x,indices,order):
    result=x.copy()
    result[:,indices]=x[np.asarray(order)[:,None],np.asarray(indices)[None,:]]
    return result


def permutation_orders(n,repeats,seed,strata=None):
    orders=[]
    if strata is not None and len(strata)!=n:raise ValueError('Strata length mismatch')
    for r in range(repeats):
        rng=np.random.default_rng(seed+r)
        if strata is None:order=rng.permutation(n)
        else:
            order=np.arange(n);strata=np.asarray(strata)
            for label in sorted(set(strata)):
                indices=np.flatnonzero(strata==label);order[indices]=rng.permutation(indices)
        orders.append(order)
    return np.array(orders)


def permutation_deltas(model,x,y,groups,repeats,seed,batch_groups=16,strata=None):
    """Return mean per-sample loss increments and per-repeat mean increments.

    Each repeat uses the same row permutation across groups. Joint permutation
    preserves relationships inside a group. This is unconditional permutation.
    """
    if repeats<1 or batch_groups<1 or not groups:raise ValueError('Invalid permutation request')
    reference=model.predict(x)
    if reference.shape!=y.shape or not np.isfinite(reference).all():raise ValueError('Invalid prediction')
    loss=abs(y-reference)
    sample_delta=np.zeros((len(x),len(groups),y.shape[1]))
    repeat_means=np.zeros((repeats,len(groups),y.shape[1]))
    orders=permutation_orders(len(x),repeats,seed,strata)
    for r,order in enumerate(orders):
        for start in range(0,len(groups),batch_groups):
            chunk=groups[start:start+batch_groups]
            stacked=np.concatenate([permuted_copy(x,g,order) for g in chunk])
            predictions=model.predict(stacked).reshape(len(chunk),len(x),y.shape[1])
            if not np.isfinite(predictions).all():raise ValueError('Invalid permuted prediction')
            delta=abs(y[None,:,:]-predictions)-loss[None,:,:]
            sample_delta[:,start:start+len(chunk),:]+=delta.transpose(1,0,2)/repeats
            repeat_means[r,start:start+len(chunk),:]=delta.mean(axis=1)
    return sample_delta,repeat_means,orders


def ranked_indices(values,k=3):
    """Rank signed loss increases; negative importance is never made positive."""
    values=np.asarray(values,float)
    if not np.isfinite(values).all() or len(values)<k:raise ValueError('Invalid importance values')
    return np.argsort(-values,kind='stable')[:k]


def validation_bootstrap(deltas,styles,repeats,seed):
    rng=np.random.default_rng(seed)
    draws=np.array([stratified_draw(styles,rng) for _ in range(repeats)])
    means=np.array([deltas[draw].mean(axis=0) for draw in draws])
    counts=np.zeros(deltas.shape[1:],dtype=int)
    for mean in means:
        for t in range(mean.shape[1]):counts[ranked_indices(mean[:,t]),t]+=1
    return means,counts,draws


def run(source,audit,baseline,previous,config_path,output):
    source,audit,baseline,previous,output=[p.resolve() for p in [source,audit,baseline,previous,output]]
    if source==output or source in output.parents:raise ValueError('Output inside original source')
    config=json.loads(config_path.read_text(encoding='utf-8'))
    ids,xf,yf,primary=load_data(source,audit)
    x,y=xf.to_numpy(float),yf.to_numpy(float)
    split=pd.read_csv(baseline/'split_manifest.csv',dtype={'beer':str}).set_index('beer').loc[ids.beer]
    folds=split.outer_test_fold.to_numpy()
    if set(folds)!={0,1,2}:raise ValueError('Unexpected fold protocol')
    saved_features=pd.read_csv(baseline/'feature_columns.csv')['field'].tolist()
    saved_targets=pd.read_csv(baseline/'target_columns.csv')['field'].tolist()
    if saved_features!=list(xf) or saved_targets!=list(yf):raise ValueError('Baseline feature/target order mismatch')
    prior_manifest=json.loads((baseline/'run_manifest.json').read_text(encoding='utf-8'))
    for name,digest in prior_manifest['source_sha256'].items():
        if sha256(source/name)!=digest:raise ValueError('Baseline source differs')
    fitlog=pd.read_csv(baseline/'fit_log.csv')
    forest_config=json.loads((baseline/'config.json').read_text(encoding='utf-8'))
    eligible_table=pd.read_csv(previous/'factor_eligibility.csv')
    eligible_fields=eligible_table.loc[eligible_table.eligible_compound,'field'].tolist()
    eligible=[list(xf).index(c) for c in eligible_fields]
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',config)
    write_json(output/'status.json',{'state':'running'})
    project=Path(__file__).resolve().parents[2]
    for p in sorted((project/'src').rglob('*.py')):
        dest=output/'source_snapshot'/p.relative_to(project);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    shutil.copy2(config_path,output/'source_snapshot/config.json')
    shutil.copy2(project/'requirements-lock.txt',output/'source_snapshot/requirements-lock.txt')
    save_csv(output,'samples.csv',ids.assign(outer_test_fold=folds))
    save_csv(output,'target_order.csv',[{'target':c} for c in yf])
    all_metric_rows=[];summaries=[]
    for variant in config['variants']:
        folder=output/variant;folder.mkdir()
        columns=list(range(x.shape[1])) if variant=='full' else eligible
        if variant not in ['full','analytes_only']:raise ValueError('Unknown feature variant')
        names=[xf.columns[c] for c in columns]
        feature_delta=np.full((len(x),len(names),y.shape[1]),np.nan)
        oof=np.full_like(y,np.nan);fold_delta=[];group_rows=[];group_members=[];fold_log=[]
        for fold in sorted(set(folds)):
            train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
            xx=x[:,columns]
            if variant=='full':
                path=baseline/'models'/f'extra_trees_fold{fold}.joblib'
                digest=fitlog.loc[fitlog.model.eq('extra_trees')&fitlog.fold.eq(fold),'model_sha256'].iloc[0]
                if sha256(path)!=digest:raise ValueError('Saved model hash differs')
                model=joblib.load(path)
            else:
                model,_=make_pipeline('extra_trees',forest_config,forest_config['seed']+int(fold))
                model.fit(xx[train],y[train]);path=folder/f'model_fold{fold}.joblib';joblib.dump(model,path,compress=3)
            np.testing.assert_allclose(model.named_steps['impute'].statistics_,np.nanmedian(xx[train],axis=0),rtol=1e-12,atol=1e-12)
            oof[test]=model.predict(xx[test])
            seed=config['seed']+int(fold)*1000
            print(f'{variant} fold {fold}: individual permutation ({len(names)} features)',flush=True)
            delta,repeats,orders=permutation_deltas(model,xx[test],y[test],[[j] for j in range(len(names))],
                                                   config['permutation_repeats'],seed,config['prediction_batch_groups'])
            feature_delta[test]=delta
            np.save(folder/f'permutation_orders_fold{fold}.npy',orders,allow_pickle=False)
            np.save(folder/f'individual_repeat_means_fold{fold}.npy',repeats,allow_pickle=False)
            for j,name in enumerate(names):
                for t,target in enumerate(yf):fold_delta.append({'fold':int(fold),'field':name,'target':target,'delta_mae':float(delta[:,j,t].mean())})
            groups,rho=correlation_groups(xx[train],config['group_abs_spearman_threshold'])
            multi=[g for g in groups if len(g)>1]
            print(f'{variant} fold {fold}: joint permutation ({len(multi)} correlated groups)',flush=True)
            if multi:
                gd,gr,_=permutation_deltas(model,xx[test],y[test],multi,config['permutation_repeats'],seed,config['prediction_batch_groups'])
                np.save(folder/f'group_repeat_means_fold{fold}.npy',gr,allow_pickle=False)
                for gi,group in enumerate(multi):
                    for member in group:group_members.append({'fold':int(fold),'group':gi,'field':names[member],'group_size':len(group)})
                    for t,target in enumerate(yf):
                        group_rows.append({'fold':int(fold),'group':gi,'target':target,'members':';'.join(names[j] for j in group),
                                           'group_size':len(group),'joint_delta_mae':float(gd[:,gi,t].mean()),
                                           'sum_individual_delta_mae':float(delta[:,group,t].sum(axis=1).mean()),
                                           'min_pair_abs_train_rho':float(np.min(abs(rho[np.ix_(group,group)])[np.triu_indices(len(group),1)]))})
            np.save(folder/f'train_correlations_fold{fold}.npy',rho,allow_pickle=False)
            fold_log.append({'fold':int(fold),'model_path':str(path),'model_sha256':sha256(path),'n_train':len(train),'n_test':len(test),
                             'correlation_groups':len(groups),'multi_member_groups':len(multi)})
            write_json(output/'status.json',{'state':'running','variant':variant,'last_completed_fold':int(fold)})
        if not np.isfinite(feature_delta).all():raise ValueError('Incomplete loss deltas')
        metrics=metric_vectors(y,oof)
        all_metric_rows.extend({'variant':variant,'target':target,'primary':target in primary,**{k:v[t] for k,v in metrics.items()}} for t,target in enumerate(yf))
        np.save(folder/'oof_predictions.npy',oof,allow_pickle=False)
        np.save(folder/'individual_sample_deltas.npy',feature_delta,allow_pickle=False)
        save_csv(folder,'feature_order.csv',[{'field':c} for c in names])
        save_csv(folder,'fold_importance.csv',fold_delta);save_csv(folder,'correlated_groups.csv',group_members)
        save_csv(folder,'group_importance.csv',group_rows);save_csv(folder,'fold_models.csv',fold_log)
        choose=[j for j,c in enumerate(names) if c in eligible_fields]
        means,counts,draws=validation_bootstrap(feature_delta[:,choose,:],ids.tasting_category_fine,
                                               config['validation_bootstrap_repeats'],config['seed']+9000)
        np.save(folder/'validation_bootstrap_means.npy',means,allow_pickle=False)
        np.save(folder/'validation_bootstrap_draws.npy',draws,allow_pickle=False)
        average=feature_delta[:,choose,:].mean(axis=0)
        lo,hi=np.quantile(means,[.025,.975],axis=0)
        table=[];top=[]
        for t,target in enumerate(yf):
            selected=ranked_indices(average[:,t]);rankmap={j:r+1 for r,j in enumerate(selected)}
            for local,j in enumerate(choose):
                record={'target':target,'field':names[j],'delta_mae':average[local,t],
                        'validation_resample_top3_frequency':counts[local,t]/config['validation_bootstrap_repeats'],
                        'fixed_model_resample_p025':lo[local,t],'fixed_model_resample_p975':hi[local,t],
                        'positive_mean':bool(average[local,t]>0),'primary':target in primary}
                table.append(record)
                if local in rankmap:top.append(record|{'rank':rankmap[local],'model_target_r2':metrics['r2'][t]})
        save_csv(folder,'individual_importance.csv',table);save_csv(folder,'top3_permutation.csv',top)
        summaries.append({'variant':variant,'n_features':len(names),'median_top3_validation_frequency':float(pd.DataFrame(top).validation_resample_top3_frequency.median()),
                          'primary_macro_mae':float(metrics['mae'][[c in primary for c in yf]].mean()),
                          'primary_macro_r2':float(np.nanmean(metrics['r2'][[c in primary for c in yf]]))})
    save_csv(output,'predictive_metrics.csv',all_metric_rows);save_csv(output,'summary.csv',summaries)
    write_json(output/'run_manifest.json',{'finished_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),'baseline':str(baseline),
               'previous_q2':str(previous),'config_sha256':sha256(config_path),'source_sha256':prior_manifest['source_sha256'],
               'limitations':['Extra Trees chosen after P1 exploration; not independent model selection confirmation.',
                              'Unconditional permutations can produce unrealistic combinations; grouped permutation only preserves within-group dependence.',
                              'Fixed-model validation resampling measures sample sensitivity; not training/model-selection uncertainty and not causal confidence intervals.',
                              'Training-only groups differ by fold and their importances must not be naively pooled as identical groups.',
                              'Signed importance is error increase, not the direction of a chemical effect.']})
    write_json(output/'status.json',{'state':'completed','variants':config['variants']})
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ['source','audit','baseline','previous','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/q2_permutation_v2.json'))
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.source,args.audit,args.baseline,args.previous,args.config,args.output)

if __name__=='__main__':main()
