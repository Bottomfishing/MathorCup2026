"""Independently recompute summaries and replay selected refits and all Q2 CV fits."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import write_json,sha256
from beer_models.baseline import load_data,make_pipeline,make_splits
from beer_models.final_model import estimator
from beer_models.style_models import make_stale_model


def verify(source,q2,q3):
    config=json.loads(Path('configs/review_followup_20261025.json').read_text())
    ids,xf,yf,primary=load_data(source,Path('outputs/p0_audit_20261023_final'))
    styles=ids.tasting_category_fine.to_numpy();y=yf.to_numpy(float);n=config['bootstrap_repeats']
    expected=[];rng=np.random.default_rng(config['seed'])
    for _ in range(n):
        expected.append(np.concatenate([rng.choice(np.where(styles==s)[0],sum(styles==s),replace=True) for s in sorted(set(styles))]))
    draws=np.asarray(expected)
    for folder in [q2,q3]:
        np.testing.assert_array_equal(pd.read_csv(folder/'bootstrap_draws.csv').to_numpy(),draws)
        summary=json.loads((folder/'summary.json').read_text())
        assert summary['config_sha256']==sha256(Path('configs/review_followup_20261025.json'))
        for name,digest in summary['source_sha256'].items():assert sha256(source/name)==digest
    excluded=json.loads(Path('configs/q2_exploratory_v1.json').read_text())['exclude_from_compound_top3']
    scopes={'all231':list(xf),'strict219':[c for c in xf if c not in excluded]}
    selections=pd.read_csv(q2/'bootstrap_selections.csv')
    assert len(selections)==n*2*50*3
    assert not selections.duplicated(['repeat','scope','target','rank']).any()
    assert selections.groupby(['repeat','scope','target']).field.nunique().eq(3).all()
    assert set(selections.field)<=set(scopes['strict219'])
    top=pd.read_csv(q2/'top3_stability.csv');comparisons=pd.read_csv(q2/'scope_comparison.csv').set_index('target')
    import networkx as nx
    graph=nx.Graph();graph.add_nodes_from(scopes['strict219'])
    edges=pd.read_csv('outputs/q2_task4_20261025/network_edges_0_6.csv')
    graph.add_edges_from((r.source,r.target) for r in edges.itertuples() if r.source in graph and r.target in graph)
    groups=pd.read_csv(q2/'network_groups.csv').set_index('field')
    for field,row in groups.iterrows():
        assert set(groups.index[groups.component.eq(row.component)])==nx.node_connected_component(graph,field)
        assert row.component_size==len(nx.node_connected_component(graph,field))
    assert selections.component.tolist()==selections.field.map(groups.component).tolist()
    replayed=0
    for name,columns in scopes.items():
        for repeat in [-1,0,n-1]:
            draw=np.arange(len(xf)) if repeat==-1 else draws[repeat]
            model,_=make_pipeline('ridge_asinh',{'ridge_alphas':[config['ridge_alpha']]},config['seed'])
            model.set_params(model__alpha=config['ridge_alpha']).fit(xf[columns].to_numpy()[draw],y[draw])
            c=model.named_steps['model'].coef_
            saved=top.loc[top.scope.eq(name)] if repeat==-1 else selections.loc[selections.scope.eq(name)&selections.repeat.eq(repeat)]
            for j,target in enumerate(yf):
                eligible=[columns.index(f) for f in scopes['strict219']]
                ordered=sorted(eligible,key=lambda k:-abs(c[j,k]))[:3]
                actual=saved.loc[saved.target.eq(target)].sort_values('rank')
                assert actual.field.tolist()==[columns[k] for k in ordered]
                np.testing.assert_allclose(actual.coefficient,c[j,ordered],atol=1e-12,rtol=0)
            replayed+=1
    for row in top.itertuples():
        chosen=selections.loc[selections.scope.eq(row.scope)&selections.target.eq(row.target)]
        np.testing.assert_allclose(row.frequency,chosen.field.eq(row.field).sum()/n,atol=1e-14)
        np.testing.assert_allclose(row.component_any_selected_frequency,chosen.loc[chosen.component.eq(row.component),'repeat'].nunique()/n,atol=1e-14)
    for target,row in comparisons.iterrows():
        a=set(top.loc[top.scope.eq('all231')&top.target.eq(target),'field']);b=set(top.loc[top.scope.eq('strict219')&top.target.eq(target),'field'])
        assert row.all231==';'.join(top.loc[top.scope.eq('all231')&top.target.eq(target)].sort_values('rank').field)
        assert row.strict219==';'.join(top.loc[top.scope.eq('strict219')&top.target.eq(target)].sort_values('rank').field)
        np.testing.assert_allclose(row.top3_jaccard,len(a&b)/len(a|b))
        a={int(groups.loc[f,'component']) for f in a};b={int(groups.loc[f,'component']) for f in b}
        np.testing.assert_allclose(row.component_jaccard,len(a&b)/len(a|b))
    predictions=pd.read_csv(q2/'oof_predictions.csv',dtype={'beer':str})
    assert len(predictions)==3*2*250*50
    assert not predictions.duplicated(['seed','scope','beer','target']).any()
    splits=pd.read_csv(q2/'splits.csv',dtype={'beer':str})
    for seed in config['split_seeds']:
        for fold,(train,test) in enumerate(make_splits(styles,3,seed)):
            saved=splits.loc[splits.seed.eq(seed)&splits.fold.eq(fold)]
            assert saved.beer.tolist()==ids.beer.iloc[test].astype(str).tolist()
            for name,columns in scopes.items():
                model,_=make_pipeline('ridge_asinh',{'ridge_alphas':[config['ridge_alpha']]},config['seed'])
                model.set_params(model__alpha=config['ridge_alpha']).fit(xf[columns].to_numpy()[train],y[train])
                part=predictions.loc[predictions.seed.eq(seed)&predictions.fold.eq(fold)&predictions.scope.eq(name)]
                ordered=part.pivot(index='beer',columns='target',values='prediction').loc[ids.beer.iloc[test],yf.columns]
                truth=part.pivot(index='beer',columns='target',values='observed').loc[ids.beer.iloc[test],yf.columns]
                np.testing.assert_allclose(ordered,model.predict(xf[columns].to_numpy()[test]),atol=1e-12,rtol=0)
                np.testing.assert_allclose(truth,y[test],atol=1e-12,rtol=0);replayed+=1
    for row in pd.read_csv(q2/'pooled_metrics.csv').itertuples():
        part=predictions.loc[predictions.seed.eq(row.seed)&predictions.scope.eq(row.scope)]
        if row.targets=='primary45':part=part.loc[part.target.isin(primary)]
        np.testing.assert_allclose(row.mae,abs(part.observed-part.prediction).mean(),atol=1e-12)
    q2_report=dict(status='passed',replayed_ridge_fits=replayed,bootstrap_rows=len(selections),oof_rows=len(predictions),
        checks=['paired draws','full original and strict rankings','first and last paired refits','all 18 repeated-CV fits and truths','all top frequencies and components','all scope overlaps','pooled errors'])
    write_json(q2/'verification.json',q2_report)
    final=json.loads(Path('configs/final_protocol.json').read_text());style=json.loads(Path('configs/q3_style_protocol.json').read_text())
    columns=[f for f in xf if f not in final['excluded_derived_features']];x=xf[columns].to_numpy();j=yf.columns.get_loc('stale_hops')
    before=pd.read_csv(q3/'frozen_before.csv',dtype={'beer':str}).set_index('beer')
    after=pd.read_csv(q3/'frozen_after.csv',dtype={'beer':str}).set_index('beer')
    for filename,digest in json.loads((q3/'summary.json').read_text())['frozen_inputs_sha256'].items():
        assert sha256(Path('outputs/q3_task3_20261025')/filename)==digest
    for frozen,current in [(before,'transfer_source_profiles.csv'),(after,'transfer_profiles.csv')]:
        original=pd.read_csv(Path('outputs/q3_task3_20261025')/current,dtype={'beer':str}).set_index('beer').loc[frozen.index,columns]
        np.testing.assert_allclose(frozen,original,atol=1e-12)
    changes=pd.read_csv(q3/'refit_changes.csv',dtype={'beer':str});oob=pd.read_csv(q3/'oob_predictions.csv',dtype={'beer':str})
    assert len(changes)==n*3*11 and not changes.duplicated(['repeat','model','beer']).any()
    np.testing.assert_allclose(changes.change,changes.after-changes.before,atol=1e-12)
    assert np.array_equal(changes.passes,changes.change.le(config['nonincrease_tolerance']))
    for row in pd.read_csv(q3/'change_summary.csv',dtype={'beer':str}).itertuples():
        values=changes.loc[changes.model.eq(row.model)&changes.beer.eq(row.beer),'change'].to_numpy()
        np.testing.assert_allclose([row.pass_fraction,row.median,row.lower,row.upper],
            [(values<=config['nonincrease_tolerance']).mean(),np.median(values),np.quantile(values,.025),np.quantile(values,.975)],atol=1e-12)
    for row in pd.read_csv(q3/'joint_pass_fraction.csv',dtype={'beer':str}).itertuples():
        part=changes.loc[changes.beer.eq(row.beer)].pivot(index='repeat',columns='model',values='change')
        assert part.shape==(n,3)
        np.testing.assert_allclose(row.all_three_pass_fraction,np.mean(np.max(part,axis=1)<=config['nonincrease_tolerance']))
    metrics=pd.read_csv(q3/'oob_metrics.csv')
    for (repeat,name),part in oob.groupby(['repeat','model']):
        held=np.setdiff1d(np.arange(len(x)),draws[repeat]);assert part.beer.tolist()==ids.beer.iloc[held].astype(str).tolist()
        np.testing.assert_allclose(part.observed,y[held,j],atol=1e-12)
        np.testing.assert_allclose(part.median_baseline,np.median(y[draws[repeat],j]),atol=1e-12)
        row=metrics.loc[metrics.repeat.eq(repeat)&metrics.model.eq(name)].iloc[0]
        np.testing.assert_allclose([row.mae,row.median_mae],[abs(part.observed-part.prediction).mean(),abs(part.observed-part.median_baseline).mean()],atol=1e-12)
    for repeat in [0,n-1]:
        seed=config['seed']+repeat
        models={'multioutput':estimator('extra_trees',final,seed),'ridge':estimator('ridge_asinh',final,seed),'focused':make_stale_model(style,seed)}
        for name,model in models.items():
            draw=draws[repeat];model.fit(x[draw],y[draw,j] if name=='focused' else y[draw])
            part=changes.loc[changes.repeat.eq(repeat)&changes.model.eq(name)].set_index('beer').loc[before.index]
            for frame,field in [(before,'before'),(after,'after')]:
                pred=model.predict(frame.to_numpy());pred=pred if name=='focused' else pred[:,j]
                np.testing.assert_allclose(part[field],pred,atol=1e-12,rtol=0)
            held=np.setdiff1d(np.arange(len(x)),draw);pred=model.predict(x[held]);pred=pred if name=='focused' else pred[:,j]
            np.testing.assert_allclose(oob.loc[oob.repeat.eq(repeat)&oob.model.eq(name),'prediction'],pred,atol=1e-12,rtol=0)
    write_json(q3/'verification.json',dict(status='passed',fixed_paths=11,refit_change_rows=len(changes),replayed_models=6,
        checks=['candidate hashes and coordinates','all paired draws','all direction and joint summaries','all OOB partitions truths and errors','first and last three model refits'],
        actual_after_stale_validated=False))
    print('Q2 and Q3 diagnostic verification passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--q2',type=Path,required=True);p.add_argument('--q3',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):verify(a.source,a.q2,a.q3)
