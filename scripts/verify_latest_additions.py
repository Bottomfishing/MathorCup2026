"""Recheck rebuilt ordinal grades and Q2 randomization results.

AI-assisted: Codex, 2026-10-24. No original-grade or causal claim.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import write_json
from beer_models.baseline import load_data,make_pipeline


def verify(source,root):
    q1=root/'modeling/q1';ordinal=root/'ordinal';q2=root/'q2_reliability'
    comparisons=[]
    for new,old,files in [(ordinal,Path('outputs/q1_ordinal_delivery_20261024'),
                          ['model_comparison.csv','metrics_by_target.csv','metrics_by_observed_grade.csv','oof_predictions.csv','missingness.csv','paired_uncertainty.csv']),
                         (q2,Path('outputs/q2_reliability_20261024'),
                          ['family_scope_top3.csv','original_top3_reliability.csv','target_diagnostics.csv','named_marker_diagnostics.csv','null_max_family.csv','null_max_strict.csv'])]:
        for name in files:
            pd.testing.assert_frame_equal(pd.read_csv(new/name),pd.read_csv(old/name),check_exact=False,atol=1e-10,rtol=1e-10)
            comparisons.append(str(new/name))
    records=pd.read_csv(ordinal/'oof_predictions.csv',dtype={'beer':str})
    p=records[[f'probability_grade_{k}' for k in range(7)]].to_numpy(float)
    assert np.isfinite(p).all() and (p>=0).all();np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
    costs=p@abs(np.arange(7)[:,None]-np.arange(7)[None,:])
    median=records.median_relative_grade.to_numpy(int);mode=records.mode_relative_grade.to_numpy(int)
    np.testing.assert_allclose(costs[np.arange(len(records)),median],costs.min(axis=1),atol=1e-12)
    np.testing.assert_allclose(p[np.arange(len(records)),mode],p.max(axis=1),atol=1e-12)
    ids,xf,yf,primary=load_data(source,Path('outputs/p0_audit_20261023_final'))
    splits=pd.read_csv(q1/'splits.csv',dtype={'beer':str});truth=yf.copy();truth.index=ids.beer.astype(str)
    record_index=records.set_index(['beer','target'])
    for fold in sorted(splits.outer_test_fold.unique()):
        train=splits.loc[splits.outer_test_fold.ne(fold),'beer'];test=splits.loc[splits.outer_test_fold.eq(fold),'beer']
        for target in yf:
            reference=truth.loc[train,target].to_numpy(float)
            for beer in test:
                observed=float(truth.loc[beer,target]);percentile=((reference<observed).sum()+.5*(reference==observed).sum())/len(reference)
                assert record_index.loc[(beer,target),'observed_relative_grade']==min(6,int(np.floor(7*percentile)))
    config=json.loads((q2/'summary.json').read_text(encoding='utf-8'))['config']
    original=json.loads(Path('configs/q2_exploratory_v1.json').read_text(encoding='utf-8'))
    strict=np.array([i for i,c in enumerate(xf) if c not in original['exclude_from_compound_top3']])
    expanded=np.array([i for i,c in enumerate(xf) if c not in original['exclude_from_compound_top3'] or c==config['family_indicator']])
    x=xf.to_numpy(float);y=yf.to_numpy(float);styles=ids.tasting_category_fine.to_numpy()
    model,_=make_pipeline('ridge_asinh',{'ridge_alphas':[config['ridge_alpha']]},config['seed'])
    model.set_params(model__alpha=config['ridge_alpha']).fit(x,y);coef=model.named_steps['model'].coef_
    target=pd.read_csv(q2/'target_diagnostics.csv').set_index('target').loc[yf.columns]
    nulls={name:pd.read_csv(q2/f'null_max_{name}.csv').loc[:,yf.columns].to_numpy(float) for name in ['strict','family']}
    for name,indices,pcol,qcol in [('strict',strict,'within_style_max_statistic_p','across_50_targets_BY_q'),
                                  ('family',expanded,'family_within_style_max_statistic_p','family_across_50_targets_BY_q')]:
        obs=abs(coef[:,indices]).max(axis=1);null=nulls[name]
        values=(1+(null>=obs).sum(axis=0))/(len(null)+1)
        order=np.argsort(values,kind='stable');factor=len(values)*sum(1/k for k in range(1,len(values)+1))
        adjusted=np.empty(len(values));running=1.
        for rank in range(len(values)-1,-1,-1):
            running=min(running,values[order[rank]]*factor/(rank+1));adjusted[order[rank]]=running
        np.testing.assert_allclose(target[pcol],values,atol=1e-12);np.testing.assert_allclose(target[qcol],adjusted,atol=1e-12)
    rng=np.random.default_rng(config['seed']);refit_count=0
    for repeat in range(config['within_style_permutations']):
        permutation=np.arange(len(styles))
        for style in sorted(set(styles)):
            indices=np.flatnonzero(styles==style);permutation[indices]=rng.permutation(indices)
        assert np.array_equal(styles,styles[permutation])
        if repeat in [0,999,1998]:
            fitted,_=make_pipeline('ridge_asinh',{'ridge_alphas':[config['ridge_alpha']]},config['seed'])
            fitted.set_params(model__alpha=config['ridge_alpha']).fit(x,y[permutation]);coefficients=abs(fitted.named_steps['model'].coef_)
            np.testing.assert_allclose(nulls['strict'][repeat],coefficients[:,strict].max(axis=1),atol=1e-12)
            np.testing.assert_allclose(nulls['family'][repeat],coefficients[:,expanded].max(axis=1),atol=1e-12);refit_count+=1
    report={'status':'passed','matching_scientific_tables':len(comparisons),'tables':comparisons,
            'ordinal_probability_rows':len(records),'optimal_absolute_grade_actions_checked':True,
            'training_only_constructed_grade_labels_checked':True,'q2_target_p_and_BY_groups_recomputed':100,
            'independent_permutation_pipeline_refits':refit_count,'null_statistics_checked':refit_count*100,
            'raw_grade_recovery':False,'causal_certification':False}
    write_json(root/'latest_additions_verification.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):verify(a.source,a.root)
