"""Audit relative-grade forecast distributions without claiming calibration.

AI-assisted: Codex, 2026-10-24. All labels are constructed relative grades.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from beer_audit.audit import save_csv,write_json,sha256
from beer_models.baseline import load_data
from beer_models.final_model import relative_grades
from beer_models.ordinal import ranked_probability_score,central_grade_interval
from beer_models.explain import stratified_draw


def run(source,q1,ordinal,out):
    ids,_,yf,primary=load_data(source,Path('outputs/p0_audit_20261023_final'))
    raw=pd.read_csv(ordinal/'oof_predictions.csv',dtype={'beer':str})
    splits=pd.read_csv(q1/'splits.csv',dtype={'beer':str});y=yf.copy();y.index=ids.beer.astype(str)
    fold_map=splits.set_index('beer').outer_test_fold
    p=raw[[f'probability_grade_{k}' for k in range(7)]].to_numpy(float)
    truth=raw.observed_relative_grade.to_numpy(int);prior=np.empty_like(p);prior_rows=[]
    for fold in sorted(splits.outer_test_fold.unique()):
        train=splits.loc[splits.outer_test_fold.ne(fold),'beer'];train_y=y.loc[train].to_numpy(float)
        train_grades=relative_grades(train_y,train_y)
        for j,target in enumerate(yf):
            counts=np.bincount(train_grades[:,j],minlength=7);distribution=counts/counts.sum()
            mask=raw.target.eq(target)&raw.beer.map(fold_map).eq(fold);prior[mask]=distribution
            for k in range(7):prior_rows.append({'fold':fold,'target':target,'grade':k,'training_count':int(counts[k]),'probability':distribution[k]})
    rows=raw[['beer','target','observed_relative_grade']].copy();rows['primary']=rows.target.isin(primary)
    for method,distribution in [('classifier',p),('training_prior',prior)]:
        score=ranked_probability_score(distribution[:,None,:],truth[:,None])[:,0]
        lower,upper=central_grade_interval(distribution[:,None,:]);lower=lower[:,0];upper=upper[:,0]
        rows[method+'_rps']=score;rows[method+'_lower90']=lower;rows[method+'_upper90']=upper
        rows[method+'_interval_covered']=(truth>=lower)&(truth<=upper);rows[method+'_interval_width']=upper-lower
        rows[method+'_zero_probability_true_grade']=distribution[np.arange(len(raw)),truth]==0
        rows[method+'_true_grade_probability']=distribution[np.arange(len(raw)),truth]
    rows['rps_difference']=rows.classifier_rps-rows.training_prior_rps
    # Independently enumerate six cumulative binary outcomes, not the helper formula.
    for distribution,method in [(p,'classifier'),(prior,'training_prior')]:
        direct=np.zeros(len(raw))
        for cutoff in range(6):
            predicted=distribution[:,:cutoff+1].sum(axis=1);observed=(truth<=cutoff).astype(float)
            direct+=(predicted-observed)**2/6
        np.testing.assert_allclose(rows[method+'_rps'],direct,atol=1e-12)
        for i in range(len(raw)):
            mass=distribution[i,int(rows[method+'_lower90'].iloc[i]):int(rows[method+'_upper90'].iloc[i])+1].sum()
            assert mass>=.9-1e-12
    out.mkdir(parents=True,exist_ok=False);save_csv(out,'row_diagnostics.csv',rows);save_csv(out,'training_grade_priors.csv',prior_rows)
    metrics=[]
    def summarize(scope,subset):
        for method in ['classifier','training_prior']:
            metrics.append({'scope':scope,'method':method,'n':len(subset),'rps':subset[method+'_rps'].mean(),
                            'forecast_interval_90_empirical_coverage':subset[method+'_interval_covered'].mean(),
                            'mean_interval_width':subset[method+'_interval_width'].mean(),
                            'zero_probability_true_grade_rows':int(subset[method+'_zero_probability_true_grade'].sum()),
                            'calibrated_coverage_guaranteed':False})
    summarize('primary45',rows.loc[rows.primary]);summarize('all50',rows)
    for grade in range(7):summarize('primary_observed_grade_'+str(grade),rows.loc[rows.primary&rows.observed_relative_grade.eq(grade)])
    save_csv(out,'metrics.csv',metrics)
    calibration=[]
    for method,distribution in [('classifier',p),('training_prior',prior)]:
        keep=rows.primary.to_numpy()
        for cutoff in range(6):
            forecast=distribution[keep,:cutoff+1].sum(axis=1);observed=truth[keep]<=cutoff
            bins=np.minimum((forecast*10).astype(int),9)
            for b in range(10):
                group=bins==b
                if group.any():calibration.append({'method':method,'cutoff_grade':cutoff,'bin':b,'n':int(group.sum()),
                    'mean_forecast_cdf':float(forecast[group].mean()),'observed_cdf_fraction':float(observed[group].mean())})
    save_csv(out,'cdf_calibration.csv',calibration)
    per_beer=rows.loc[rows.primary].groupby('beer').rps_difference.mean().reindex(splits.beer).to_numpy(float)
    rng=np.random.default_rng(20261024);draws=np.array([stratified_draw(splits.tasting_category_fine,rng) for _ in range(2000)])
    bootstrap=per_beer[draws].mean(axis=1);np.save(out/'bootstrap_draws.npy',draws)
    write_json(out/'summary.json',{'status':'passed','rows_checked':len(rows),'source_sha256':{n:sha256(source/n) for n in ['1.csv','4.csv','8.xlsx']},
        'ordinal_oof_sha256':sha256(ordinal/'oof_predictions.csv'),'script_sha256':sha256(Path(__file__)),
        'primary_rps_difference_from_training_prior':float(per_beer.mean()),'fixed_oof_rps_difference_p025':float(np.quantile(bootstrap,.025)),
        'fixed_oof_rps_difference_p975':float(np.quantile(bootstrap,.975)),
        'checks':['six-threshold direct RPS recomputation','intervals contain at least 90% of their own forecast mass','prior labels use only outer training folds'],
        'limitations':['Intervals use uncalibrated forecast distributions; their empirical coverage is measured, not guaranteed.',
                       'All targets are constructed relative grades, not original panel ratings.',
                       'Fixed OOF sample bootstrap omits refitting and model-selection uncertainty. Existing folds are exploratory.']})
    print(pd.DataFrame(metrics).query("scope == 'primary45'").to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','q1','ordinal','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.source,a.q1,a.ordinal,a.output)
