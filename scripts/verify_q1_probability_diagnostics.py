"""Independent arithmetic audit of relative-grade probability diagnostics."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from beer_models.baseline import load_data
from beer_audit.audit import write_json


def verify(source,q1,ordinal,out):
    ids,_,yf,primary=load_data(source,Path('outputs/p0_audit_20261023_final'))
    raw=pd.read_csv(ordinal/'oof_predictions.csv',dtype={'beer':str});rows=pd.read_csv(out/'row_diagnostics.csv',dtype={'beer':str})
    pd.testing.assert_frame_equal(rows[['beer','target','observed_relative_grade']],raw[['beer','target','observed_relative_grade']])
    splits=pd.read_csv(q1/'splits.csv',dtype={'beer':str});fold_map=splits.set_index('beer').outer_test_fold
    truth=yf.copy();truth.index=ids.beer.astype(str)
    p=raw[[f'probability_grade_{k}' for k in range(7)]].to_numpy(float);prior=np.empty_like(p)
    priors=pd.read_csv(out/'training_grade_priors.csv').set_index(['fold','target','grade'])
    for fold in sorted(splits.outer_test_fold.unique()):
        train=splits.loc[splits.outer_test_fold.ne(fold),'beer']
        for target in yf:
            values=truth.loc[train,target].to_numpy(float)
            percentile=((values[:,None]<values[None,:]).sum(axis=0)+.5*(values[:,None]==values[None,:]).sum(axis=0))/len(values)
            grades=np.minimum(6,np.floor(7*percentile).astype(int));counts=np.bincount(grades,minlength=7)
            for k in range(7):
                record=priors.loc[(fold,target,k)]
                assert record.training_count==counts[k]
                np.testing.assert_allclose(record.probability,counts[k]/len(values),atol=1e-12)
            mask=raw.target.eq(target)&raw.beer.map(fold_map).eq(fold);prior[mask]=counts/len(values)
    y=raw.observed_relative_grade.to_numpy(int);cutoffs=np.arange(6)
    for method,distribution in [('classifier',p),('training_prior',prior)]:
        predicted=np.column_stack([distribution[:,:k+1].sum(axis=1) for k in cutoffs])
        observed=y[:,None]<=cutoffs
        rps=np.sum((predicted-observed)**2,axis=1)/6
        np.testing.assert_allclose(rows[method+'_rps'],rps,atol=1e-12)
        lower=rows[method+'_lower90'].to_numpy(int);upper=rows[method+'_upper90'].to_numpy(int)
        for i in range(len(rows)):
            assert distribution[i,:lower[i]+1].sum()>=.05-1e-12 and distribution[i,:lower[i]].sum()<=.05+1e-12
            assert distribution[i,:upper[i]+1].sum()>=.95-1e-12 and distribution[i,:upper[i]].sum()<=.95+1e-12
        np.testing.assert_array_equal(rows[method+'_interval_covered'],(y>=lower)&(y<=upper))
        np.testing.assert_array_equal(rows[method+'_interval_width'],upper-lower)
    for record in pd.read_csv(out/'metrics.csv').itertuples():
        if record.scope=='all50':keep=np.ones(len(rows),bool)
        elif record.scope=='primary45':keep=rows.target.isin(primary)
        else:keep=rows.target.isin(primary)&rows.observed_relative_grade.eq(int(record.scope.rsplit('_',1)[-1]))
        subset=rows.loc[keep];method=record.method
        np.testing.assert_allclose([record.rps,record.forecast_interval_90_empirical_coverage,record.mean_interval_width],
                                   [subset[method+'_rps'].mean(),subset[method+'_interval_covered'].mean(),subset[method+'_interval_width'].mean()],atol=1e-12)
        assert not record.calibrated_coverage_guaranteed and record.n==len(subset)
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'));draws=np.load(out/'bootstrap_draws.npy')
    delta=rows.loc[rows.target.isin(primary)].groupby('beer').rps_difference.mean().reindex(splits.beer).to_numpy(float)
    bootstrap=delta[draws].mean(axis=1)
    for draw in draws:
        pd.testing.assert_series_equal(splits.tasting_category_fine.iloc[draw].value_counts().sort_index(),splits.tasting_category_fine.value_counts().sort_index())
    np.testing.assert_allclose([summary['primary_rps_difference_from_training_prior'],summary['fixed_oof_rps_difference_p025'],summary['fixed_oof_rps_difference_p975']],
                               [delta.mean(),np.quantile(bootstrap,.025),np.quantile(bootstrap,.975)],atol=1e-12)
    write_json(out/'verification.json',{'status':'passed','forecast_rows_checked':len(rows),'training_prior_rows':len(priors),
        'metric_groups':18,'independent_six_threshold_scores':True,'forecast_quantiles_checked':True,
        'bootstrap_draws_checked':len(draws),'coverage_guaranteed':False,'raw_rating_recovery':False})
    print(f'Probability diagnostics verified: {len(rows)} rows, {len(priors)} training priors.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','q1','ordinal','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.source,a.q1,a.ordinal,a.output)
