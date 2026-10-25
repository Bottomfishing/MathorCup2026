"""Reconstruct train-only pattern masks and recompute every saved Q1 summary."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from beer_audit.audit import write_json,sha256
from beer_models.baseline import load_data
from beer_models.final_model import relative_grades


def verify(source,output):
    config=json.loads(Path('configs/review_missingness_20261025.json').read_text())
    ids,xf,yf,primary=load_data(source,Path('outputs/p0_audit_20261023_final'))
    excluded=json.loads(Path('configs/final_protocol.json').read_text())['excluded_derived_features']
    xf=xf[[f for f in xf if f not in excluded]];x=xf.to_numpy(float);y=yf.to_numpy(float)
    splits=pd.read_csv('outputs/final_20261024/q1/splits.csv',dtype={'beer':str}).set_index('beer').loc[ids.beer.astype(str),'outer_test_fold'].to_numpy()
    predictions=pd.read_csv(output/'scenario_predictions.csv.gz',dtype={'beer':str})
    assert len(predictions)==20*2*250*50 and not predictions.duplicated(['scenario','repeat','beer','target']).any()
    assert np.isfinite(predictions[['observed','clean','masked']].to_numpy()).all()
    assert predictions[['observed','clean','masked']].isin(range(7)).all().all()
    masks=pd.read_csv(output/'added_missing_cells.csv',dtype={'beer':str})
    donors=pd.read_csv(output/'training_pattern_donors.csv',dtype={'beer':str,'training_donor':str})
    rates=pd.read_csv(output/'training_missing_rates.csv')
    expected_masks=[];expected_donors=[]
    for fold in sorted(set(splits)):
        train=np.where(splits!=fold)[0];test=np.where(splits==fold)[0];missing=~np.isfinite(x[train]);truth=relative_grades(y[train],y[test])
        saved=rates.loc[rates.fold.eq(fold)].set_index('field').loc[xf.columns]
        np.testing.assert_array_equal(saved.training_missing_count,missing.sum(0));assert saved.training_n.eq(len(train)).all()
        for repeat in range(config['repeats']):
            rng=np.random.default_rng(config['seed']+fold*1000+repeat);donor=rng.integers(len(train),size=len(test))
            expected_donors.extend((fold,repeat,str(ids.beer.iloc[i]),str(ids.beer.iloc[train[donor[k]]])) for k,i in enumerate(test))
            uniform=rng.random(x[test].shape)<missing.mean()
            for scenario,mask in [('empirical_train_rows',missing[donor]),('uniform_train_rate',uniform)]:
                for k,j in zip(*np.where(mask&np.isfinite(x[test]))):
                    expected_masks.append((fold,repeat,scenario,str(ids.beer.iloc[test[k]]),xf.columns[j]))
                part=predictions.loc[predictions.fold.eq(fold)&predictions.repeat.eq(repeat)&predictions.scenario.eq(scenario)]
                observed=part.pivot(index='beer',columns='target',values='observed').loc[ids.beer.iloc[test],yf.columns]
                np.testing.assert_array_equal(observed,truth)
    assert list(masks.itertuples(index=False,name=None))==expected_masks
    assert list(donors.itertuples(index=False,name=None))==expected_donors
    reference=pd.read_csv('outputs/q1_task2_20261025/scores_all52_long.csv',dtype={'beer':str})
    lookup=reference.loc[reference.panel_target_available].set_index(['beer','target']).estimated_grade_0_6
    np.testing.assert_array_equal(predictions.clean,lookup.reindex(pd.MultiIndex.from_frame(predictions[['beer','target']])).to_numpy(int))
    assert np.array_equal(predictions.primary,predictions.target.isin(primary))
    for row in pd.read_csv(output/'metrics.csv').itertuples():
        g=predictions.loc[predictions.scenario.eq(row.scenario)&predictions.repeat.eq(row.repeat)&predictions.primary]
        error=abs(g.masked-g.observed);clean=abs(g.clean-g.observed)
        np.testing.assert_allclose([row.mae,row.clean_mae,row.mae_change,row.changed_grade_fraction,row.exact_accuracy],
            [error.mean(),clean.mean(),(error-clean).mean(),g.masked.ne(g.clean).mean(),error.eq(0).mean()],atol=1e-12)
    base=predictions.loc[predictions.scenario.eq('empirical_train_rows')&predictions.repeat.eq(0)&predictions.primary]
    for row in pd.read_csv(output/'clean_confusion_primary45.csv').itertuples():
        count=int((base.observed.eq(row.observed)&base.clean.eq(row.predicted)).sum())
        assert row.count==count
        np.testing.assert_allclose(row.row_fraction,count/base.observed.eq(row.observed).sum(),atol=1e-12)
    summary=json.loads((output/'summary.json').read_text())
    for name,digest in summary['source_sha256'].items():assert sha256(source/name)==digest
    assert summary['config_sha256']==sha256(Path('configs/review_missingness_20261025.json'))
    write_json(output/'independent_verification.json',dict(status='passed',scenario_rows=len(predictions),mask_cells=len(masks),
        checks=['all training-only donors and exact random masks','all clean grades and constructed truths','all 40 scenario metrics','all 49 confusion cells'],
        limitation='Mask generation, labels and summaries independently replayed; masked classifier predictions not refitted by this verifier.'))
    print('Q1 missingness independent verification passed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.source,a.output)
