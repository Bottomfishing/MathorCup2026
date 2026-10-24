"""Independent style-permutation checks. AI-assisted: Codex, 2026-10-24."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256


def run(out):
    manifest=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    parent=Path(manifest['parent']);base=parent/'analytes_only'
    source=Path(json.loads((parent/'run_manifest.json').read_text(encoding='utf-8'))['source'])
    samples=pd.read_csv(parent/'samples.csv',dtype={'beer':str});styles=samples.tasting_category_fine.to_numpy()
    features=pd.read_csv(base/'feature_order.csv').field.tolist();targets=pd.read_csv(parent/'target_order.csv').target.tolist()
    x=pd.read_csv(source/'1.csv',dtype={'beer':str}).set_index('beer').loc[samples.beer,features].to_numpy(float)
    y=pd.read_csv(source/'4.csv',dtype={'beer':str}).set_index('beer').loc[samples.beer,targets].to_numpy(float)
    d=np.load(out/'individual_sample_deltas.npy');assert np.isfinite(d).all()
    for fit in pd.read_csv(base/'fold_models.csv').itertuples():
        path=Path(fit.model_path);assert sha256(path)==fit.model_sha256;model=joblib.load(path)
        test=np.flatnonzero(samples.outer_test_fold.to_numpy()==fit.fold)
        orders=np.load(out/f'orders_fold{fit.fold}.npy');reference=abs(y[test]-model.predict(x[test]))
        for order in orders:
            np.testing.assert_array_equal(np.sort(order),np.arange(len(test)))
            np.testing.assert_array_equal(styles[test][order],styles[test])
        for j in [0,len(features)//2,len(features)-1]:
            actual=[]
            for order in orders:
                changed=x[test].copy();changed[:,j]=x[test][order,j]
                actual.append(abs(y[test]-model.predict(changed))-reference)
            np.testing.assert_allclose(np.mean(actual,axis=0),d[test,j],atol=1e-12)
        np.testing.assert_allclose(d[test].mean(axis=0),np.load(out/f'repeat_means_fold{fit.fold}.npy').mean(axis=0),atol=1e-12)
    means=np.load(out/'bootstrap_means.npy');draws=np.load(out/'bootstrap_draws.npy');counts=np.zeros(means.shape[1:],int)
    for draw,mean in zip(draws,means):
        np.testing.assert_allclose(d[draw].mean(axis=0),mean,atol=1e-12)
        pd.testing.assert_series_equal(pd.Series(styles[draw]).value_counts().sort_index(),pd.Series(styles).value_counts().sort_index())
        for t in range(len(targets)):counts[np.argsort(-mean[:,t],kind='stable')[:3],t]+=1
    table=pd.read_csv(out/'importance.csv');top=pd.read_csv(out/'top3.csv')
    for col,value in [('delta_mae',d.mean(axis=0)),('validation_resample_top3_frequency',counts/len(draws)),
                      ('fixed_model_resample_p025',np.quantile(means,.025,axis=0)),('fixed_model_resample_p975',np.quantile(means,.975,axis=0))]:
        np.testing.assert_allclose(table.pivot(index='field',columns='target',values=col).loc[features,targets],value,atol=1e-12)
    for t,target in enumerate(targets):
        assert top.loc[top.target.eq(target)].sort_values('rank').field.tolist()==[features[j] for j in np.argsort(-d[:,:,t].mean(axis=0),kind='stable')[:3]]
    result={'status':'passed','within_style_permutations_checked':30,'feature_fold_replays':9,
            'bootstrap_draws_recomputed':len(draws),'importance_cells':len(table),'all_target_top3_checked':True}
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1):run(p.parse_args().output)
