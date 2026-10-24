"""Independent held-out matching/prediction replay. AI-assisted: Codex, 2026-10-24."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256,write_json
from beer_models.transforms import RobustAsinh
from complete_modeling import load_q1_artifacts


def verify(source,output):
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'));q1=Path(summary['q1'])
    cfg=json.loads((output/'config.json').read_text(encoding='utf-8'))
    ids,xf,yf,_,splits=load_q1_artifacts(source,Path('outputs/p0_audit_20261023_final'),q1)
    x=xf.to_numpy(float);y=yf.stale_hops.to_numpy(float);styles=ids.tasting_category_fine.to_numpy();beers=ids.beer.astype(str).to_numpy()
    table=pd.read_csv(output/'predictions.csv',dtype={'beer':str,'reference_beer':str}).set_index('beer')
    correction=pd.read_csv(output/'corrections.csv',dtype={'beer':str,'reference_beer':str}).set_index(['beer','model'])
    folds=json.loads((output/'folds.json').read_text(encoding='utf-8'));stale=list(yf).index('stale_hops')
    protected=[xf.columns.get_loc(c) for c in cfg['protected_features']];other=[j for j in range(len(xf.columns)) if j not in protected]
    checked=0
    for fold,(train,test) in enumerate(splits):
        assert folds[fold]['training_beers']==beers[train].tolist() and folds[fold]['test_beers']==beers[test].tolist()
        assert not set(beers[train])&set(beers[test])
        impute=SimpleImputer(strategy='median').fit(x[train]);filled=impute.transform(x);compress=RobustAsinh().fit(filled[train]);z=compress.transform(filled)
        models={}
        for name,digest in folds[fold]['model_sha256'].items():
            path=output/f'{name}_fold{fold}.joblib';assert sha256(path)==digest;models[name]=joblib.load(path)
        for i in test:
            beer=beers[i];row=table.loc[beer]
            candidates=[j for j in train if styles[j]==styles[i] and np.isfinite(x[j]).all()]
            assert row.n_reference_candidates==len(candidates)
            if not candidates:assert not row.available;continue
            # Explicit loop and scalar distance, independent of nearest_reference helper.
            distances=[float(np.sqrt(np.mean((z[j,other]-z[i,other])**2))) for j in candidates]
            reference=candidates[min(range(len(candidates)),key=lambda k:distances[k])]
            assert row.reference_beer==beers[reference] and reference not in test
            np.testing.assert_allclose([row.observed,row.reference_observed,row.matching_distance],
                                       [y[i],y[reference],min(distances)],atol=1e-12)
            assert not row.reference_selection_uses_query_outcome
            hybrid=filled[reference].copy();hybrid[protected]=filled[i,protected]
            four=[];full=[]
            for name,model in models.items():
                v=model.predict(np.vstack([filled[reference],hybrid,filled[i]]));v=v if name=='focused' else v[:,stale]
                c=correction.loc[(beer,name)]
                np.testing.assert_allclose([c.predicted_reference,c.predicted_four_coordinate_hybrid,c.predicted_query],v,atol=1e-12)
                a=y[reference]+v[1]-v[0];b=y[reference]+v[2]-v[0]
                np.testing.assert_allclose([row['four_coordinate_'+name],row['full_coordinate_'+name],row['direct_'+name]],[a,b,v[2]],atol=1e-12)
                four.append(a);full.append(b)
            np.testing.assert_allclose([row.four_coordinate_max3,row.full_coordinate_max3],[max(four),max(full)],atol=1e-12)
            checked+=1
    metrics=pd.read_csv(output/'metrics.csv');uncertainty=pd.read_csv(output/'paired_uncertainty.csv')
    for scope in ['Tripel','all_styles']:
        selected=table.loc[table.available & (table['style'].eq('Tripel') if scope=='Tripel' else True)]
        for row in metrics.loc[metrics.scope.eq(scope)].itertuples():
            error=selected[row.method]-selected.observed
            np.testing.assert_allclose([row.mae,row.rmse,row.mean_bias,row.fraction_prediction_ge_observed],
                                       [abs(error).mean(),np.sqrt((error**2).mean()),error.mean(),(error>=0).mean()],atol=1e-12)
        draws=np.load(output/f'{scope}_bootstrap_draws.npy')
        for draw in draws:
            pd.testing.assert_series_equal(selected['style'].iloc[draw].value_counts().sort_index(),selected['style'].value_counts().sort_index())
        for row in uncertainty.loc[uncertainty.scope.eq(scope)].itertuples():
            delta=abs(selected[row.method].to_numpy()-selected.observed.to_numpy())-abs(selected[row.comparator].to_numpy()-selected.observed.to_numpy())
            boot=delta[draws].mean(axis=1)
            np.testing.assert_allclose([row.mae_difference,row.fixed_oof_p025,row.fixed_oof_p975],
                                       [delta.mean(),np.quantile(boot,.025),np.quantile(boot,.975)],atol=1e-12)
    write_json(output/'verification.json',{'status':'passed','matched_queries_replayed':checked,
        'model_fold_replays':9,'metrics_recomputed':len(metrics),'query_labels_excluded_from_matching':True,
        'bootstrap_summaries_recomputed':len(uncertainty),'intervention_validated':False})
    print(f'Reference holdout verified: {checked} queries, {len(metrics)} metric groups.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):verify(a.source,a.output)
