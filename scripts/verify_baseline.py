"""Independent result/serialization checks. AI-assisted: OpenAI Codex, 2026-10-23."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from threadpoolctl import threadpool_limits


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    args=parser.parse_args();out=args.output
    manifest=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    source=Path(manifest['source'])
    x=pd.read_csv(source/'1.csv',dtype={'beer':str}).set_index('beer')
    y=pd.read_csv(source/'4.csv',dtype={'beer':str}).set_index('beer')
    features=pd.read_csv(out/'feature_columns.csv')['field'].tolist()
    targets=pd.read_csv(out/'target_columns.csv')['field'].tolist()
    splits=pd.read_csv(out/'split_manifest.csv',dtype={'beer':str})
    inner=pd.read_csv(out/'inner_split_manifest.csv',dtype={'beer':str})
    oof=pd.read_csv(out/'oof_predictions.csv',dtype={'beer':str})
    metrics=pd.read_csv(out/'metrics_by_target.csv')
    fits=pd.read_csv(out/'fit_log.csv')
    assert not set(features)&{'beer','beer_id','tasting_category_fine'}
    assert splits.beer.is_unique
    assert not oof.duplicated(['beer','model','target']).any()
    assert np.isfinite(oof[['observed','predicted']]).all().all()
    checked_models=0
    max_reload_difference=0.
    for fit in fits.itertuples():
        test=splits.loc[splits.outer_test_fold.eq(fit.fold),'beer'].tolist()
        train=splits.loc[~splits.outer_test_fold.eq(fit.fold),'beer'].tolist()
        assert not set(test)&set(inner.loc[inner.outer_fold.eq(fit.fold),'beer'])
        path=out/'models'/f'{fit.model}_fold{fit.fold}.joblib'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==fit.model_sha256
        model=joblib.load(path) # trusted local model, verified against its fit log
        np.testing.assert_allclose(model.named_steps['impute'].statistics_,
                                   np.nanmedian(x.loc[train,features].to_numpy(float),axis=0),rtol=1e-12,atol=1e-12)
        with threadpool_limits(limits=1):pred=model.predict(x.loc[test,features].to_numpy(float))
        saved=oof.loc[oof.model.eq(fit.model)&oof.fold.eq(fit.fold)]
        saved_pred=saved.pivot(index='beer',columns='target',values='predicted').loc[test,targets].to_numpy()
        np.testing.assert_allclose(pred,saved_pred,rtol=1e-10,atol=1e-10)
        max_reload_difference=max(max_reload_difference,float(abs(pred-saved_pred).max()))
        checked_models+=1
    for row in metrics.itertuples():
        group=oof.loc[oof.model.eq(row.model)&oof.target.eq(row.target)].set_index('beer')
        np.testing.assert_allclose(group.observed,y.loc[group.index,row.target],rtol=0,atol=1e-12)
        truth,pred=group.observed,group.predicted
        np.testing.assert_allclose([row.mae,row.rmse,row.r2],
                                   [mean_absolute_error(truth,pred),np.sqrt(mean_squared_error(truth,pred)),r2_score(truth,pred)],
                                   rtol=1e-10,atol=1e-10)
    assert len(oof)==len(splits)*len(targets)*metrics.model.nunique()
    for file,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((source/file).read_bytes()).hexdigest()==digest
    result={'status':'passed','oof_rows':len(oof),'models_reloaded':checked_models,
            'target_model_metric_groups_checked':len(metrics),'max_prediction_reload_abs_difference':max_reload_difference,
            'train_only_imputation_checked':True,'outer_test_absent_from_inner_cv':True,
            'source_files_unchanged':True,'scope':'independent metric recomputation and saved-model replay; no full retraining'}
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
