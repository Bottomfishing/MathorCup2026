"""Test reference anchoring on real held-out outcomes. AI-assisted: Codex, 2026-10-24."""
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
from beer_audit.audit import save_csv,write_json,sha256
from beer_models.transforms import RobustAsinh
from beer_models.final_model import estimator
from beer_models.style_models import make_stale_model
from beer_models.reference_validation import nearest_reference,anchored_prediction
from beer_models.explain import stratified_draw
from complete_modeling import load_q1_artifacts


def run(source,q1,output,config_path):
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    ids,xf,yf,bundle,splits=load_q1_artifacts(source,Path('outputs/p0_audit_20261023_final'),q1)
    qcfg=json.loads((q1/'run_manifest.json').read_text(encoding='utf-8'))['config']
    scfg=json.loads(Path('configs/q3_style_protocol.json').read_text(encoding='utf-8'))
    x=xf.to_numpy(float);y=yf.stale_hops.to_numpy(float);styles=ids.tasting_category_fine.to_numpy();beers=ids.beer.astype(str).to_numpy()
    protected=[xf.columns.get_loc(c) for c in cfg['protected_features']];other=[j for j in range(len(xf.columns)) if j not in protected]
    stale=list(yf).index('stale_hops');output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',cfg);write_json(output/'status.json',{'state':'running'})
    records=[];corrections=[];fold_records=[]
    for fold,(train,test) in enumerate(splits):
        impute=SimpleImputer(strategy='median').fit(x[train]);filled=impute.transform(x)
        np.testing.assert_allclose(impute.statistics_,np.nanmedian(x[train],axis=0))
        compress=RobustAsinh().fit(filled[train]);z=compress.transform(filled)
        selected=joblib.load(q1/f'fold{fold}.joblib') # locally rebuilt trusted model
        ridge=estimator('ridge_asinh',qcfg,qcfg['seed']+fold).fit(x[train],yf.to_numpy(float)[train])
        focused=make_stale_model(scfg,scfg['seed']+fold).fit(x[train],y[train])
        models={'selected':selected,'ridge':ridge,'focused':focused}
        for name,model in models.items():joblib.dump(model,output/f'{name}_fold{fold}.joblib',compress=3)
        fold_records.append({'fold':fold,'training_beers':beers[train].tolist(),'test_beers':beers[test].tolist(),
                             'model_sha256':{name:sha256(output/f'{name}_fold{fold}.joblib') for name in models}})
        def predict(model,values,name):
            values=model.predict(values)
            return values if name=='focused' else values[:,stale]
        for i in test:
            candidates=train[(styles[train]==styles[i])&np.isfinite(x[train]).all(axis=1)]
            record={'beer':beers[i],'style':styles[i],'fold':fold,'observed':y[i],
                    'training_global_median':float(np.median(y[train])),
                    'training_style_median':float(np.median(y[train[styles[train]==styles[i]]])),
                    'n_reference_candidates':len(candidates),'query_missing_features':int(np.isnan(x[i]).sum())}
            if not len(candidates):records.append(record|{'available':False});continue
            local,distance=nearest_reference(z[i],z[candidates],other);reference=int(candidates[local])
            assert reference in train and reference not in test and reference!=i
            hybrid=filled[reference].copy();hybrid[protected]=filled[i,protected]
            record.update({'available':True,'reference_beer':beers[reference],'reference_observed':y[reference],
                           'matching_distance':distance,'reference_selection_uses_query_outcome':False})
            four=[];full=[]
            for name,model in models.items():
                pred=predict(model,np.vstack([filled[reference],hybrid,filled[i]]),name)
                four_value=float(anchored_prediction(y[reference],pred[1],pred[0]))
                full_value=float(anchored_prediction(y[reference],pred[2],pred[0]))
                record['direct_'+name]=pred[2];record['four_coordinate_'+name]=four_value;record['full_coordinate_'+name]=full_value
                four.append(four_value);full.append(full_value)
                corrections.append({'beer':beers[i],'reference_beer':beers[reference],'fold':fold,'model':name,
                                    'reference_observed':y[reference],'predicted_reference':pred[0],
                                    'predicted_four_coordinate_hybrid':pred[1],'predicted_query':pred[2],
                                    'four_coordinate_anchored':four_value,'full_coordinate_anchored':full_value})
            record['four_coordinate_max3']=max(four);record['full_coordinate_max3']=max(full);records.append(record)
        print(f'Reference holdout fold {fold} complete',flush=True)
    table=pd.DataFrame(records);save_csv(output,'predictions.csv',table);save_csv(output,'corrections.csv',corrections)
    write_json(output/'folds.json',fold_records)
    methods=['training_global_median','training_style_median','reference_observed']+[prefix+name for prefix in ['direct_','four_coordinate_','full_coordinate_'] for name in ['selected','ridge','focused']]+['four_coordinate_max3','full_coordinate_max3']
    metric_rows=[];uncertainty=[]
    for scope,mask in [('Tripel',table['style'].eq(cfg['primary_style'])),('all_styles',np.ones(len(table),bool))]:
        subset=table.loc[mask&table.available].copy();truth=subset.observed.to_numpy()
        for method in methods:
            predicted=subset[method].to_numpy();error=predicted-truth
            metric_rows.append({'scope':scope,'method':method,'n':len(subset),'eligible_queries':int(np.sum(mask)),
                                'mae':abs(error).mean(),'rmse':np.sqrt(np.mean(error**2)),
                                'mean_bias':error.mean(),'fraction_prediction_ge_observed':float((error>=0).mean()),
                                'coverage_interpretation':'empirical_sign_fraction_not_a_calibrated_bound'})
        rng=np.random.default_rng(cfg['seed']);draws=np.array([stratified_draw(subset['style'],rng) for _ in range(cfg['bootstrap_repeats'])])
        np.save(output/f'{scope}_bootstrap_draws.npy',draws)
        for comparator in cfg['primary_comparators']:
            delta=abs(subset[cfg['primary_method']].to_numpy()-truth)-abs(subset[comparator].to_numpy()-truth)
            boot=delta[draws].mean(axis=1)
            uncertainty.append({'scope':scope,'method':cfg['primary_method'],'comparator':comparator,'mae_difference':delta.mean(),
                                'fixed_oof_p025':np.quantile(boot,.025),'fixed_oof_p975':np.quantile(boot,.975),
                                'scope_note':'Exploratory paired fixed-OOF resampling; not refit, model selection or intervention uncertainty.'})
    save_csv(output,'metrics.csv',metric_rows);save_csv(output,'paired_uncertainty.csv',uncertainty)
    summary={'status':'completed','n_queries':len(table),'n_matched':int(table.available.sum()),
             'n_Tripel':int((table.available&table['style'].eq('Tripel')).sum()),
             'source_sha256':{n:sha256(source/n) for n in ['1.csv','4.csv','8.xlsx']},
             'q1':str(q1.resolve()),'config_sha256':sha256(config_path),'script_sha256':sha256(Path(__file__)),
             'limitations':[cfg['scope'],'Reference selected without query outcomes; design-time outcome screening in actual transfers is a different process.',
                            'Other 220 coordinates of a held-out query differ from its reference. Error is not an isolated four-coordinate intervention effect.',
                            'Maximum of three point predictions is not a confidence bound; its observed coverage is diagnostic only.']}
    write_json(output/'summary.json',summary);write_json(output/'status.json',{'state':'completed'})
    print(pd.DataFrame(metric_rows).query("scope == 'Tripel'").to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--q1',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('configs/q3_reference_holdout.json'))
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.q1,a.output,a.config)
