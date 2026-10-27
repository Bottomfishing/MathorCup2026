"""Leakage-controlled baseline experiments, not final competition results.

AI-assisted: OpenAI Codex (GPT-6), OpenAI, 2026-10-23.
Source labels already include upstream panelist normalization; this program
cannot reverse or undo that upstream processing. See docs/P1阶段交接.md.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from beer_audit.checks import IDS, align_samples, numeric_frame
from beer_audit.audit import sha256, write_json, save_csv
from .transforms import RobustAsinh


def make_splits(styles, n_splits, seed):
    styles=np.asarray(styles)
    counts=pd.Series(styles).value_counts()
    if n_splits < 2 or counts.min() < n_splits:
        raise ValueError('Not enough samples in every style for requested stratification')
    splitter=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    splits=list(splitter.split(np.zeros(len(styles)),styles))
    seen=np.zeros(len(styles),dtype=int)
    for train,test in splits:
        if np.intersect1d(train,test).size or len(train)+len(test)!=len(styles):
            raise ValueError('Invalid train/validation partition')
        seen[test]+=1
    if not np.all(seen==1):raise ValueError('Every sample needs exactly one OOF prediction')
    return splits


def make_pipeline(name, config, seed):
    # No label, style, ID, correlation matrix or full-data imputation is a feature.
    steps=[('impute',SimpleImputer(strategy='median',keep_empty_features=True))]
    if name.endswith('_asinh'):
        steps.append(('compress',RobustAsinh()))
        name=name.removesuffix('_asinh')
    grid={}
    if name=='mean':model=DummyRegressor(strategy='mean')
    elif name=='ridge':
        steps.append(('scale',StandardScaler()))
        model=Ridge()
        grid={'model__alpha':config['ridge_alphas']}
    elif name=='pls':
        # PLSRegression scales both X and y using the current training fold.
        model=PLSRegression(scale=True,max_iter=1000)
        grid={'model__n_components':config['pls_components']}
    elif name=='extra_trees':
        model=ExtraTreesRegressor(**config['extra_trees'],random_state=seed,n_jobs=1)
    elif name=='gradient_boosting':
        model=MultiOutputRegressor(GradientBoostingRegressor(**config['gradient_boosting'],random_state=seed),n_jobs=1)
    else:raise ValueError('Unknown model: '+name)
    steps.append(('model',model))
    return Pipeline(steps),grid


def mask_observed(x, rate, seed):
    """Nested masks for the same seed across rates, preserving original missingness."""
    if not 0 <= rate <= 1:raise ValueError('Missing rate must be in [0,1]')
    x=np.asarray(x,float)
    mask=(np.random.default_rng(seed).random(x.shape)<rate)&np.isfinite(x)
    result=x.copy();result[mask]=np.nan
    return result,mask


def metric_vectors(y,pred):
    y,pred=np.asarray(y,float),np.asarray(pred,float)
    if y.shape!=pred.shape or y.ndim!=2 or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Invalid truth/prediction shape or values')
    residual=y-pred
    sse=np.square(residual).sum(axis=0)
    sst=np.square(y-y.mean(axis=0)).sum(axis=0)
    r2=np.full(y.shape[1],np.nan)
    np.divide(sse,sst,out=r2,where=sst>1e-15)
    r2=1-r2
    return {'mae':np.mean(abs(residual),axis=0),'rmse':np.sqrt(np.mean(residual**2,axis=0)),'r2':r2}


def load_data(source,audit):
    inventory=pd.read_csv(audit/'data_inventory.csv')
    for name in ['1.csv','4.csv','8.xlsx']:
        expected=inventory.loc[inventory.file.eq(name),'sha256']
        if len(expected)!=1 or sha256(source/name)!=expected.iloc[0]:
            raise ValueError('Source changed since audit: '+name)
    raw=pd.read_csv(source/'1.csv',dtype=str)
    target=align_samples(raw,pd.read_csv(source/'4.csv',dtype=str))
    x=numeric_frame(raw.drop(columns=IDS))
    y=numeric_frame(target.drop(columns=IDS))
    if y.isna().any().any():raise ValueError('Missing target labels require a separate protocol')
    if x.isna().all().any():raise ValueError('Entirely missing feature requires explicit exclusion')
    mapping=pd.read_csv(audit/'sensory_mapping.csv')
    if mapping.R_name.duplicated().any() or set(mapping.R_name)!=set(y):
        raise ValueError('Target dictionary mismatch')
    primary=mapping.loc[mapping.scope.eq('aroma_flavour_primary_candidate'),'R_name'].tolist()
    return raw[IDS],x,y,primary


def run(source,audit,config_path,output):
    source,audit,output=source.resolve(),audit.resolve(),output.resolve()
    if source==output or source in output.parents:raise ValueError('Cannot write inside original source')
    config=json.loads(config_path.read_text(encoding='utf-8'))
    ids,xf,yf,primary=load_data(source,audit)
    x,y=xf.to_numpy(float),yf.to_numpy(float)
    scopes={'primary_45':np.array([c in primary for c in yf]),'all_50':np.ones(y.shape[1],dtype=bool)}
    splits=make_splits(ids.tasting_category_fine,config['outer_folds'],config['seed'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'models').mkdir()
    write_json(output/'config.json',config)
    save_csv(output,'feature_columns.csv',[{'field':c,'position':i} for i,c in enumerate(xf)])
    save_csv(output,'target_columns.csv',[{'field':c,'position':i,'primary':c in primary} for i,c in enumerate(yf)])
    oof={name:np.full_like(y,np.nan) for name in config['models']}
    corrupt={(name,rate,rep):np.full_like(y,np.nan) for name in config['models']
             for rate in config['missing_rates'] for rep in range(config['missing_repeats'])}
    folds=np.full(len(y),-1,dtype=int)
    inner_rows=[]; fit_rows=[]; mask_rows=[]; fold_metrics=[]; inner_scores=[]
    start=time.perf_counter()
    for fold,(train,test) in enumerate(splits):
        folds[test]=fold
        inner=make_splits(ids.tasting_category_fine.iloc[train],config['inner_folds'],config['seed']+fold+1)
        for inner_fold,(it,iv) in enumerate(inner):
            for role,positions in [('train',it),('validation',iv)]:
                inner_rows.extend({'outer_fold':fold,'inner_fold':inner_fold,'role':role,'beer':ids.beer.iloc[train[p]]} for p in positions)
        masks={}
        for rate in config['missing_rates']:
            for rep in range(config['missing_repeats']):
                masked,mask=mask_observed(x[test],rate,config['seed']+1000+fold*100+rep)
                masks[rate,rep]=masked
                mask_rows.append({'fold':fold,'rate':rate,'repeat':rep,'masked_cells':int(mask.sum()),
                                  'observed_cells':int(np.isfinite(x[test]).sum()),
                                  'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()})
        for name in config['models']:
            print(f'fold={fold+1}/{len(splits)} model={name}',flush=True)
            tick=time.perf_counter()
            pipe,grid=make_pipeline(name,config,config['seed']+fold)
            if grid:
                # All preprocessing belongs inside the estimator cloned for each inner split.
                fitted=GridSearchCV(pipe,grid,scoring='neg_mean_absolute_error',cv=inner,
                                    n_jobs=1,error_score='raise',refit=True)
                fitted.fit(x[train],y[train])
                params=fitted.best_params_
                best=fitted.best_estimator_
                for i,p in enumerate(fitted.cv_results_['params']):
                    inner_scores.append({'fold':fold,'model':name,'parameters':json.dumps(p),
                                         'mean_validation_mae':-fitted.cv_results_['mean_test_score'][i],
                                         'selected':i==fitted.best_index_})
            else:
                best=pipe.fit(x[train],y[train]);params={}
            pred=best.predict(x[test])
            if pred.shape!=y[test].shape or not np.isfinite(pred).all():raise ValueError('Invalid OOF prediction')
            oof[name][test]=pred
            for (rate,rep),masked in masks.items():corrupt[name,rate,rep][test]=best.predict(masked)
            model_path=output/'models'/f'{name}_fold{fold}.joblib'
            joblib.dump(best,model_path,compress=3)
            fit_rows.append({'fold':fold,'model':name,'n_train':len(train),'n_test':len(test),
                             'parameters':json.dumps(params),'seconds':time.perf_counter()-tick,
                             'model_sha256':sha256(model_path)})
            metrics=metric_vectors(y[test],pred)
            for scope,select in scopes.items():
                fold_metrics.append({'fold':fold,'model':name,'scope':scope,
                                     **{key:float(np.nanmean(value[select])) for key,value in metrics.items()}})
    if np.any(folds<0):raise ValueError('OOF coverage incomplete')
    metrics_rows=[]; aggregate=[]; prediction_frames=[];style_rows=[]; robustness=[]
    for name,pred in oof.items():
        metrics=metric_vectors(y,pred)
        for j,target in enumerate(yf):
            metrics_rows.append({'model':name,'target':target,'primary':target in primary,
                                 **{k:v[j] for k,v in metrics.items()}})
        for scope,select in scopes.items():
            aggregate.append({'model':name,'scope':scope,'n_samples':len(y),'n_targets':int(select.sum()),
                              **{'macro_'+k:float(np.nanmean(v[select])) for k,v in metrics.items()}})
        prediction_frames.append(pd.DataFrame({'beer':np.repeat(ids.beer.to_numpy(),y.shape[1]),
                                               'style':np.repeat(ids.tasting_category_fine.to_numpy(),y.shape[1]),
                                               'fold':np.repeat(folds,y.shape[1]),'model':name,
                                               'target':np.tile(yf.columns,len(y)),
                                               'observed':y.ravel(),'predicted':pred.ravel(),
                                               'scale':'provided_standardized_panel_score','prediction_kind':'out_of_fold'}))
        for style in sorted(ids.tasting_category_fine.unique()):
            idx=ids.tasting_category_fine.eq(style).to_numpy()
            m=metric_vectors(y[idx],pred[idx])
            style_rows.append({'model':name,'style':style,'n':int(idx.sum()),
                               'primary_mae':float(m['mae'][scopes['primary_45']].mean()),
                               'primary_rmse':float(m['rmse'][scopes['primary_45']].mean())})
        for rate in config['missing_rates']:
            for rep in range(config['missing_repeats']):
                m=metric_vectors(y,corrupt[name,rate,rep])
                for scope,select in scopes.items():
                    robustness.append({'model':name,'additional_missing_rate':rate,'repeat':rep,'scope':scope,
                                       **{'macro_'+k:float(np.nanmean(v[select])) for k,v in m.items()}})
    split_manifest=ids.copy();split_manifest['outer_test_fold']=folds
    save_csv(output,'split_manifest.csv',split_manifest)
    save_csv(output,'inner_split_manifest.csv',inner_rows)
    save_csv(output,'inner_cv_scores.csv',inner_scores)
    save_csv(output,'fit_log.csv',fit_rows)
    save_csv(output,'fold_metrics.csv',fold_metrics)
    save_csv(output,'metrics_by_target.csv',metrics_rows)
    save_csv(output,'metrics_summary.csv',aggregate)
    save_csv(output,'metrics_by_style.csv',style_rows)
    save_csv(output,'oof_predictions.csv',pd.concat(prediction_frames,ignore_index=True))
    save_csv(output,'missingness_robustness.csv',robustness)
    save_csv(output,'missingness_masks.csv',mask_rows)
    project=Path(__file__).resolve().parents[2]
    manifest={'stage':'P1 exploratory standardized-score baseline','started_protocol':str(config_path),
              'finished_utc':datetime.now(timezone.utc).isoformat(),'duration_seconds':time.perf_counter()-start,
              'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
              'source':str(source),'audit':str(audit),'config_sha256':sha256(config_path),
              'source_sha256':{n:sha256(source/n) for n in ['1.csv','4.csv','8.xlsx']},
              'audit_mapping_sha256':sha256(audit/'sensory_mapping.csv'),
              'code_sha256':{str(p.relative_to(project)):sha256(p) for p in sorted((project/'src').rglob('*.py'))},
              'original_0_6_scale_identified':False,
              'warnings':['Exploratory model comparison, not independent evaluation of a post-hoc selected winner.',
                          'Supplied labels were normalized upstream by panelist; original raw scale is unavailable.',
                          'Ridge/PLS hyperparameters selected by inner mean MAE across all 50 targets.',
                          'Fixed tree settings chosen before this run; no outer-fold tuning.',
                          'Random additional missingness is a stress test, not evidence of the real missingness mechanism.',
                          'Style is used for splitting/diagnostics only, not as model input.',
                          'Persisted joblib models are for trusted local use only.']}
    write_json(output/'run_manifest.json',manifest)
    print(pd.DataFrame(aggregate).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--audit',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=Path('configs/baseline_v1.json'))
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    # Limit BLAS threads as well as estimator threads for reproducibility and responsiveness.
    with threadpool_limits(limits=1):run(args.source,args.audit,args.config,args.output)

if __name__=='__main__':main()
