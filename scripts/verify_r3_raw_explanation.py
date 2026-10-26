"""Replay R3 fits and held-out attributions; verify derived ranks and label gaps."""
import io
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256,write_json
from beer_models.baseline import make_splits
from beer_models.raw_panel import median_grades
from beer_models.raw_explain import (fit_raw_ridge,explain_linear,explain_quantile,
    check_additivity,decision,training_modules)
from r2_raw_q1 import load_inputs,content_hash
from r3_raw_explanation import PROTOCOL,BASE_PROTOCOL,fit_gbr,select_gbr,assemble,parser,text_hash


def exported(frame):
    return pd.read_csv(io.StringIO(pd.DataFrame(frame).to_csv(index=False)),float_precision='round_trip')


def verify(args):
    o=args.output;s=json.loads((o/'summary.json').read_text(encoding='utf-8'))
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'));bp=json.loads(BASE_PROTOCOL.read_text(encoding='utf-8'))
    assert s['protocol_content_sha256']==content_hash(p)
    assert json.loads((o/'protocol.json').read_text(encoding='utf-8'))==p
    for name,digest in s['source_sha256'].items():assert sha256(args.source/name)==digest
    for name,digest in s['input_lf_normalized_sha256'].items():assert text_hash(ROOT/name)==digest
    for name,digest in s['attribution_sha256'].items():assert sha256(o/name)==digest
    assert sha256(args.raw)==s['raw_source_sha256']
    d=load_inputs(args.source,args.raw,bp);x,q=d['x'],d['distribution'];targets=d['targets']
    styles=d['ids'].tasting_category_fine.to_numpy();splits=make_splits(styles,p['outer_folds'],p['outer_seed'])
    read=lambda name:pd.read_csv(o/name,float_precision='round_trip')
    def compare(name,expected):
        pd.testing.assert_frame_equal(read(name),exported(expected),check_dtype=False,check_exact=False,atol=1e-10,rtol=0,obj=name)
    def matrix(frame,beers):
        assert not frame.duplicated(['beer','target']).any()
        return frame.pivot(index='beer',columns='target',values='estimated_grade').loc[beers,targets].to_numpy(int)
    inner=read('inner_predictions.csv.gz');npz={name:np.load(o/f'{name}_OOF_attributions.npz',allow_pickle=False) for name in ['gbr_selected','ridge_raw_mean']}
    values={name:npz[name]['values'] for name in npz};bases={name:npz[name]['base_values'] for name in npz}
    continuous={name:npz[name]['continuous'] for name in npz}
    for name in npz:
        np.testing.assert_array_equal(npz[name]['beer'],d['ids'].beer)
        np.testing.assert_array_equal(npz[name]['targets'],targets);np.testing.assert_array_equal(npz[name]['features'],d['features'])
        assert values[name].shape==(250,50,224)
        check_additivity(values[name],bases[name],continuous[name])
    grades={'training_median':np.empty((len(x),len(targets)),int)}
    grades.update({name:decision(pred,d['parents']) for name,pred in continuous.items()})
    folds=np.empty(len(x),int);coefficients=[];groups_all=[];edges_all=[]
    selection=[];splitlog=[];background_log=[];members=[];edge_rows=[];errors=[];refits=0
    for fold,(train,test) in enumerate(splits):
        folds[test]=fold;print(f'Refit R3 fold {fold}: inner selection and outer models',flush=True)
        inner_pred={name:np.empty((len(train),len(targets)),int) for name in p['gbr_candidates']}
        for number,(it,iv) in enumerate(make_splits(styles[train],p['inner_folds'],p['inner_seed']+fold)):
            for role,index in [('train',train[it]),('validation',train[iv])]:
                splitlog.extend(dict(outer_fold=fold,inner_fold=number,role=role,beer=int(d['ids'].beer.iloc[i])) for i in index)
            for name in p['gbr_candidates']:
                fresh=fit_gbr(d,train[it],name,p['model_seed']+fold*100+number,bp);refits+=1
                prediction=fresh.predict(x[train[iv]])['grade'];inner_pred[name][iv]=prediction
                saved=inner.loc[inner.outer_fold.eq(fold)&inner.inner_fold.eq(number)&inner.method.eq(name)]
                np.testing.assert_array_equal(matrix(saved,d['ids'].beer.iloc[train[iv]]),prediction)
        chosen,losses=select_gbr(inner_pred,q[train],p['gbr_candidates'])
        selection.extend(dict(outer_fold=fold,candidate=name,inner_raw_mae=loss,selected=name==chosen) for name,loss in losses.items())
        fresh_gbr=fit_gbr(d,train,chosen,p['model_seed']+fold*100+99,bp)
        fresh_ridge=fit_raw_ridge(x[train],q[train],p['ridge_alpha'],p['model_seed']+fold*100+99);refits+=2
        saved=joblib.load(o/'models'/f'outer{fold}.joblib');assert saved['chosen']==chosen
        np.testing.assert_array_equal(saved['train'],train);np.testing.assert_array_equal(saved['test'],test)
        background=np.sort(np.random.default_rng(p['background_seed']+fold).choice(train,min(p['background_size'],len(train)),replace=False))
        np.testing.assert_array_equal(saved['background'],background);assert not np.intersect1d(background,test).size
        background_log.extend(dict(outer_fold=fold,beer=int(d['ids'].beer.iloc[i])) for i in background)
        for name,model,fun in [('gbr_selected',fresh_gbr,explain_quantile),('ridge_raw_mean',fresh_ridge,explain_linear)]:
            imputer=model.imputer if name=='gbr_selected' else model.named_steps['impute']
            np.testing.assert_allclose(imputer.statistics_,np.nanmedian(x[train],axis=0),atol=0,rtol=0)
            v,b,c=fun(model,x[test],x[background])
            np.testing.assert_allclose(values[name][test],v,atol=1e-10,rtol=0)
            np.testing.assert_allclose(bases[name][test],b,atol=1e-10,rtol=0)
            np.testing.assert_allclose(continuous[name][test],c,atol=1e-10,rtol=0)
            errors.append(dict(outer_fold=fold,method=name,max_additivity_error=check_additivity(v,b,c)))
            stored=saved['gbr'] if name=='gbr_selected' else saved['ridge']
            saved_c=stored.model.predict(stored.imputer.transform(x[test])) if name=='gbr_selected' else stored.predict(x[test])
            np.testing.assert_allclose(saved_c,c,atol=1e-10,rtol=0)
        coefficients.append(fresh_ridge.named_steps['model'].coef_)
        reference=np.broadcast_to(q[train].mean(0),(len(test),len(targets),7))
        grades['training_median'][test]=decision(median_grades(reference),d['parents'])
        groups,edges=training_modules(fresh_gbr.imputer.transform(x[train]),p['correlation_threshold'])
        groups_all.append(groups);edges_all.append(edges)
        members.extend(dict(outer_fold=fold,module_id=g,field=d['features'][i]) for g,group in enumerate(groups) for i in group)
        edge_rows.extend(dict(outer_fold=fold,source=d['features'][i],target=d['features'][j],spearman_rho=r) for i,j,r in edges)
    tables=assemble(d,p,folds,grades,values,bases,continuous,np.array(coefficients),groups_all,edges_all)
    for name,frame in tables.items():compare(name,frame)
    for name,frame in [('inner_selection.csv',selection),('inner_splits.csv',splitlog),('background_beers.csv',background_log),
        ('training_module_members.csv',members),('training_correlation_edges.csv',edge_rows),('additivity_checks.csv',errors)]:compare(name,frame)
    compare('outer_folds.csv',pd.DataFrame(dict(beer=d['ids'].beer,outer_fold=folds)))
    slots=read('top3_raw_all52_slots.csv');assert len(slots)==312
    missing=slots.target.isin(bp['unlabelled_card_targets']);assert missing.sum()==12
    assert slots.loc[missing,'field'].isna().all()
    known=read('top3_raw_all50.csv');assert (known.ranking_score>1e-12).all()
    assert not known.causal_effect_established.any() and not known.migration_control_validated.any()
    result=dict(status='passed',source_beers=250,targets=50,features=224,independently_refitted_models=refits,
        refitted_inner_GBR_models=12,refitted_outer_models=6,recomputed_outer_explanations=6,
        attribution_cells_per_method=int(values['gbr_selected'].size),training_only_background_beers_per_fold=100,
        inner_prediction_rows=len(inner),rank_rows=len(known),scorecard_rank_slots=len(slots),
        empty_acetaldehyde_slots=int(missing.sum()),all_derived_tables_recomputed=len(tables),
        max_additivity_error=max(row['max_additivity_error'] for row in errors),
        historical_Q2_source_hashes_unchanged=True,Q1_Q3_not_retrained=True,
        scope='Same source and folds replayed, not an independent data validation or causal audit.')
    write_json(o/'verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):verify(parser().parse_args())
