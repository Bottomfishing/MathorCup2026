"""Recompute tail gates, nested calibration, risk evidence and real CLI outputs."""
import argparse
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
from beer_models.raw_panel import RawPanelModel,median_grades,rating_loss,predict_raw_frame
from beer_models.tail_risk import (RawOrdinalModel,base_probabilities,cost_decision,fit_risk,
    survival,binary_loss,risk_frame,tail_metrics,feasible)
from r2_raw_q1 import load_inputs,content_hash,RAW_MANIFEST
from r2b_tail_risk import PROTOCOL,BASE_PROTOCOL,select_point,risk_tables,paired_intervals,calibration_records


def exported(frame):
    return pd.read_csv(io.StringIO(pd.DataFrame(frame).to_csv(index=False)),float_precision='round_trip')


def verify(args):
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'));bp=json.loads(BASE_PROTOCOL.read_text(encoding='utf-8'))
    o=args.output;s=json.loads((o/'summary.json').read_text(encoding='utf-8'))
    assert s['protocol_content_sha256']==content_hash(p)
    assert json.loads((o/'protocol.json').read_text(encoding='utf-8'))==p
    assert sha256(o/'model.joblib')==s['model_sha256']
    for name,digest in s['source_sha256'].items():assert sha256(args.source/name)==digest
    d=load_inputs(args.source,args.raw,bp);x,q,ids,targets=d['x'],d['distribution'],d['ids'],d['targets']
    assert sha256(args.raw)==s['raw_source_sha256']
    styles=ids.tasting_category_fine.to_numpy();splits=make_splits(styles,bp['outer_folds'],bp['outer_seed'])
    def read(name):return pd.read_csv(o/name,float_precision='round_trip')
    def compare(name,expected):
        pd.testing.assert_frame_equal(read(name),exported(expected),check_dtype=False,check_exact=False,rtol=0,atol=1e-12,obj=name)
    def matrix(frame,column,beers=ids.beer):
        return frame.pivot(index='beer',columns='target',values=column).loc[beers,targets].to_numpy()
    def probabilities(frame,beers=ids.beer):
        a=np.stack([matrix(frame,f'p{k}',beers) for k in range(7)],axis=2)
        assert np.isfinite(a).all() and (a>=-1e-14).all()
        np.testing.assert_allclose(a.sum(2),1.,rtol=0,atol=1e-12)
        return a
    base=read('oof_base_probabilities.csv.gz');risk=read('oof_risk_probabilities.csv.gz');inner=read('inner_probabilities.csv.gz')
    assert len(base)==25000 and len(risk)==37500 and len(inner)==112500
    for a,keys in [(base,['method','beer','target']),(risk,['method','beer','target']),(inner,['context','source','beer','target'])]:
        assert not a.duplicated(keys).any()
    po={name:probabilities(base.loc[base.method.eq(name)]) for name in p['base_models']}
    ro={name:probabilities(risk.loc[risk.method.eq(name)]) for name in ['training_prior','selected_uncalibrated','selected_calibrated']}
    points=read('oof_point_predictions.csv.gz');assert len(points)==87500
    point_matrices={name:matrix(part,'predicted_grade').astype(int) for name,part in points.groupby('candidate')}
    for name,prob in po.items():
        for cost,weights in p['cost_weights'].items():
            np.testing.assert_array_equal(point_matrices[name+'__'+cost],cost_decision(prob,weights,d['parents'])[0])
    old=pd.read_csv(args.r2_output/'oof_predictions.csv.gz')
    np.testing.assert_array_equal(point_matrices['extra_trees__median'],matrix(old.loc[old.method.eq('extra_trees')],'predicted_grade'))

    split_log=read('inner_splits.csv');choices=read('point_inner_selection.csv');risk_choices=read('risk_inner_selection.csv')
    cal=read('calibration_parameters.csv');refits=0;calibration_contexts=0
    for number in range(len(splits)+1):
        full=number==len(splits);train,test=(np.arange(len(x)),np.array([],int)) if full else splits[number]
        context='full_training' if full else f'outer_{number}'
        ii=inner.loc[inner.context.eq(context)]
        pp={name:probabilities(ii.loc[ii.source.eq(name)],ids.beer.iloc[train]) for name in ['prior',*p['base_models']]}
        inner_splits=make_splits(styles[train],bp['inner_folds'],bp['inner_seed']+number)
        for fold,(it,iv) in enumerate(inner_splits):
            log=split_log.loc[split_log.context.eq(context)&split_log.inner_fold.eq(fold)]
            for role,index in [('train',train[it]),('validation',train[iv])]:
                np.testing.assert_array_equal(log.loc[log.role.eq(role),'beer'],ids.beer.iloc[index])
            np.testing.assert_allclose(pp['prior'][iv],np.broadcast_to(q[train[it]].mean(axis=0),pp['prior'][iv].shape),rtol=0,atol=1e-12)
        chosen,rows=select_point(pp,q[train],d['parents'],p)
        expected=pd.DataFrame([dict(context=context,**row) for row in rows])
        actual=choices.loc[choices.context.eq(context)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,exported(expected),check_dtype=False,check_exact=False,atol=1e-12,rtol=0)
        bank=joblib.load(o/'model.joblib')['risk_panel_model'] if full else joblib.load(o/'models'/f'outer{number}.joblib')['risk']
        if not full:
            assert joblib.load(o/'models'/f'outer{number}.joblib')['point_choice']==chosen
            np.testing.assert_array_equal(point_matrices['inner_selected'][test],point_matrices[chosen][test])
            for name in p['base_models']:
                print(f'Refit outer {number} {name}',flush=True)
                if name=='raw_ordinal':fresh=RawOrdinalModel(p).fit(x[train],q[train])
                else:
                    lookup=np.full(len(x),-1,int);lookup[train]=np.arange(len(train));keep=lookup[d['positions']]>=0
                    fresh=RawPanelModel('extra_trees',bp,bp['model_seed']+number*100+99,d['parents']).fit(
                        x[train],lookup[d['positions'][keep]],d['scores'][keep],d['weights'][keep],q[train])
                np.testing.assert_allclose(base_probabilities(fresh,x[test]),po[name][test],rtol=0,atol=1e-12)
                np.testing.assert_allclose(base_probabilities(bank.models[name],x[test]),po[name][test],rtol=0,atol=1e-12)
                np.testing.assert_allclose(bank.models[name].imputer.statistics_,np.nanmedian(x[train],axis=0),rtol=0,atol=0)
                refits+=1
        rebuilt,rows=fit_risk(bank.models,pp,q[train],p)
        assert bank.choices==rebuilt.choices
        expected=pd.DataFrame([dict(context=context,target=targets[row.pop('target_index')],**row) for row in rows])
        actual=risk_choices.loc[risk_choices.context.eq(context)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,exported(expected),check_dtype=False,check_exact=False,atol=1e-12,rtol=0)
        expected=exported(calibration_records(rebuilt,context,targets))
        pd.testing.assert_frame_equal(cal.loc[cal.context.eq(context)].reset_index(drop=True),expected,check_dtype=False,check_exact=False,atol=1e-12,rtol=0)
        if not full:
            np.testing.assert_allclose(rebuilt.predict_distribution(x[test]),ro['selected_calibrated'][test],rtol=0,atol=1e-12)
            np.testing.assert_allclose(bank.predict_uncalibrated_distribution(x[test]),ro['selected_uncalibrated'][test],rtol=0,atol=1e-12)
            np.testing.assert_allclose(ro['training_prior'][test],np.broadcast_to(q[train].mean(0),ro['training_prior'][test].shape),rtol=0,atol=1e-12)
        calibration_contexts+=1

    compare('point_comparison.csv',[dict(candidate=name,**tail_metrics(q,g)) for name,g in point_matrices.items()]
            if list(point_matrices)==read('point_comparison.csv').candidate.tolist() else
            [dict(candidate=name,**tail_metrics(q,point_matrices[name])) for name in read('point_comparison.csv').candidate])
    all_risk=dict(training_prior=ro['training_prior'],extra_trees_uncalibrated=po['extra_trees'],raw_ordinal_uncalibrated=po['raw_ordinal'],
                  selected_uncalibrated=ro['selected_uncalibrated'],selected_calibrated=ro['selected_calibrated'])
    metrics,curves=risk_tables(all_risk,q,targets,p);compare('risk_metrics.csv',metrics);compare('risk_reliability.csv',curves)
    loss=rating_loss(q,point_matrices['inner_selected']).mean(1)-rating_loss(q,point_matrices['extra_trees__median']).mean(1)
    mean,lo,hi=paired_intervals(loss,styles,p,[.025,.975])
    qualified=feasible(tail_metrics(q,point_matrices['inner_selected']),float(rating_loss(q,point_matrices['extra_trees__median']).mean()),p['point_limits']) and hi<=p['point_limits']['maximum_raw_mae_increase']
    assert qualified==s['tail_point_qualified']
    compare('point_uncertainty.csv',[dict(mean_raw_mae_increase=mean,lower95=lo,upper95=hi,qualified=qualified)])
    defects=[]
    for target in p['defect_targets']:
        j=targets.index(target);truth=1-q[:,j,0]
        delta=binary_loss(1-ro['training_prior'][:,j,0],truth)-binary_loss(1-ro['selected_calibrated'][:,j,0],truth)
        gain,low,high=paired_intervals(delta,styles,p,p['defect_interval_quantiles'])
        status='historical_nonzero_probability_support_not_safety' if low>0 else 'insufficient_heldout_probability_support'
        defects.append(dict(target=target,brier_improvement=gain,lower99=low,upper99=high,status=status,safety_certified=False,interval_scope='fixed_OOF_beer_loss_bootstrap'))
    compare('defect_risk_validation.csv',defects)
    main=read('oof_scores_all52.csv');assert len(main)==13000 and not main.duplicated(['beer','target']).any()
    missing=main.target.isin(s['unlabelled_targets']);assert missing.sum()==500
    assert main.loc[missing,'estimated_grade_0_6'].isna().all() and not main.safety_certified.any()
    for field,k in [('probability_rating_ge1',0),('probability_rating_ge5',4),('probability_rating_6',5)]:
        np.testing.assert_allclose(matrix(main.loc[~missing],field),survival(ro['selected_calibrated'])[:,:,k],rtol=0,atol=1e-12)
        assert main.loc[missing,field].isna().all()
    deployed=point_matrices['inner_selected'] if qualified else point_matrices['extra_trees__median']
    np.testing.assert_array_equal(matrix(main.loc[~missing],'estimated_grade_0_6'),deployed)
    bundle=joblib.load(o/'model.joblib');frame=pd.read_csv(args.source/'1.csv',dtype=str)
    long,primary,wide,quality=predict_raw_frame(frame,bundle)
    risk_values=risk_frame(frame,bundle,long.beer.drop_duplicates().to_numpy())
    long=long.merge(risk_values,on=['beer','target'],how='left',validate='one_to_one',sort=False);long['safety_certified']=False
    compare('full_fit_scores_all52.csv',long);compare('full_fit_grades_all52.csv',wide);compare('sample_quality.csv',quality)
    if not qualified:
        original=RawPanelModel('extra_trees',bp,bp['model_seed']+999,d['parents']).fit(
            x,d['positions'],d['scores'],d['weights'],q)
        np.testing.assert_array_equal(bundle['raw_panel_model'].predict(x)['grade'],original.predict(x)['grade'])
    for name,table in [('scores_all52_long.csv',long),('estimated_grades_primary47.csv',primary),('estimated_grades_all52.csv',wide),('sample_quality.csv',quality)]:
        actual=pd.read_csv(args.smoke/name,float_precision='round_trip')
        pd.testing.assert_frame_equal(actual,exported(table),check_dtype=False,check_exact=False,atol=1e-12,rtol=0)
    info=json.loads((args.smoke/'prediction_info.json').read_text(encoding='utf-8'))
    assert info['model_sha256']==s['model_sha256'] and info['input_sha256']==sha256(args.source/'1.csv')
    assert info['tail_point_qualified']==qualified and info['safety_certified'] is False
    write_json(o/'cli_verification_info.json',info)
    result=dict(status='passed',protocol_content_sha256=content_hash(p),source_beers=len(x),raw_rating_cells=d['scores'].size,
        inner_probability_rows=len(inner),outer_base_probability_rows=len(base),outer_risk_probability_rows=len(risk),
        point_prediction_rows=len(points),outer_base_models_independently_refitted=refits,
        training_only_calibration_contexts_rebuilt=calibration_contexts,point_limits_recomputed=True,
        risk_losses_reliability_and_intervals_recomputed=True,actual_cli_rows=len(long),
        blank_unlabelled_card_rows=int(missing.sum()),tail_point_qualified=bool(qualified),
        full_fallback_matches_R2=not qualified,full_R2_fallback_independently_refitted=not qualified,safety_certified=False,
        scope='Outer models refitted; inner base models and full-data ordinal model not all independently refitted.')
    write_json(o/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--raw',type=Path,default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',type=Path,required=True)
    parser.add_argument('--r2-output',type=Path,default=ROOT/'outputs/r2_raw_q1_20261026',
                        help='Rebuilt R2 baseline directory; defaults to the versioned historical baseline')
    with threadpool_limits(limits=1):verify(parser.parse_args())
