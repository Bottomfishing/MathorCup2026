"""Deterministic Stale model bounds for supplied-coordinate tolerance scenarios.

AI-assisted: Codex, 2026-10-24. No scenario radius is claimed to be measured
assay error; bounds do not certify actual flavor, style or physical feasibility.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv,write_json,sha256
from beer_models.interval_bounds import prediction_bounds
from beer_models.generation import StyleGenerator
from complete_modeling import load_q1_artifacts


def run(source,q1,style_dir,resolution,reference,output,config_path):
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    summary=json.loads((reference/'summary.json').read_text(encoding='utf-8'))
    parent=json.loads((resolution/'summary.json').read_text(encoding='utf-8'))
    if summary['parent_resolution_summary_sha256']!=sha256(resolution/'summary.json'):raise ValueError('Parent differs')
    for name,digest in parent['style_models_sha256'].items():
        if sha256(style_dir/(name+'.joblib'))!=digest:raise ValueError('Style model differs')
    ids,xf,yf,bundle,_=load_q1_artifacts(source,Path('outputs/p0_audit_20261023_final'),q1)
    x=xf.to_numpy(float);beers=ids.beer.astype(str).to_numpy();lookup={b:i for i,b in enumerate(beers)}
    styles=ids.tasting_category_fine.to_numpy();g=StyleGenerator().fit(x,styles)
    lower_support=g.x_[styles=='Tripel'].min(axis=0);upper_support=g.x_[styles=='Tripel'].max(axis=0)
    q1_raw,q3_raw=np.nanquantile(x,[.25,.75],axis=0);iqr=q3_raw-q1_raw
    protected=[xf.columns.get_loc(c) for c in cfg['protected_features']]
    other=[j for j in range(len(xf.columns)) if j not in protected]
    identity_nonnegative=[xf.columns.get_loc(c) for c in ['X2.3.butanedione','X.E..2.nonenal']]
    evidence=pd.read_csv(resolution/'source_defect_evidence.csv',dtype={'beer':str}).set_index(['beer','field'])
    transfers=pd.read_csv(reference/'transfer_results.csv',dtype={'beer':str,'reference_beer':str})
    profiles=pd.read_csv(reference/'transfer_profiles.csv',dtype={'beer':str}).set_index('beer')
    models={'selected':bundle['model'],'ridge':bundle['guard'],'focused':joblib.load(style_dir/'focused_stale.joblib')}
    stale=list(yf).index('stale_hops')
    output.mkdir(parents=True,exist_ok=False);write_json(output/'config.json',cfg)
    save_csv(output,'feature_scales.csv',[{'field':field,'training_iqr':iqr[j]} for j,field in enumerate(xf)])
    detail=[];boxes=[];results=[];sample_checks=[]
    for row in transfers.itertuples():
        i=lookup[row.beer];d=lookup[row.reference_beer]
        center=profiles.loc[row.beer,xf.columns].to_numpy(float);donor=x[d]
        assert np.isfinite(donor).all()
        caps=np.array([min(float(evidence.loc[(row.beer,xf.columns[j]),'constraint_cap_coordinate']),donor[j]) for j in protected])
        source_y=float(yf.stale_hops.iloc[i]);reference_y=float(yf.stale_hops.iloc[d])
        donor_predictions={name:float(model.predict(donor[None,:])[0] if name=='focused' else model.predict(donor[None,:])[0,stale]) for name,model in models.items()}
        for ri,radius in enumerate(cfg['radius_training_iqr_fractions']):
            lo=center-radius*iqr;hi=center+radius*iqr
            lo[other]=np.maximum(lo[other],lower_support[other]);hi=np.minimum(hi,upper_support)
            hi[protected]=np.minimum(hi[protected],caps)
            lo[identity_nonnegative]=np.maximum(lo[identity_nonnegative],0)
            if (lo>center+1e-10).any() or (hi<center-1e-10).any() or (lo>hi).any():raise ValueError('Nominal candidate outside scenario box')
            boxes.append({'beer':row.beer,'reference_beer':row.reference_beer,'radius':radius,
                          **{'lower__'+field:lo[j] for j,field in enumerate(xf)},
                          **{'upper__'+field:hi[j] for j,field in enumerate(xf)}})
            rng=np.random.default_rng(cfg['seed']+i*100+ri)
            samples=rng.uniform(lo,hi,size=(cfg['monte_carlo_spot_checks'],len(center)))
            margins=[];upper_values=[]
            for name,model in models.items():
                target=0 if name=='focused' else stale
                low,high=prediction_bounds(model,lo,hi,target)
                nominal=float(model.predict(center[None,:])[0] if name=='focused' else model.predict(center[None,:])[0,stale])
                pred=model.predict(samples);pred=pred if name=='focused' else pred[:,stale]
                assert low-1e-10<=nominal<=high+1e-10
                assert (pred>=low-1e-10).all() and (pred<=high+1e-10).all()
                after_low=reference_y+low-donor_predictions[name];after_high=reference_y+high-donor_predictions[name]
                anchored_nominal=reference_y+nominal-donor_predictions[name]
                if radius==0:np.testing.assert_allclose([low,high],[nominal,nominal],atol=1e-10)
                detail.append({'beer':row.beer,'reference_beer':row.reference_beer,'radius':radius,'model':name,
                               'source_observed_stale':source_y,'reference_observed_stale':reference_y,
                               'raw_model_lower':low,'raw_model_upper':high,'reference_model':donor_predictions[name],
                               'anchored_nominal':anchored_nominal,'anchored_lower':after_low,'anchored_upper':after_high,
                               'upper_margin_to_source':source_y-after_high,'model_upper_nonworsening':bool(after_high<=source_y+1e-10),
                               'bound_kind':'conservative_forest_sum_or_exact_transformed_ridge_range_not_confidence_interval'})
                sample_checks.append({'beer':row.beer,'radius':radius,'model':name,'sample_min':float(pred.min()),'sample_max':float(pred.max()),
                                      'n_samples':len(samples),'all_samples_inside_model_bounds':True})
                margins.append(source_y-after_high);upper_values.append(after_high)
            results.append({'beer':row.beer,'radius':radius,'all_model_upper_bounds_passed':bool(min(margins)>=-1e-10),
                            'worst_upper_margin':min(margins),'maximum_anchored_upper':max(upper_values),
                            'n_nonzero_width_features':int((hi>lo).sum()),'entire_box_style_verified':False,
                            'actual_outcome_certified':False,'radius_is_measured_assay_error':False})
        print(f'Box sensitivity beer {row.beer} complete',flush=True)
    save_csv(output,'model_bounds.csv',detail);save_csv(output,'coordinate_boxes.csv',boxes)
    save_csv(output,'scenario_results.csv',results);save_csv(output,'spot_checks.csv',sample_checks)
    frame=pd.DataFrame(results);counts=frame.groupby('radius').all_model_upper_bounds_passed.agg(['sum','count']).reset_index()
    save_csv(output,'summary_by_radius.csv',counts)
    write_json(output/'summary.json',{'status':'completed','config_sha256':sha256(config_path),'source_sha256':parent['source_sha256'],
        'q1':str(q1.resolve()),'style_models':str(style_dir.resolve()),'resolution':str(resolution.resolve()),'reference':str(reference.resolve()),
        'parent_summary_sha256':sha256(reference/'summary.json'),'n_model_box_bounds':len(detail),
        'model_bounds_scope':'Fitted prediction functions only; reference error-invariance assumption remains unvalidated.',
        'limitations':[cfg['scope'],'Forest sum-of-leaf extrema can be conservative because different trees need not attain their maxima together.',
                       'Failure of a conservative bound does not prove that any point violates the model guard.',
                       'No label uncertainty, causal error, mass balance or fermentation constraints are included.']})
    print(counts.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','q1','style-models','resolution','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/q3_reference_box_protocol.json'))
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.q1,a.style_models,a.resolution,a.reference,a.output,a.config)
