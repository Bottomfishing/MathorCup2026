"""Add an explicit model-box criterion to existing reference candidates.

AI-assisted: Codex, 2026-10-24. No physical or statistical certification.
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


def run(source,q1,style,resolution,reference,output,config_path):
    cfg=json.loads(config_path.read_text(encoding='utf-8'))
    if json.loads((reference/'verification.json').read_text())['status']!='passed':raise ValueError('Parent not verified')
    parent=json.loads((reference/'summary.json').read_text(encoding='utf-8'))
    if sha256(resolution/'summary.json')!=parent['parent_resolution_summary_sha256']:raise ValueError('Parent differs')
    resolved=json.loads((resolution/'summary.json').read_text(encoding='utf-8'))
    for name,digest in resolved['style_models_sha256'].items():
        if sha256(style/(name+'.joblib'))!=digest:raise ValueError('Style model differs')
    ids,xf,yf,bundle,_=load_q1_artifacts(source,Path('outputs/p0_audit_20261023_final'),q1)
    models={'selected':bundle['model'],'ridge':bundle['guard'],'focused':joblib.load(style/'focused_stale.joblib')}
    x=xf.to_numpy(float);beers=ids.beer.astype(str).to_numpy();lookup={b:i for i,b in enumerate(beers)}
    g=StyleGenerator().fit(x,ids.tasting_category_fine.to_numpy());tripel=g.x_[ids.tasting_category_fine.eq('Tripel').to_numpy()]
    center=g.robust_.transform(tripel).mean(axis=0);qlo,qhi=np.nanquantile(x,[.25,.75],axis=0);iqr=qhi-qlo
    protected=[xf.columns.get_loc(f) for f in cfg['protected_features']];other=[j for j in range(len(xf.columns)) if j not in protected]
    identities=[xf.columns.get_loc(f) for f in ['X2.3.butanedione','X.E..2.nonenal']];stale=list(yf).index('stale_hops')
    evidence=pd.read_csv(resolution/'source_defect_evidence.csv',dtype={'beer':str}).set_index(['beer','field'])
    original=pd.read_csv(reference/'all_reference_pairs.csv',dtype={'beer':str,'reference_beer':str})
    nominal=pd.read_csv(reference/'transfer_results.csv',dtype={'beer':str,'reference_beer':str}).set_index('beer')
    output.mkdir(parents=True,exist_ok=False);write_json(output/'config.json',cfg)
    candidates=[];bound_rows=[];results=[];profiles=[];sensory=[]
    for beer,group in original.loc[original.passed_all].groupby('beer',sort=False):
        i=lookup[beer];source_row=g.x_[i].copy();source_y=float(yf.stale_hops.iloc[i])
        source_caps=np.array([evidence.loc[(beer,xf.columns[j]),'constraint_cap_coordinate'] for j in protected])
        source_row[protected]=source_caps;zsource=g.robust_.transform(source_row[None,:])[0]
        choices=[]
        for pair in group.itertuples():
            d=lookup[pair.reference_beer];donor=x[d];assert np.isfinite(donor).all()
            candidate=donor.copy();caps=np.minimum(source_caps,donor[protected]);candidate[protected]=caps
            radius=cfg['radius_training_iqr_fraction'];lo=candidate-radius*iqr;hi=np.minimum(candidate+radius*iqr,tripel.max(axis=0))
            lo[other]=np.maximum(lo[other],tripel.min(axis=0)[other]);hi[protected]=np.minimum(hi[protected],caps);lo[identities]=np.maximum(lo[identities],0)
            if (lo>hi).any():raise ValueError('Invalid scenario box')
            margins=[]
            for name,model in models.items():
                target=0 if name=='focused' else stale
                low,high=prediction_bounds(model,lo,hi,target)
                v=model.predict(donor[None,:]);base=float(v[0] if name=='focused' else v[0,stale])
                upper=float(yf.stale_hops.iloc[d]+high-base);margin=source_y-upper;margins.append(margin)
                bound_rows.append({'beer':beer,'reference_beer':pair.reference_beer,'model':name,
                                   'raw_model_lower':low,'raw_model_upper':high,'reference_model':base,
                                   'anchored_upper':upper,'upper_margin_to_source':margin})
            z=g.robust_.transform(candidate[None,:])[0];distance=float(np.sqrt(np.mean((z-center)**2)));change=float(np.sqrt(np.mean((z-zsource)**2)))
            w=cfg['target_distance_weight'];objective=w*distance+(1-w)*change
            row={'beer':beer,'reference_beer':pair.reference_beer,'radius':radius,'target_distance':distance,'change_distance':change,
                 'objective':objective,'worst_upper_margin':min(margins),'all_model_bounds_passed':bool(min(margins)>=-1e-10)}
            candidates.append(row)
            if row['all_model_bounds_passed']:choices.append((objective,row,candidate,d))
        if not choices:
            results.append({'beer':beer,'selected':False,'status':'no_box_checked_candidate','physical_constraints_certified':False});continue
        _,row,candidate,d=min(choices,key=lambda c:c[0])
        source_distance=float(np.sqrt(np.mean((zsource-center)**2)))
        results.append(row|{'selected':True,'status':'reference_model_box_candidate','source_target_distance':source_distance,
                            'distance_reduction_fraction':1-row['target_distance']/source_distance,
                            'previous_reference_beer':nominal.loc[beer,'reference_beer'],
                            'reference_changed':row['reference_beer']!=nominal.loc[beer,'reference_beer'],
                            'entire_box_style_verified':False,'physical_constraints_certified':False,
                            'actual_after_stale_measured':False})
        profiles.append(dict(zip(xf.columns,candidate),beer=beer))
        before=bundle['model'].predict(x[[d]])[0];after=bundle['model'].predict(candidate[None,:])[0]
        for j,target in enumerate(yf):
            sensory.append({'beer':beer,'reference_beer':row['reference_beer'],'target':target,
                            'reference_observed':yf.iloc[d,j],'reference_model':before[j],'after_model':after[j],
                            'reference_anchored_after':yf.iloc[d,j]+after[j]-before[j]})
        print(f'Box-checked reference beer {beer} <- {row["reference_beer"]}: {len(choices)} choices',flush=True)
    for name,table in [('candidate_results',candidates),('candidate_model_bounds',bound_rows),('transfer_results',results),('transfer_profiles',profiles),('transfer_sensory',sensory)]:save_csv(output,name+'.csv',table)
    write_json(output/'summary.json',{'status':'completed','selected':sum(r['selected'] for r in results),'candidate_pairs':len(candidates),
        'source_sha256':parent['source_sha256'],'q1':str(q1.resolve()),'style_models':str(style.resolve()),'resolution':str(resolution.resolve()),'reference':str(reference.resolve()),
        'parent_summary_sha256':sha256(reference/'summary.json'),'config_sha256':sha256(config_path),'script_sha256':sha256(Path(__file__)),
        'bounds_module_sha256':sha256(Path('src/beer_models/interval_bounds.py')),'limitations':[cfg['scope'],cfg['selection_note'],
        'Reference anchoring still lacks held-out predictive support and paired intervention validation; model-box checks do not remove that limitation.']})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','q1','style-models','resolution','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/q3_reference_robust_protocol.json'));a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.q1,a.style_models,a.resolution,a.reference,a.output,a.config)
