"""Recompute every model-box candidate and selection. AI-assisted: Codex, 2026-10-24."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256,write_json
from beer_models.generation import StyleGenerator
from complete_modeling import load_q1_artifacts
from verify_q3_reference_box import independent_bounds


def verify(source,out):
    s=json.loads((out/'summary.json').read_text(encoding='utf-8'));cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    q1=Path(s['q1']);style=Path(s['style_models']);resolution=Path(s['resolution']);reference=Path(s['reference'])
    assert sha256(reference/'summary.json')==s['parent_summary_sha256']
    resolved=json.loads((resolution/'summary.json').read_text(encoding='utf-8'))
    for name,digest in resolved['style_models_sha256'].items():assert sha256(style/(name+'.joblib'))==digest
    ids,xf,yf,bundle,_=load_q1_artifacts(source,Path('outputs/p0_audit_20261023_final'),q1)
    x=xf.to_numpy(float);beers=ids.beer.astype(str).to_numpy();lookup={b:i for i,b in enumerate(beers)}
    g=StyleGenerator().fit(x,ids.tasting_category_fine.to_numpy());tripel=g.x_[ids.tasting_category_fine.eq('Tripel').to_numpy()]
    center=g.robust_.transform(tripel).mean(axis=0);qlo,qhi=np.nanquantile(x,[.25,.75],axis=0);iqr=qhi-qlo
    protected=[xf.columns.get_loc(f) for f in cfg['protected_features']];other=[j for j in range(len(xf.columns)) if j not in protected]
    identities=[xf.columns.get_loc(f) for f in ['X2.3.butanedione','X.E..2.nonenal']];stale=list(yf).index('stale_hops')
    models={'selected':bundle['model'],'ridge':bundle['guard'],'focused':joblib.load(style/'focused_stale.joblib')}
    evidence=pd.read_csv(resolution/'source_defect_evidence.csv',dtype={'beer':str}).set_index(['beer','field'])
    original=pd.read_csv(reference/'all_reference_pairs.csv',dtype={'beer':str,'reference_beer':str});original=original.loc[original.passed_all]
    candidates=pd.read_csv(out/'candidate_results.csv',dtype={'beer':str,'reference_beer':str}).set_index(['beer','reference_beer'])
    assert set(candidates.index)==set(zip(original.beer,original.reference_beer))
    bounds=pd.read_csv(out/'candidate_model_bounds.csv',dtype={'beer':str,'reference_beer':str}).set_index(['beer','reference_beer','model'])
    results=pd.read_csv(out/'transfer_results.csv',dtype={'beer':str,'reference_beer':str}).set_index('beer')
    profiles=pd.read_csv(out/'transfer_profiles.csv',dtype={'beer':str}).set_index('beer')
    sensory=pd.read_csv(out/'transfer_sensory.csv',dtype={'beer':str}).set_index(['beer','target'])
    valid={};all_profiles={}
    for row in original.itertuples():
        beer,reference_beer=row.beer,row.reference_beer;i=lookup[beer];d=lookup[reference_beer];donor=x[d]
        source_row=g.x_[i].copy();caps=np.array([evidence.loc[(beer,xf.columns[j]),'constraint_cap_coordinate'] for j in protected]);source_row[protected]=caps
        profile=donor.copy();profile[protected]=np.minimum(caps,donor[protected]);all_profiles[beer,reference_beer]=profile
        radius=cfg['radius_training_iqr_fraction'];lo=profile-radius*iqr;hi=np.minimum(profile+radius*iqr,tripel.max(axis=0))
        lo[other]=np.maximum(lo[other],tripel.min(axis=0)[other]);lo[identities]=np.maximum(lo[identities],0);hi[protected]=np.minimum(hi[protected],profile[protected])
        margins=[]
        for name,model in models.items():
            target=0 if name=='focused' else stale;low,high=independent_bounds(model,lo,hi,target)
            base=model.predict(donor[None,:]);base=float(base[0] if name=='focused' else base[0,stale])
            upper=float(yf.stale_hops.iloc[d]+high-base);margin=float(yf.stale_hops.iloc[i]-upper)
            b=bounds.loc[(beer,reference_beer,name)]
            np.testing.assert_allclose([b.raw_model_lower,b.raw_model_upper,b.reference_model,b.anchored_upper,b.upper_margin_to_source],
                                       [low,high,base,upper,margin],atol=1e-10);margins.append(margin)
        z=g.robust_.transform(profile[None,:])[0];source_z=g.robust_.transform(source_row[None,:])[0]
        distance=float(np.sqrt(np.mean((z-center)**2)));change=float(np.sqrt(np.mean((z-source_z)**2)));w=cfg['target_distance_weight'];objective=w*distance+(1-w)*change
        c=candidates.loc[(beer,reference_beer)]
        np.testing.assert_allclose([c.target_distance,c.change_distance,c.objective,c.worst_upper_margin],
                                   [distance,change,objective,min(margins)],atol=1e-10)
        assert c.all_model_bounds_passed==bool(min(margins)>=-1e-10)
        if c.all_model_bounds_passed:valid.setdefault(beer,[]).append((objective,reference_beer))
    for beer,row in results.iterrows():
        assert not row.physical_constraints_certified
        if beer not in valid:assert not row.selected and beer not in profiles.index;continue
        winner=min(valid[beer],key=lambda v:v[0])[1]
        assert row.selected and row.reference_beer==winner and not row.entire_box_style_verified
        expected=all_profiles[beer,winner];actual=profiles.loc[beer,xf.columns].to_numpy(float)
        np.testing.assert_allclose(actual,expected,atol=1e-10)
        d=lookup[winner];before=bundle['model'].predict(x[[d]])[0];after=bundle['model'].predict(actual[None,:])[0]
        for j,target in enumerate(yf):
            r=sensory.loc[(beer,target)]
            np.testing.assert_allclose([r.reference_model,r.after_model,r.reference_anchored_after],
                                       [before[j],after[j],yf.iloc[d,j]+after[j]-before[j]],atol=1e-10)
    write_json(out/'verification.json',{'status':'passed','candidate_pairs_independently_recomputed':len(candidates),
        'model_ranges_recomputed':len(bounds),'selected_profiles':len(profiles),'sensory_rows':len(sensory),
        'original_criterion_preserved':True,'physical_certification':False})
    print(f'Verified {len(candidates)} pairs, {len(bounds)} ranges, {len(profiles)} selected profiles.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):verify(a.source,a.output)
