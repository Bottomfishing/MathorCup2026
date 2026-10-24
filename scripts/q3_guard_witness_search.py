"""Bounded search for actual fitted-model violations, never a safety proof.

AI-assisted: Codex, 2026-10-24. Only examines candidates rejected by the
existing conservative 1%-IQR model-box check. Failure to find a witness is
reported as unresolved; no constraints or radii are changed.
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
from beer_models.generation import StyleGenerator
from complete_modeling import load_q1_artifacts


def run(source,robust,out):
    s=json.loads((robust/'summary.json').read_text(encoding='utf-8'));cfg=json.loads((robust/'config.json').read_text(encoding='utf-8'))
    if json.loads((robust/'verification.json').read_text())['status']!='passed':raise ValueError('Unverified candidate study')
    q1=Path(s['q1']);style=Path(s['style_models']);resolution=Path(s['resolution'])
    ids,xf,yf,bundle,_=load_q1_artifacts(source,Path('outputs/p0_audit_20261023_final'),q1)
    x=xf.to_numpy(float);lookup={str(b):i for i,b in enumerate(ids.beer)};stale=list(yf).index('stale_hops')
    g=StyleGenerator().fit(x,ids.tasting_category_fine.to_numpy());tripel=g.x_[ids.tasting_category_fine.eq('Tripel').to_numpy()]
    qlo,qhi=np.nanquantile(x,[.25,.75],axis=0);iqr=qhi-qlo
    protect=[xf.columns.get_loc(f) for f in cfg['protected_features']];other=[j for j in range(len(xf.columns)) if j not in protect]
    identities=[xf.columns.get_loc(f) for f in ['X2.3.butanedione','X.E..2.nonenal']]
    evidence=pd.read_csv(resolution/'source_defect_evidence.csv',dtype={'beer':str}).set_index(['beer','field'])
    models={'selected':bundle['model'],'ridge':bundle['guard'],'focused':joblib.load(style/'focused_stale.joblib')}
    classifiers={name:joblib.load(style/(name+'.joblib')) for name in ['logistic','extra_trees']}
    rows=pd.read_csv(robust/'candidate_model_bounds.csv',dtype={'beer':str,'reference_beer':str});rows=rows.loc[rows.upper_margin_to_source.lt(-1e-10)]
    out.mkdir(parents=True,exist_ok=False);results=[];points=[]
    for ri,row in enumerate(rows.itertuples()):
        i=lookup[row.beer];d=lookup[row.reference_beer];donor=x[d];center=donor.copy()
        caps=np.array([min(evidence.loc[(row.beer,xf.columns[j]),'constraint_cap_coordinate'],donor[j]) for j in protect]);center[protect]=caps
        radius=cfg['radius_training_iqr_fraction'];lo=center-radius*iqr;hi=np.minimum(center+radius*iqr,tripel.max(axis=0))
        lo[other]=np.maximum(lo[other],tripel.min(axis=0)[other]);hi[protect]=np.minimum(hi[protect],caps);lo[identities]=np.maximum(lo[identities],0)
        model=models[row.model];target=0 if row.model=='focused' else stale
        def predict(values):
            p=model.predict(values);return p if row.model=='focused' else p[:,target]
        base=float(predict(donor[None,:])[0]);limit=float(yf.stale_hops.iloc[i]-yf.stale_hops.iloc[d]+base)
        rng=np.random.default_rng(20261024+ri)
        population=np.vstack([center,lo,hi,rng.uniform(lo,hi,size=(64,len(center)))])
        values=predict(population);best=int(np.argmax(values));point=population[best].copy();value=float(values[best]);iterations=0
        if row.model=='ridge':
            coef=model.named_steps['model'].coef_[target];point=np.where(coef>=0,hi,lo);value=float(predict(point[None,:])[0])
        else:
            for iteration in range(24):
                if value>limit+1e-10:break
                trials=np.tile(point,(3*len(center),1))
                for j in range(len(center)):trials[3*j:3*j+3,j]=[lo[j],(lo[j]+hi[j])/2,hi[j]]
                predicted=predict(trials);best=int(np.argmax(predicted));iterations=iteration+1
                if predicted[best]<=value+1e-12:break
                point=trials[best].copy();value=float(predicted[best])
        assert (point>=lo-1e-12).all() and (point<=hi+1e-12).all() and (point[protect]<=caps+1e-12).all()
        anchored=float(yf.stale_hops.iloc[d]+value-base);violation=anchored-float(yf.stale_hops.iloc[i]);witness=violation>1e-10
        witness_id=f'pair_{ri:03d}_{row.beer}_{row.reference_beer}_{row.model}'
        results.append({'witness_id':witness_id,'beer':row.beer,'reference_beer':row.reference_beer,'model':row.model,'radius':radius,
                        'search_status':'actual_model_guard_witness' if witness else 'no_witness_found_not_a_safety_proof',
                        'source_observed_stale':float(yf.stale_hops.iloc[i]),'reference_observed_stale':float(yf.stale_hops.iloc[d]),
                        'raw_prediction':value,'reference_model':base,'anchored_prediction':anchored,'violation':violation,
                        'greedy_iterations':iterations,'logistic_label':classifiers['logistic'].predict(point[None,:])[0],
                        'extra_trees_label':classifiers['extra_trees'].predict(point[None,:])[0],
                        'physical_or_intervention_certification':False})
        points.append({'witness_id':witness_id,**dict(zip(xf.columns,point))})
        print(f'Guard search {row.beer}/{row.reference_beer}/{row.model}: {results[-1]["search_status"]}',flush=True)
    save_csv(out,'search_results.csv',results);save_csv(out,'searched_profiles.csv',points)
    write_json(out/'summary.json',{'status':'completed','parent':str(robust.resolve()),'parent_summary_sha256':sha256(robust/'summary.json'),
        'seed':20261024,'initial_random_points':64,'max_greedy_iterations':24,'tested_model_pairs':len(results),
        'witnesses_found':sum(r['violation']>1e-10 for r in results),'script_sha256':sha256(Path(__file__)),
        'interpretation':'A found point disproves only a fitted-model guard over this coordinate scenario. An unsuccessful finite search proves neither model safety nor physical safety. Radius and original candidate family are unchanged.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--robust',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.robust,a.output)
