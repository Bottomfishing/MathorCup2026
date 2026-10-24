"""Replay each saved fitted-model witness; do not infer physical effects."""
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


def verify(source,out):
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'));parent=Path(summary['parent'])
    assert sha256(parent/'summary.json')==summary['parent_summary_sha256']
    s=json.loads((parent/'summary.json').read_text(encoding='utf-8'));cfg=json.loads((parent/'config.json').read_text(encoding='utf-8'))
    q1=Path(s['q1']);style=Path(s['style_models']);resolution=Path(s['resolution'])
    ids,xf,yf,bundle,_=load_q1_artifacts(source,Path('outputs/p0_audit_20261023_final'),q1)
    x=xf.to_numpy(float);lookup={str(b):i for i,b in enumerate(ids.beer)};stale=list(yf).index('stale_hops')
    g=StyleGenerator().fit(x,ids.tasting_category_fine.to_numpy());tripel=g.x_[ids.tasting_category_fine.eq('Tripel').to_numpy()]
    qlo,qhi=np.nanquantile(x,[.25,.75],axis=0);iqr=qhi-qlo
    protect=[xf.columns.get_loc(f) for f in cfg['protected_features']];other=[j for j in range(len(xf.columns)) if j not in protect]
    identities=[xf.columns.get_loc(f) for f in ['X2.3.butanedione','X.E..2.nonenal']]
    evidence=pd.read_csv(resolution/'source_defect_evidence.csv',dtype={'beer':str}).set_index(['beer','field'])
    models={'selected':bundle['model'],'ridge':bundle['guard'],'focused':joblib.load(style/'focused_stale.joblib')}
    classifiers={n:joblib.load(style/(n+'.joblib')) for n in ['logistic','extra_trees']}
    results=pd.read_csv(out/'search_results.csv',dtype={'beer':str,'reference_beer':str});points=pd.read_csv(out/'searched_profiles.csv').set_index('witness_id')
    bounds=pd.read_csv(parent/'candidate_model_bounds.csv',dtype={'beer':str,'reference_beer':str});failed=bounds.loc[bounds.upper_margin_to_source.lt(-1e-10)]
    assert set(zip(results.beer,results.reference_beer,results.model))==set(zip(failed.beer,failed.reference_beer,failed.model))
    witnesses=0
    for row in results.itertuples():
        i=lookup[row.beer];d=lookup[row.reference_beer];donor=x[d];center=donor.copy()
        caps=np.array([min(evidence.loc[(row.beer,xf.columns[j]),'constraint_cap_coordinate'],donor[j]) for j in protect]);center[protect]=caps
        radius=cfg['radius_training_iqr_fraction'];lo=center-radius*iqr;hi=np.minimum(center+radius*iqr,tripel.max(axis=0))
        lo[other]=np.maximum(lo[other],tripel.min(axis=0)[other]);hi[protect]=np.minimum(hi[protect],caps);lo[identities]=np.maximum(lo[identities],0)
        point=points.loc[row.witness_id,xf.columns].to_numpy(float)
        assert np.isfinite(point).all() and (point>=lo-1e-12).all() and (point<=hi+1e-12).all()
        assert (point[protect]<=caps+1e-12).all()
        model=models[row.model];prediction=model.predict(np.vstack([donor,point]));prediction=prediction if row.model=='focused' else prediction[:,stale]
        anchored=float(yf.stale_hops.iloc[d]+prediction[1]-prediction[0]);violation=anchored-float(yf.stale_hops.iloc[i])
        np.testing.assert_allclose([row.raw_prediction,row.reference_model,row.anchored_prediction,row.violation],
                                   [prediction[1],prediction[0],anchored,violation],atol=1e-10)
        if violation>1e-10:assert row.search_status=='actual_model_guard_witness';witnesses+=1
        else:assert row.search_status=='no_witness_found_not_a_safety_proof'
        for name,classifier in classifiers.items():assert getattr(row,name+'_label')==classifier.predict(point[None,:])[0]
        assert not row.physical_or_intervention_certification
    assert witnesses==summary['witnesses_found']
    write_json(out/'verification.json',{'status':'passed','saved_points_replayed':len(results),'actual_fitted_guard_witnesses':witnesses,
        'source_reference_caps_and_box_checked':True,'physical_or_intervention_certification':False,
        'search_failure_is_safety_proof':False})
    print(f'Checked {len(results)} points: {witnesses} actual fitted-model guard witnesses.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):verify(a.source,a.output)
