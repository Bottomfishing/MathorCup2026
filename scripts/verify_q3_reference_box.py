"""Independent path-intersection checks for model interval scenarios.

AI-assisted: Codex, 2026-10-24. This verifies mathematical model ranges only.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256,write_json
from beer_models.generation import StyleGenerator
from complete_modeling import load_q1_artifacts


def independent_bounds(model,lo,hi,target):
    if isinstance(model,TransformedTargetRegressor):
        low,high=independent_bounds(model.regressor_,lo,hi,target)
        vectors=np.zeros((2,model.transformer_.n_features_in_));vectors[:,target]=[low,high]
        return tuple(model.transformer_.inverse_transform(vectors)[:,target])
    if isinstance(model,Pipeline):
        endpoints=model[:-1].transform(np.vstack([lo,hi]))
        return independent_bounds(model.steps[-1][1],endpoints[0],endpoints[1],target)
    if isinstance(model,Ridge):
        coef=np.asarray(model.coef_);intercept=np.asarray(model.intercept_)
        if coef.ndim==2:coef=coef[target];intercept=intercept[target]
        low_corner=np.where(coef<0,hi,lo);high_corner=np.where(coef<0,lo,hi)
        return float(intercept+coef@low_corner),float(intercept+coef@high_corner)
    # Traverse with accumulated path intersections, rather than production's
    # node-wise reachability tests against the original global endpoints.
    minima=[];maxima=[]
    for estimator in model.estimators_:
        tree=estimator.tree_;values=[]
        stack=[(0,lo.astype(np.float32).astype(float),hi.astype(np.float32).astype(float))]
        while stack:
            node,lower,upper=stack.pop()
            if (lower>upper).any():continue
            feature=tree.feature[node]
            if feature<0:values.append(tree.value[node,target,0]);continue
            threshold=tree.threshold[node]
            left_upper=upper.copy();left_upper[feature]=min(left_upper[feature],threshold)
            right_lower=lower.copy();right_lower[feature]=max(right_lower[feature],np.nextafter(threshold,np.inf))
            stack.append((tree.children_left[node],lower,left_upper))
            stack.append((tree.children_right[node],right_lower,upper))
        assert len(values)>0
        minima.append(min(values));maxima.append(max(values))
    return float(np.mean(minima)),float(np.mean(maxima))


def verify(source,out):
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'));cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
    q1=Path(summary['q1']);style=Path(summary['style_models']);resolution=Path(summary['resolution']);reference=Path(summary['reference'])
    assert sha256(reference/'summary.json')==summary['parent_summary_sha256']
    ids,xf,yf,bundle,_=load_q1_artifacts(source,Path('outputs/p0_audit_20261023_final'),q1)
    x=xf.to_numpy(float);lookup={str(b):i for i,b in enumerate(ids.beer)}
    g=StyleGenerator().fit(x,ids.tasting_category_fine.to_numpy());tripel=g.x_[ids.tasting_category_fine.eq('Tripel').to_numpy()]
    a,b=np.nanquantile(x,[.25,.75],axis=0);iqr=b-a
    protected=[xf.columns.get_loc(c) for c in cfg['protected_features']];other=[j for j in range(len(xf.columns)) if j not in protected]
    identities=[xf.columns.get_loc(c) for c in ['X2.3.butanedione','X.E..2.nonenal']]
    profiles=pd.read_csv(reference/'transfer_profiles.csv',dtype={'beer':str}).set_index('beer')
    transfers=pd.read_csv(reference/'transfer_results.csv',dtype={'beer':str,'reference_beer':str}).set_index('beer')
    evidence=pd.read_csv(resolution/'source_defect_evidence.csv',dtype={'beer':str}).set_index(['beer','field'])
    boxes=pd.read_csv(out/'coordinate_boxes.csv',dtype={'beer':str,'reference_beer':str})
    bounds=pd.read_csv(out/'model_bounds.csv',dtype={'beer':str}).set_index(['beer','radius','model'])
    results=pd.read_csv(out/'scenario_results.csv',dtype={'beer':str}).set_index(['beer','radius'])
    models={'selected':bundle['model'],'ridge':bundle['guard'],'focused':joblib.load(style/'focused_stale.joblib')};stale=list(yf).index('stale_hops')
    checked=0
    for row in boxes.itertuples(index=False,name=None):
        record=dict(zip(boxes.columns,row));beer=record['beer'];radius=record['radius'];donor=x[lookup[record['reference_beer']]]
        center=profiles.loc[beer,xf.columns].to_numpy(float)
        lo=np.array([record['lower__'+f] for f in xf]);hi=np.array([record['upper__'+f] for f in xf])
        expected_lo=center-radius*iqr;expected_hi=np.minimum(center+radius*iqr,tripel.max(axis=0))
        expected_lo[other]=np.maximum(expected_lo[other],tripel.min(axis=0)[other]);expected_lo[identities]=np.maximum(expected_lo[identities],0)
        caps=np.array([min(evidence.loc[(beer,xf.columns[j]),'constraint_cap_coordinate'],donor[j]) for j in protected])
        expected_hi[protected]=np.minimum(expected_hi[protected],caps)
        np.testing.assert_allclose(lo,expected_lo,atol=1e-12);np.testing.assert_allclose(hi,expected_hi,atol=1e-12)
        assert (lo<=hi).all() and (hi[protected]<=caps+1e-10).all()
        margins=[]
        for name,model in models.items():
            target=0 if name=='focused' else stale;recorded=bounds.loc[(beer,radius,name)]
            low,high=independent_bounds(model,lo,hi,target)
            np.testing.assert_allclose([recorded.raw_model_lower,recorded.raw_model_upper],[low,high],atol=1e-10)
            base=model.predict(donor[None,:]);base=float(base[0] if name=='focused' else base[0,stale])
            ry=float(yf.stale_hops.iloc[lookup[record['reference_beer']]]);sy=float(yf.stale_hops.iloc[lookup[beer]])
            upper=ry+high-base;lower=ry+low-base;margin=sy-upper
            np.testing.assert_allclose([recorded.anchored_lower,recorded.anchored_upper,recorded.upper_margin_to_source],[lower,upper,margin],atol=1e-10)
            assert recorded.model_upper_nonworsening==bool(upper<=sy+1e-10)
            margins.append(margin);checked+=1
        result=results.loc[(beer,radius)]
        assert result.all_model_upper_bounds_passed==bool(min(margins)>=-1e-10)
        assert not result.actual_outcome_certified and not result.radius_is_measured_assay_error and not result.entire_box_style_verified
    write_json(out/'verification.json',{'status':'passed','coordinate_boxes_reconstructed':len(boxes),
        'model_ranges_independently_recomputed':checked,'method':'accumulated tree-path intersections and affine corners',
        'physical_or_intervention_certification':False,'joint_style_verification':False})
    print(f'Verified {len(boxes)} boxes and {checked} fitted-model ranges.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):verify(a.source,a.output)
