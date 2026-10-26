"""Verify full-grid structure and gates; replay raw guards for fixed and selected points."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256,write_json
from beer_models.generation import StyleGenerator
from beer_models.tail_risk import survival
from r2_raw_q1 import load_inputs,RAW_MANIFEST,PROTOCOL as BASE_PROTOCOL


def verify(args):
    output=args.output;summary=json.loads((output/'summary.json').read_text(encoding='utf-8'));p=summary['protocol'];tol=p['nonincrease_tolerance']
    for name,digest in summary['input_sha256'].items():assert sha256(ROOT/name)==digest
    for name,digest in summary['source_sha256'].items():assert sha256(args.source/name)==digest
    assert sha256(args.raw)==summary['raw_source_sha256']
    assert sha256(args.q1/'model.joblib')==summary['model_sha256']
    assert sha256(ROOT/'configs/q3_full_grid_protocol_20261026.json')==summary['protocol_sha256']
    d=load_inputs(args.source,args.raw,json.loads(BASE_PROTOCOL.read_text(encoding='utf-8')))
    bundle=joblib.load(args.q1/'model.joblib');generator=StyleGenerator().fit(d['x'],d['ids'].tasting_category_fine.to_numpy())
    tripel=generator.x_[generator.indices_['Tripel']];lo,hi=tripel.min(0),tripel.max(0);center=generator.robust_.transform(tripel).mean(0)
    classifiers=[joblib.load(output/'models'/(name+'.joblib')) for name in ['logistic','extra_trees']]
    read=lambda path:pd.read_csv(path,float_precision='round_trip',dtype={'beer':str})
    grid=read(output/'full_grid_checks.csv.gz');results=read(output/'transfer_results.csv').set_index('beer')
    selected=read(output/'selected_profiles.csv').set_index('beer')
    before=read(ROOT/'outputs/q3_task3_20261025/transfer_source_profiles.csv').set_index('beer')
    anchors=read(ROOT/p['anchors']).set_index('anchor')[d['features']].to_numpy(float)
    old=read(ROOT/'outputs/task4_raw_q3_20261026/migration_grid_checks.csv.gz')
    n=len(anchors)*p['steps']*len(p['policies']);assert len(grid)==len(before)*n==113520
    assert not grid.duplicated(['beer','candidate']).any()
    protected=[d['features'].index(c) for c in p['protected_features']];other=[j for j in range(len(d['features'])) if j not in protected]
    targets=[d['targets'].index(t) for t in p['defect_targets']];checked_raw=0;control_rows=0
    for beer,g in grid.groupby('beer',sort=False):
        g=g.sort_values('candidate').reset_index(drop=True);np.testing.assert_array_equal(g.candidate,np.arange(n))
        cid=g.candidate.to_numpy();per_policy=len(anchors)*p['steps']
        np.testing.assert_array_equal(g.anchor,(cid%per_policy)//p['steps'])
        np.testing.assert_allclose(g.fraction,(cid%p['steps']+1)/p['steps'],rtol=0,atol=1e-15)
        assert g.policy.tolist()==[p['policies'][i//per_policy] for i in cid]
        src=before.loc[beer,d['features']].to_numpy(float)
        values=src+g.fraction.to_numpy()[:,None]*(anchors[g.anchor.to_numpy()]-src)
        preserve=g.policy.eq('preserve').to_numpy()
        for j in protected:
            values[preserve,j]=src[j];values[~preserve,j]=np.minimum(values[~preserve,j],src[j])
        box=(values<=hi+1e-10).all(1)&(values[:,other]>=lo[other]-1e-10).all(1)
        chemical=(values[:,protected]<=src[protected]+tol).all(1)
        style=np.logical_and.reduce([m.predict(values)=='Tripel' for m in classifiers])
        np.testing.assert_array_equal(g.range_pass,box);np.testing.assert_array_equal(g.chemical_pass,chemical);np.testing.assert_array_equal(g.style_pass,style)
        z=generator.robust_.transform(values);zs=generator.robust_.transform(src[None])[0]
        distance=np.sqrt(np.mean((z-center)**2,axis=1));change=np.sqrt(np.mean((z-zs)**2,axis=1));initial=np.sqrt(np.mean((zs-center)**2))
        np.testing.assert_allclose(g.target_distance,distance,atol=1e-12,rtol=0);np.testing.assert_allclose(g.change_distance,change,atol=1e-12,rtol=0)
        raw=(g[[t+'_grade_delta' for t in p['defect_targets']]]<=0).all(axis=1).to_numpy()
        prob=[f'{t}_{kind}_delta_{cut}' for t in p['defect_targets'] for kind in ['calibrated','uncalibrated'] for cut in ['ge1','ge5','6']]
        assert np.isfinite(g[prob].to_numpy()).all()
        cal=(g[[c for c in prob if '_calibrated_' in c]]<=tol).all(axis=1).to_numpy()
        uncal=(g[[c for c in prob if '_uncalibrated_' in c]]<=tol).all(axis=1).to_numpy()
        good=box&chemical&style&(distance<initial-1e-8)&raw&cal&uncal
        for col,truth in [('all_raw_grades_pass',raw),('all_calibrated_risks_pass',cal),('all_uncalibrated_risks_pass',uncal),('all_current_constraints_pass',good)]:np.testing.assert_array_equal(g[col],truth)
        prior=old.loc[old.beer.eq(beer)].set_index('candidate');shared=[c for c in prior if c in g and c!='beer']
        pd.testing.assert_frame_equal(g.set_index('candidate').loc[prior.index,shared],prior[shared],check_dtype=False,check_exact=False,atol=1e-10,rtol=1e-10)
        np.testing.assert_array_equal(g.historically_eligible,np.isin(cid,prior.index));control_rows+=len(prior)
        chosen=[]
        if good.any():
            eligible=np.flatnonzero(good);best=min(eligible,key=lambda k:p['weight']*distance[k]+(1-p['weight'])*change[k]);chosen=[best]
            assert int(results.loc[beer,'candidate'])==best and results.loc[beer,'selected']
            np.testing.assert_allclose(selected.loc[beer,d['features']].to_numpy(float),values[best],rtol=0,atol=1e-12)
        else:assert not results.loc[beer,'selected'] and beer not in selected.index
        assert results.loc[beer,'complete_pass_count']==sum(good)
        sample=np.unique([0,n//2,n-1,*chosen]);xp=np.vstack([src,values[sample]])
        grades=bundle['raw_panel_model'].predict(xp)['grade'][:,targets]
        np.testing.assert_array_equal(g.iloc[sample][[t+'_grade_delta' for t in p['defect_targets']]],grades[1:]-grades[0])
        for kind,method in [('calibrated','predict_distribution'),('uncalibrated','predict_uncalibrated_distribution')]:
            distribution=getattr(bundle['risk_panel_model'],method)(xp)
            # Independent survival sums instead of the generator's defect_guard.
            for j,target in zip(targets,p['defect_targets']):
                for cut,event in [(1,'ge1'),(5,'ge5'),(6,'6')]:
                    a=distribution[:,j,cut:].sum(1)
                    np.testing.assert_allclose(g.iloc[sample][f'{target}_{kind}_delta_{event}'],a[1:]-a[0],rtol=0,atol=1e-10)
        checked_raw+=len(sample)
    assert int(results.selected.sum())==summary['selected_sources'] and int(grid.all_current_constraints_pass.sum())==summary['complete_pass_count']
    raw_only=grid.all_raw_grades_pass&grid.all_calibrated_risks_pass&grid.all_uncalibrated_risks_pass
    pd.testing.assert_frame_equal(read(output/'raw_only_candidates_rejected_by_structure.csv'),
        grid.loc[raw_only&~grid.all_current_constraints_pass].reset_index(drop=True),check_dtype=False,check_exact=False,atol=1e-12,rtol=0)
    write_json(output/'verification.json',dict(status='passed',full_grid_rows=len(grid),historical_control_rows=control_rows,
        independently_reconstructed_spectra=len(grid),style_predictions_replayed=len(grid)*2,
        raw_prediction_points_replayed=checked_raw,selected_sources=int(results.selected.sum()),
        scope='All path identities, coordinates, distances, structural gates, saved raw-gate logic and selections verified. All 8293 historical controls matched. Raw predictions replayed at three fixed grid points per source and any selected point, not every newly explored point. No physical or refit-stability validation.'))
    print('Full-grid verification passed.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','q1','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--raw',type=Path,default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    with threadpool_limits(limits=1):verify(p.parse_args())
