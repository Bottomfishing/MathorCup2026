"""Recheck the original full path grid with current raw guards and a frozen historical control."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv,write_json,sha256
from beer_models.generation import StyleGenerator
from beer_models.style_models import make_style_model
from beer_models.defect_constraints import project_defects,defect_aware_box
from beer_models.q3_raw import raw_predictions,defect_guard
from r2_raw_q1 import load_inputs,RAW_MANIFEST,PROTOCOL as BASE_PROTOCOL
from beer_models.raw_panel import portrait


def run(args):
    cp=ROOT/'configs/q3_full_grid_protocol_20261026.json';p=json.loads(cp.read_text(encoding='utf-8'))
    d=load_inputs(args.source,args.raw,json.loads(BASE_PROTOCOL.read_text(encoding='utf-8')))
    q1_summary=json.loads((args.q1/'summary.json').read_text(encoding='utf-8'))
    if sha256(args.q1/'model.joblib')!=q1_summary['model_sha256']:raise ValueError('Q1 model hash mismatch')
    bundle=joblib.load(args.q1/'model.joblib')
    assert bundle['features']==d['features'] and bundle['targets']==d['targets']
    output=args.output;output.mkdir(parents=True,exist_ok=False);(output/'models').mkdir()
    write_json(output/'protocol.json',p)
    generator=StyleGenerator().fit(d['x'],d['ids'].tasting_category_fine.to_numpy())
    tripel=generator.x_[generator.indices_[p['target_style']]];lower,upper=tripel.min(0),tripel.max(0)
    center=generator.robust_.transform(tripel).mean(0)
    sc=json.loads((ROOT/'configs/q3_style_protocol.json').read_text(encoding='utf-8'))
    classifiers={name:make_style_model(name,sc,sc['seed']).fit(d['x'],d['ids'].tasting_category_fine.to_numpy()) for name in ['logistic','extra_trees']}
    for name,model in classifiers.items():joblib.dump(model,output/'models'/(name+'.joblib'),compress=3)
    read=lambda path:pd.read_csv(path,float_precision='round_trip',dtype={'beer':str})
    before_path=ROOT/'outputs/q3_task3_20261025/transfer_source_profiles.csv'
    before=read(before_path).set_index('beer')
    anchors_path=ROOT/p['anchors'];anchors=read(anchors_path).set_index('anchor')[d['features']].to_numpy(float)
    assert len(anchors)==129 and len(before)==11
    source_beers=d['ids'].loc[d['ids'].tasting_category_fine.isin(p['source_styles']),'beer'].astype(str)
    assert set(source_beers)==set(before.index)
    old_path=ROOT/'outputs/task4_raw_q3_20261026/migration_grid_checks.csv.gz';old=read(old_path)
    protected=[d['features'].index(t) for t in p['protected_features']];indices=[d['targets'].index(t) for t in p['defect_targets']]
    all_grid=[];results=[];selected_profiles=[];control=[];portraits=[]
    for beer,source_series in before.iterrows():
        source=source_series[d['features']].to_numpy(float)
        fractions=np.tile(np.arange(1,p['steps']+1)/p['steps'],len(anchors))
        anchor_ids=np.repeat(np.arange(len(anchors)),p['steps'])
        raw=source+fractions[:,None]*(anchors[anchor_ids]-source)
        values=np.vstack([project_defects(raw,source,protected,policy) for policy in p['policies']])
        policies=np.repeat(p['policies'],len(raw));fractions=np.tile(fractions,len(p['policies']));anchor_ids=np.tile(anchor_ids,len(p['policies']))
        bpred=raw_predictions(bundle,source[None],d['parents']);apred=raw_predictions(bundle,values,d['parents'])
        okay,deltas,passes=defect_guard(bpred,apred,indices,p['nonincrease_tolerance'])
        chemical=(values[:,protected]<=source[protected]+p['nonincrease_tolerance']).all(1)
        box=defect_aware_box(values,lower,upper,protected)
        style=np.logical_and.reduce([m.predict(values)==p['target_style'] for m in classifiers.values()])
        z=generator.robust_.transform(values);zs=generator.robust_.transform(source[None])[0]
        distance=np.sqrt(np.mean((z-center)**2,axis=1));change=np.sqrt(np.mean((z-zs)**2,axis=1))
        initial=float(np.sqrt(np.mean((zs-center)**2)));improves=distance<initial-1e-8
        structural=chemical&box&style&improves;good=structural&okay
        prior=old.loc[old.beer.eq(beer)].set_index('candidate')
        grid=pd.DataFrame(dict(beer=beer,candidate=np.arange(len(values)),anchor=anchor_ids,fraction=fractions,policy=policies,
            target_distance=distance,change_distance=change,range_pass=box,chemical_pass=chemical,style_pass=style,
            distance_improves=improves,historically_eligible=np.isin(np.arange(len(values)),prior.index),
            all_raw_grades_pass=passes['grade'],all_calibrated_risks_pass=passes['calibrated'],
            all_uncalibrated_risks_pass=passes['uncalibrated'],all_current_constraints_pass=good))
        for j,target in enumerate(p['defect_targets']):
            grid[target+'_grade_delta']=deltas['grade'][:,j]
            for kind in ['calibrated','uncalibrated']:
                for k,event in enumerate(['ge1','ge5','6']):grid[f'{target}_{kind}_delta_{event}']=deltas[kind][:,j,k]
        # Reproduce every old candidate's scientific fields before reporting new ones.
        shared=[c for c in prior if c in grid and c!='beer']
        actual=grid.set_index('candidate').loc[prior.index,shared]
        pd.testing.assert_frame_equal(actual,prior[shared],check_dtype=False,check_exact=False,atol=1e-10,rtol=1e-10)
        control.append(dict(beer=beer,historical_rows=len(prior),shared_columns=len(shared),status='passed'))
        all_grid.append(grid)
        row=dict(beer=beer,full_candidates=len(values),historical_candidates=len(prior),structurally_eligible=int(structural.sum()),
            raw_scenario_pass_before_structure=int(okay.sum()),complete_pass_count=int(good.sum()),
            new_complete_pass_count=int((good&~grid.historically_eligible.to_numpy()).sum()),selected=False,candidate=None,
            status='no_full_grid_candidate_passes_current_scenario',main_results_replaced=False,physical_feasibility_certified=False,
            actual_after_measured=False,refit_stability_validated=False)
        if good.any():
            eligible=np.flatnonzero(good);k=eligible[np.argmin(p['weight']*distance[eligible]+(1-p['weight'])*change[eligible])]
            row.update(selected=True,candidate=int(k),anchor=int(anchor_ids[k]),fraction=float(fractions[k]),policy=policies[k],
                target_distance=float(distance[k]),change_distance=float(change[k]),distance_reduction_fraction=1-distance[k]/initial,
                status='supplementary_current_model_candidate_requires_refit_validation')
            selected_profiles.append(dict(beer=beer,**dict(zip(d['features'],values[k]))))
            frame=pd.DataFrame([dict(beer=beer+'_before',**dict(zip(d['features'],source))),
                                dict(beer=beer+'_after',**dict(zip(d['features'],values[k])))])
            long,_,_,_=portrait(frame,bundle);long['source_beer']=beer;portraits.append(long)
        results.append(row)
        print(f'Full grid {beer}: {len(values)} checked, {int(good.sum())} complete passes',flush=True)
    grid=pd.concat(all_grid,ignore_index=True)
    grid.to_csv(output/'full_grid_checks.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    raw_only=grid.all_raw_grades_pass&grid.all_calibrated_risks_pass&grid.all_uncalibrated_risks_pass
    save_csv(output,'raw_only_candidates_rejected_by_structure.csv',grid.loc[raw_only&~grid.all_current_constraints_pass])
    save_csv(output,'transfer_results.csv',results)
    save_csv(output,'selected_profiles.csv',pd.DataFrame(selected_profiles,columns=['beer',*d['features']]))
    if portraits:pd.concat(portraits,ignore_index=True).to_csv(output/'selected_sensory52.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    save_csv(output,'historical_control.csv',control)
    write_json(output/'summary.json',dict(status='completed',protocol=p,protocol_sha256=sha256(cp),
        input_sha256={path.relative_to(ROOT).as_posix():sha256(path) for path in [before_path,anchors_path,old_path]},
        raw_source_sha256=sha256(args.raw),model_sha256=sha256(args.q1/'model.joblib'),
        source_sha256={name:sha256(args.source/name) for name in ['1.csv','4.csv','8.xlsx']},
        full_grid_size=len(grid),historical_control_size=len(old),selected_sources=sum(r['selected'] for r in results),
        complete_pass_count=int(grid.all_current_constraints_pass.sum()),main_results_replaced=False,
        physical_feasibility_certified=False,actual_after_measured=False,refit_stability_validated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','q1','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--raw',type=Path,default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    with threadpool_limits(limits=1):run(p.parse_args())
