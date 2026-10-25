"""Replay the current path grid and expose distance/change tradeoffs without changing it."""
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
from beer_models.defect_constraints import project_defects,defect_aware_box
from beer_models.review_diagnostics import pareto_mask
from complete_modeling import load_q1_artifacts


def run(source,q1,style_dir,output):
    config=json.loads(Path('configs/q3_resolution_protocol.json').read_text())
    cp=Path('configs/review_path_weights_20261025.json');protocol=json.loads(cp.read_text())
    ids,xf,yf,bundle,_=load_q1_artifacts(source,Path('outputs/p0_audit_20261023_final'),q1)
    output.mkdir(parents=True,exist_ok=False)
    generator=StyleGenerator().fit(xf.to_numpy(float),ids.tasting_category_fine.to_numpy())
    tripel=generator.x_[generator.indices_['Tripel']];lower,upper=tripel.min(0),tripel.max(0)
    center=generator.robust_.transform(tripel).mean(0)
    seed=json.loads((q1/'run_manifest.json').read_text())['config']['seed']+90000
    samples,_=generator.sample('Tripel',100,'interpolate',seed)
    anchors=np.vstack([tripel,tripel.mean(0),samples]);protected=[xf.columns.get_loc(f) for f in config['protected_features']]
    classifiers={n:joblib.load(style_dir/(n+'.joblib')) for n in ['logistic','extra_trees']}
    models={'multioutput':bundle['model'],'ridge':bundle['guard'],'focused':joblib.load(style_dir/'focused_stale.joblib')}
    current=Path('outputs/q3_task3_20261025');before=pd.read_csv(current/'transfer_source_profiles.csv',dtype={'beer':str}).set_index('beer')
    after=pd.read_csv(current/'transfer_profiles.csv',dtype={'beer':str}).set_index('beer')
    expected=pd.read_csv(current/'transfer_results.csv',dtype={'beer':str}).set_index('beer')
    rows=[];chosen=[];j=yf.columns.get_loc('stale_hops')
    for beer,source_row in before.iterrows():
        source_row=source_row[xf.columns].to_numpy(float)
        fractions=np.tile(np.linspace(1/config['transfer_steps'],1,config['transfer_steps']),len(anchors))
        parents=np.repeat(np.arange(len(anchors)),config['transfer_steps'])
        raw=source_row+fractions[:,None]*(anchors[parents]-source_row)
        candidates=np.vstack([project_defects(raw,source_row,protected,p) for p in config['defect_policies']])
        policies=np.repeat(config['defect_policies'],len(raw));fractions=np.tile(fractions,len(config['defect_policies']));parents=np.tile(parents,len(config['defect_policies']))
        good=defect_aware_box(candidates,lower,upper,protected)
        for model in classifiers.values():good &= model.predict(candidates)=='Tripel'
        for name,model in models.items():
            a=model.predict(candidates);b=model.predict(source_row[None])
            good &= (a-b[0] if name=='focused' else a[:,j]-b[0,j])<=1e-10
        z=generator.robust_.transform(candidates);zs=generator.robust_.transform(source_row[None])[0]
        distance=np.sqrt(np.mean((z-center)**2,axis=1));change=np.sqrt(np.mean((z-zs)**2,axis=1))
        original=float(np.sqrt(np.mean((zs-center)**2)));good &= distance<original-1e-8
        eligible=np.flatnonzero(good)
        if not len(eligible):raise ValueError('No feasible candidate for previously selected path: '+beer)
        costs=np.column_stack([distance[eligible],change[eligible]]);front=pareto_mask(costs)
        # Independent pairwise dominance check on this bounded feasible grid.
        direct=np.array([not np.any(np.all(costs<=p,axis=1)&np.any(costs<p,axis=1)) for p in costs])
        np.testing.assert_array_equal(front,direct)
        for k,on_front in zip(eligible,front):
            rows.append(dict(beer=beer,candidate=int(k),anchor=int(parents[k]),fraction=fractions[k],policy=policies[k],
                target_distance=distance[k],change_distance=change[k],pareto=bool(on_front)))
        main_k=eligible[np.argmin(.7*distance[eligible]+.3*change[eligible])]
        np.testing.assert_allclose(candidates[main_k],after.loc[beer,xf.columns],atol=1e-12,rtol=0)
        np.testing.assert_allclose([distance[main_k],change[main_k]],expected.loc[beer,['target_distance','change_distance']].to_numpy(float),atol=1e-12)
        for weight in protocol['weights']:
            k=eligible[np.argmin(weight*distance[eligible]+(1-weight)*change[eligible])]
            chosen.append(dict(beer=beer,weight=weight,candidate=int(k),anchor=int(parents[k]),fraction=fractions[k],policy=policies[k],
                target_distance=distance[k],change_distance=change[k],objective=weight*distance[k]+(1-weight)*change[k],
                same_as_main=k==main_k,distance_reduction_fraction=1-distance[k]/original))
        print(f'Weight replay {beer}: {len(eligible)} feasible, {sum(front)} Pareto',flush=True)
    save_csv(output,'feasible_candidates.csv',rows);save_csv(output,'weight_choices.csv',chosen)
    write_json(output/'verification.json',dict(status='passed',main_profiles_reproduced=len(before),
        pareto_checked_by_independent_pairwise_dominance=True,weights=protocol['weights']))
    write_json(output/'summary.json',dict(status='completed',protocol=protocol,config_sha256=sha256(cp),
        q1_manifest_sha256=sha256(q1/'run_manifest.json'),style_models_sha256={n:sha256(style_dir/(n+'.joblib')) for n in ['logistic','extra_trees','focused_stale']},
        frozen_inputs_sha256={n:sha256(current/n) for n in ['transfer_source_profiles.csv','transfer_profiles.csv','transfer_results.csv']},
        feasible_candidates=len(rows),main_results_replaced=False,actual_after_stale_validated=False,
        alternative_ordinal_grade_guard_checked=False,
        limitation='Replay of the original continuous-guard path grid. Alternative weight choices have not been screened by the later task2 ordinal-grade guard and are not replacement deliveries.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','q1','style-models','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.q1,a.style_models,a.output)
