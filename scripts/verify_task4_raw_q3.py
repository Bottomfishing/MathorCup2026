"""Replay Task4 fits, generation, current guards and fixed-path sensitivity."""
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
from r2_raw_q1 import load_inputs,content_hash
from task4_raw_q3 import (PROTOCOL,BASE_PROTOCOL,read,text_hash,run_generation,run_migration,refit_fixed_paths,parser)


def exported(table):return pd.read_csv(io.StringIO(pd.DataFrame(table).to_csv(index=False)),float_precision='round_trip')


def verify(args):
    output=args.output;s=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    p=json.loads(PROTOCOL.read_text(encoding='utf-8'));bp=json.loads(BASE_PROTOCOL.read_text(encoding='utf-8'))
    assert content_hash(p)==s['protocol_content_sha256']
    assert json.loads((output/'protocol.json').read_text(encoding='utf-8'))==p
    for name,digest in s['source_sha256'].items():assert sha256(args.source/name)==digest
    assert sha256(args.raw)==s['raw_source_sha256']
    for name,digest in s['input_lf_normalized_sha256'].items():assert text_hash(ROOT/name)==digest
    assert sha256(args.q1/'model.joblib')==s['q1_model_sha256']
    for name,digest in s['q1_outer_model_sha256'].items():assert sha256(args.q1/'models'/name)==digest
    d=load_inputs(args.source,args.raw,bp);bundle=joblib.load(args.q1/'model.joblib')
    cache=args.cache;cache.mkdir(parents=True,exist_ok=False);(cache/'models').mkdir()
    replay=argparse.Namespace(**{**vars(args),'output':cache})
    generation,selected,classifiers,generator=run_generation(d,p,bundle,replay)
    migration,before,after=run_migration(d,p,bundle,classifiers,generator,replay)
    refits=refit_fixed_paths(d,p,bp,before,after,replay)
    tables={**generation,**migration,**refits}
    for name,table in tables.items():
        actual=read(output/name);expected=exported(table)
        if name=='raw_refit_fit_log.csv':actual=actual.drop(columns='seconds');expected=expected.drop(columns='seconds')
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=0,atol=1e-10,obj=name)
    grid=read(output/'migration_grid_checks.csv.gz',dtype={'beer':str})
    choices=read(output/'weight_choices_complete.csv',dtype={'beer':str});tol=p['nonincrease_tolerance']
    for beer,g in grid.groupby('beer',sort=False):
        initial=float(choices.loc[choices.beer.eq(beer),'source_target_distance'].iloc[0])
        grade_cols=[t+'_grade_delta' for t in p['defect_targets']]
        prob_cols=[t+'_'+version+'_delta_'+cut for t in p['defect_targets'] for version in ['calibrated','uncalibrated'] for cut in ['ge1','ge5','6']]
        direct=g.chemical_pass&g.range_pass&g.style_pass&(g[grade_cols]<=0).all(axis=1)&(g[prob_cols]<=tol).all(axis=1)&(g.target_distance<initial-1e-8)
        np.testing.assert_array_equal(g.all_current_constraints_pass,direct)
        feasible=g[direct]
        for row in choices[choices.beer.eq(beer)].itertuples():
            assert row.n_pass_current==len(feasible)
            historical=g[g.candidate.eq(row.original_weight_candidate)].iloc[0]
            assert row.original_weight_pass_current==historical.all_current_constraints_pass
            if not len(feasible):assert not row.selected
            else:
                k=(row.weight*feasible.target_distance+(1-row.weight)*feasible.change_distance).idxmin()
                chosen=feasible.loc[k];assert row.selected and int(row.candidate)==int(chosen.candidate)
                costs=feasible[['target_distance','change_distance']].to_numpy(float)
                point=chosen[['target_distance','change_distance']].to_numpy(float)
                assert not np.any(np.all(costs<=point,axis=1)&np.any(costs<point,axis=1))
    long=read(output/'generated_sensory52.csv.gz');assert len(long)==15600
    missing=long.target.isin(['A_acetaldehyde','F_acetaldehyde']);assert missing.sum()==600
    assert long.loc[missing,'estimated_grade_0_6'].isna().all()
    assert long.loc[missing,'probability_rating_ge1'].isna().all() and not long.safety_certified.any()
    all_synthetic=read(output/'generation_candidate_sensory52.csv.gz');assert len(all_synthetic)==31200
    checks=read(output/'original_weight_five_defect_checks.csv');assert len(checks)==77*5
    fixed=read(output/'original_fixed_path_five_defect_checks.csv');assert len(fixed)==55
    assert not checks.safety_certified.any() and not fixed.actual_after_measured.any()
    assert selected==s['selected_generation_method']
    assert int(migration['transfer_results.csv'].selected.sum())==s['main_scenario_selected']
    result=dict(status='passed',compared_tables=len(tables),source_beers=len(d['x']),features=len(d['features']),
        generation_validation_cases=len(generation['generation_validation.csv']),generator_refits=8,style_classifier_refits=8,
        raw_ET_bootstrap_refits=p['refit_repeats'],fixed_historical_paths=11,historical_pool_rows=len(grid),
        checked_weight_choices=len(choices),checked_weight_five_defect_rows=len(checks),
        generated_scorecard_rows=len(long),unlabelled_generated_rows=int(missing.sum()),
        selected_generation_method=selected,main_scenario_selected=s['main_scenario_selected'],
        complete_constraints_and_selected_weight_optima_independently_recomputed=True,
        parent_raw_Q1_models_reused_not_refitted=True,legacy_200_refits_reused_not_repeated=True,
        actual_after_measured=False,safety_certified=False,
        scope='Numerical replay on same data, frozen pool and parameters; not external validation or intervention confirmation.')
    write_json(output/'verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=parser();p.add_argument('--cache',type=Path,default=ROOT/'checkpoints/task4_replay_20261026')
    with threadpool_limits(limits=1):verify(p.parse_args())
