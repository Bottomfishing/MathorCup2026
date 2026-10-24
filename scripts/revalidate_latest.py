"""Rebuild and verify the post-delivery Q3 extensions. AI-assisted: Codex, 2026-10-24.

The original reproduce_delivery.py deliberately verifies its older delivery base.
This runner additionally covers style models, repaired paths and measured references.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def run(source, output, reuse_base=False, include_q1_q2=False, include_diagnostics=False):
    project=Path(__file__).resolve().parents[1]
    source,output=source.resolve(),output.resolve()
    os.chdir(project)
    if reuse_base:
        if not (output/'modeling/run_manifest.json').is_file():raise ValueError('No completed base to reuse')
        if (output/'latest_revalidation.json').exists():raise FileExistsError('Report already exists')
    else:output.mkdir(parents=True,exist_ok=False)
    logs=output/'latest_logs';logs.mkdir()
    env=dict(os.environ,PYTHONPATH=str(project/'src'),PYTHONIOENCODING='utf-8')
    report={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','joblib']},
            'reused_base':reuse_base,'include_q1_q2':include_q1_q2,'include_diagnostics':include_diagnostics,'steps':[],'table_comparisons':[],
            'scope':'Local environment rebuild and independent verification of current Q1/Q3 base plus latest Q3 extensions; not intervention validation or a fresh-environment audit.'}
    report_path=output/'latest_revalidation.json'
    def write():report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    def step(name,*args):
        command=[sys.executable,*map(str,args)];start=time.perf_counter()
        print('Starting '+name,flush=True)
        with (logs/(name+'.log')).open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        report['steps'].append({'name':name,'command':command,'exit_code':result.returncode,'seconds':round(time.perf_counter()-start,3)})
        write()
        if result.returncode:raise RuntimeError('Failed step: '+name+'; see '+str(logs/(name+'.log')))
        print('Completed '+name,flush=True)
    base=output/'modeling';q1=base/'q1';q3=base/'q3';style=output/'style';resolution=output/'resolution';reference=output/'reference'
    try:
        step('tests','-m','unittest','discover','-s','tests','-q')
        if not reuse_base:step('base_training','scripts/complete_modeling.py','--source',source,'--output',base)
        step('base_verification','scripts/verify_continuation.py','--source',source,'--output',base)
        step('style_training','scripts/q3_style_study.py','--source',source,'--q1',q1,'--q3',q3,'--output',style)
        step('style_verification','scripts/verify_q3_style_study.py','--source',source,'--q1',q1,'--q3',q3,'--output',style)
        step('resolution_training','scripts/resolve_q3.py','--source',source,'--q1',q1,'--style-models',style,'--output',resolution)
        step('resolution_verification','scripts/verify_q3_resolution.py','--source',source,'--q1',q1,'--style-models',style,'--output',resolution)
        step('reference_design','scripts/q3_measured_reference.py','--source',source,'--q1',q1,'--style-models',style,'--resolution',resolution,'--output',reference)
        step('reference_verification','scripts/verify_q3_reference.py','--source',source,'--q1',q1,'--style-models',style,'--resolution',resolution,'--output',reference)
        if include_q1_q2:
            step('ordinal_training','scripts/q1_ordinal_experiment.py','--source',source,'--q1',q1,'--output',output/'ordinal')
            step('q2_reliability','scripts/q2_reliability_study.py','--source',source,'--output',output/'q2_reliability')
            step('latest_additions_verification','scripts/verify_latest_additions.py','--source',source,'--root',output)
        if include_diagnostics:
            holdout=output/'reference_holdout';boxes=output/'reference_boxes';robust=output/'reference_robust';probability=output/'grade_probability'
            step('reference_holdout','scripts/q3_reference_holdout.py','--source',source,'--q1',q1,'--output',holdout)
            step('holdout_verification','scripts/verify_q3_reference_holdout.py','--source',source,'--output',holdout)
            shared=['--source',source,'--q1',q1,'--style-models',style,'--resolution',resolution,'--reference',reference]
            step('reference_boxes','scripts/q3_reference_box_study.py',*shared,'--output',boxes)
            step('box_verification','scripts/verify_q3_reference_box.py','--source',source,'--output',boxes)
            step('reference_robust','scripts/q3_reference_robust.py',*shared,'--output',robust)
            step('robust_verification','scripts/verify_q3_reference_robust.py','--source',source,'--output',robust)
            step('diagnostic_figures','scripts/report_q3_reference_diagnostics.py','--holdout',holdout,'--box',boxes,'--output',output/'diagnostic_figures')
            step('grade_probability','scripts/q1_probability_diagnostics.py','--source',source,'--q1',q1,'--ordinal',output/'ordinal','--output',probability)
            step('probability_verification','scripts/verify_q1_probability_diagnostics.py','--source',source,'--q1',q1,'--ordinal',output/'ordinal','--output',probability)
        import pandas as pd
        pairs=[(q1/'model_comparison.csv',project/'outputs/final_20261024/q1/model_comparison.csv'),
               (q1/'q1_scores_primary45.csv',project/'outputs/final_20261024/q1/q1_scores_primary45.csv'),
               (q3/'generated_profiles.csv',project/'outputs/final_20261024/q3/generated_profiles.csv'),
               (q3/'validation_summary.csv',project/'outputs/final_20261024/q3/validation_summary.csv')]
        for folder,old,files in [(style,'q3_style_20261024',['style_metrics.csv','stale_metrics.csv','style_oof.csv','stale_oof.csv']),
                                 (resolution,'q3_resolution_20261024',['transfer_profiles.csv','transfer_results.csv','stale_loss_comparison.csv','defect_checks.csv']),
                                 (reference,'q3_reference_20261024',['transfer_profiles.csv','transfer_results.csv','transfer_sensory.csv','stale_reference_checks.csv'])]:
            pairs.extend((folder/f,project/'outputs'/old/f) for f in files)
        if include_diagnostics:
            for folder,old,files in [(holdout,'q3_reference_holdout_20261024',['metrics.csv','paired_uncertainty.csv']),
                                     (boxes,'q3_reference_box_20261024',['scenario_results.csv']),
                                     (robust,'q3_reference_robust_20261024',['transfer_profiles.csv','transfer_results.csv']),
                                     (probability,'q1_probability_diagnostics_20261024',['metrics.csv'])]:
                pairs.extend((folder/f,project/'outputs'/old/f) for f in files)
        for actual,expected in pairs:
            pd.testing.assert_frame_equal(pd.read_csv(actual),pd.read_csv(expected),check_exact=False,atol=1e-10,rtol=1e-10)
            report['table_comparisons'].append({'actual':str(actual.relative_to(output)),'reference':str(expected.relative_to(project)),'status':'passed'})
        report['status']='passed';report['finished_utc']=datetime.now(timezone.utc).isoformat();write()
        print(f'Latest pipeline verified; {len(pairs)} scientific tables match.',flush=True)
    except Exception as error:
        report['status']='failed';report['error']=str(error);write();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-base',action='store_true',help='Use an explicitly completed base in output/modeling')
    p.add_argument('--include-q1-q2',action='store_true',help='Also rebuild the optional ordinal model and Q2 reliability study')
    p.add_argument('--include-diagnostics',action='store_true',help='Also include the current reference and probability diagnostics; implies --include-q1-q2')
    a=p.parse_args();run(a.source,a.output,a.reuse_base,a.include_q1_q2 or a.include_diagnostics,a.include_diagnostics)
