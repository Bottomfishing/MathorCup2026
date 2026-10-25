"""Rebuild the selected task2/3/4 delivery; retain versioned historical evidence explicitly."""
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
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from beer_audit.audit import sha256,write_json


def compare_csv(actual, expected):
    import pandas as pd
    pd.testing.assert_frame_equal(pd.read_csv(actual),pd.read_csv(expected),check_exact=False,atol=1e-10,rtol=1e-10)


def run(source, output, reference=None, statement=None):
    project=Path(__file__).resolve().parents[1]
    source,output=source.resolve(),output.resolve()
    reference=reference.resolve() if reference else None
    statement=statement.resolve() if statement else source.parent/'2026年MathorCup大数据挑战赛-赛道A初赛.pdf'
    os.chdir(project);output.mkdir(parents=True,exist_ok=False)
    logs=output/'logs';logs.mkdir()
    env=dict(os.environ,PYTHONPATH=str(project/'src'),PYTHONIOENCODING='utf-8',MPLCONFIGDIR=str(output/'matplotlib_cache'))
    report={'status':'running','started_utc':datetime.now(timezone.utc).isoformat(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'working_tree_dirty':bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip()),
        'python':platform.python_version(),'executable':sys.executable,'system':platform.platform(),
        'source':str(source),'steps':[],'table_comparisons':[],
        'rebuilt':['Q1 continuous fold/full models','Q1 ordinal fold/full models and task2 bundle',
            'Q3 generator, style classifiers, focused Stale and resolved paths','task2/3/4 primary tables'],
        'retained_evidence':['versioned audit definitions and task1 scope (independently rechecked)',
            'Q2 200-refit selections and 1999-permutation nulls, forest comparisons and original annotations',
            'Q3 separate measured-reference, 1% IQR and reference-holdout diagnostics'],
        'interpretation':'Current selected workflow rebuild, not a retraining of every historical experiment or validation of raw intensity, causal effects or brewing outcomes.'}
    report_path=output/'reproduction.json'
    def write():write_json(report_path,report)
    def step(name,*arguments):
        command=[sys.executable,*map(str,arguments)];start=time.perf_counter();log=logs/(name+'.log')
        print('Starting '+name,flush=True)
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        report['steps'].append({'name':name,'command':command,'exit_code':result.returncode,
            'seconds':round(time.perf_counter()-start,3),'log':str(log.relative_to(output))})
        write()
        if result.returncode:
            print(log.read_text(encoding='utf-8')[-4000:],flush=True)
            raise RuntimeError('Failed step: '+name+'; see '+str(log))
        print('Completed '+name,flush=True)
    def compare(actual,expected):
        compare_csv(actual,expected)
        report['table_comparisons'].append({'rebuilt':str(actual.relative_to(output)),
            'reference':str(expected.relative_to(project)),'status':'passed','rebuilt_sha256':sha256(actual),'reference_sha256':sha256(expected)})
    write()
    try:
        if sys.version_info[:2]!=(3,12):raise RuntimeError('Use Python 3.12 and requirements-lock.txt')
        report['packages']={}
        for line in (project/'requirements-lock.txt').read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.startswith('#'):continue
            name,expected=line.strip().split('==');actual=importlib.metadata.version(name)
            report['packages'][name]=actual
            if actual!=expected:raise RuntimeError(f'{name}: installed {actual}; lock requires {expected}')
        report['code_sha256']={str(p.relative_to(project)):sha256(p) for folder in ['src','scripts','configs']
            for p in (project/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.json']}
        write();step('dependency_check','-m','pip','check');step('tests','-m','unittest','discover','-s','tests','-q')
        if reference is None:
            step('public_reference','scripts/fetch_reference.py')
            reference=project/'data/reference/zenodo10653704/Supplemental Files and Figure source files.xlsx'
        manifest=json.loads((project/'data/reference/zenodo10653704/download_manifest.json').read_text(encoding='utf-8'))
        expected=next(r['sha256'] for r in manifest if r['file']=='Supplemental Files and Figure source files.xlsx')
        if sha256(reference)!=expected:raise ValueError('Reference workbook differs from pinned source')
        report['reference']={'path':str(reference),'sha256':expected};write()
        audit=output/'audit';scope=output/'scope';base=output/'modeling';q1=base/'q1';q3=base/'q3'
        ordinal=output/'ordinal';scorecard=output/'q1';style=output/'style';resolution=output/'paths';diagnostics=output/'generation_diagnostics'
        step('source_audit','-m','beer_audit','--source',source,'--output',audit)
        step('scope_audit','scripts/audit_task1_scope.py','--source',source,'--statement',statement,'--output',scope)
        for name in ['feature_dictionary.csv','sensory_mapping.csv','correlation_method_comparison.csv']:
            compare(audit/name,project/'outputs/p0_audit_20261023_final'/name)
        for name in ['scorecard_scope.csv','defect_scope.csv']:
            compare(scope/name,project/'outputs/task1_acceptance_20261025'/name)
        step('base_training','scripts/complete_modeling.py','--source',source,'--audit',audit,'--output',base)
        step('base_verification','scripts/verify_continuation.py','--source',source,'--audit',audit,'--output',base)
        step('ordinal_training','scripts/q1_ordinal_experiment.py','--source',source,'--q1',q1,'--output',ordinal)
        step('q1_assembly','scripts/complete_q1_task2.py','--source',source,'--q1',q1,'--ordinal',ordinal,'--output',scorecard)
        step('prediction_interface','scripts/predict_q1.py','--model',scorecard/'model.joblib','--input',source/'1.csv','--output',output/'prediction_interface')
        step('q1_verification','scripts/verify_q1_task2.py','--source',source,'--q1',q1,'--ordinal',ordinal,
            '--output',scorecard,'--smoke',output/'prediction_interface')
        step('style_training','scripts/q3_style_study.py','--source',source,'--q1',q1,'--q3',q3,'--output',style)
        step('style_verification','scripts/verify_q3_style_study.py','--source',source,'--q1',q1,'--q3',q3,'--output',style)
        step('path_design','scripts/resolve_q3.py','--source',source,'--q1',q1,'--style-models',style,'--output',resolution)
        step('path_verification','scripts/verify_q3_resolution.py','--source',source,'--q1',q1,'--style-models',style,'--output',resolution)
        step('generation_diagnostics','scripts/complete_q3.py','--source',source,'--audit',audit,'--q1',q1,'--q3',q3,'--output',diagnostics)
        shared_q3=['--source',source,'--q1',q1,'--scorecard',scorecard,'--q3',q3,'--style-models',style,
            '--resolution',resolution,'--delivery',diagnostics,'--output',output/'q3']
        step('q3_assembly','scripts/complete_q3_task3.py',*shared_q3)
        step('q3_verification','scripts/verify_q3_task3.py',*shared_q3)
        shared_q2=['--source',source,'--reference',reference,'--audit',audit,'--scorecard',scorecard,'--output',output/'q2']
        step('q2_assembly','scripts/complete_q2_task4.py',*shared_q2)
        step('q2_verification','scripts/verify_q2_task4.py',*shared_q2)
        for rebuilt,frozen in [(scorecard,project/'outputs/q1_task2_20261025'),
            (output/'q3',project/'outputs/q3_task3_20261025'),(output/'q2',project/'outputs/q2_task4_20261025')]:
            filenames={p.name for p in frozen.glob('*.csv')}
            if filenames!={p.name for p in rebuilt.glob('*.csv')}:raise ValueError('Primary CSV set differs: '+str(rebuilt))
            for name in sorted(filenames):compare(rebuilt/name,frozen/name)
        import networkx as nx
        rebuilt_graph=nx.read_graphml(output/'q2/network_0_6.graphml')
        frozen_graph=nx.read_graphml(project/'outputs/q2_task4_20261025/network_0_6.graphml')
        if dict(rebuilt_graph.nodes(data=True))!=dict(frozen_graph.nodes(data=True)):raise ValueError('Graph nodes differ')
        if dict(((a,b),d) for a,b,d in rebuilt_graph.edges(data=True))!=dict(((a,b),d) for a,b,d in frozen_graph.edges(data=True)):
            raise ValueError('Graph edges differ')
        report['graph_comparison']='passed'
        report['checks']={question:json.loads((output/question/'verification.json').read_text(encoding='utf-8')) for question in ['q1','q2','q3']}
        if any(r['status']!='passed' for r in report['checks'].values()):raise ValueError('A final question verification did not pass')
        report['scientific_limitations']={'q1_original_card_intensity_recovered':False,'q2_unidentified_rank_slots':6,
            'q2_causal_effects_established':False,'q3_actual_after_stale_validated':False}
        report['rebuilt_model_files']=[str(p.relative_to(output)) for p in output.rglob('*.joblib')]
        report['status']='passed';report['finished_utc']=datetime.now(timezone.utc).isoformat();write()
        print(f'Current delivery passed: {len(report["steps"])} steps, {len(report["table_comparisons"])} CSV comparisons and signed graph.',flush=True)
    except BaseException as error:
        report['status']='failed';report['error']=f'{type(error).__name__}: {error}'
        report['finished_utc']=datetime.now(timezone.utc).isoformat();write();raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/current_delivery_local'))
    p.add_argument('--reference',type=Path,help='Existing pinned public workbook; otherwise fetch/verify the pinned copy')
    p.add_argument('--statement',type=Path,help='Original statement PDF; defaults to the parent of the attachment directory')
    a=p.parse_args();run(a.source,a.output,a.reference,a.statement)
