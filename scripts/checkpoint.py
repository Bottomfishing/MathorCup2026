"""Local source snapshot and artifact integrity inventory. AI-assisted: Codex, 2026-10-24."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

ACTIVE=[
 'outputs/p0_audit_20261023_final','outputs/p1_baseline_20261023_v1','outputs/p1_robust_20261023_v2',
 'outputs/p1_review_20261023','outputs/q2_exploratory_20261023_v1','outputs/q2_permutation_20261024_v2',
 'outputs/q2_within_style_20261024_v3','outputs/q2_review_20261024_v3','outputs/q3_exploratory_20261024_v2',
 'outputs/q3_review_20261024','outputs/q3_threshold_scenarios_20261024_v2','outputs/q1_relative_grades_20261024_v1',
 'outputs/reproduction_latest_20261024','outputs/q3_reference_holdout_20261024','outputs/q3_reference_box_20261024',
 'outputs/q3_reference_robust_20261024','outputs/q3_reference_diagnostics_20261024_v2','outputs/q1_probability_diagnostics_20261024',
 'outputs/q3_guard_witness_search_20261024']


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def create(project,out):
    out.mkdir(parents=True,exist_ok=False)
    files=set()
    for name in ['src','scripts','tests','configs','docs','.github']:
        for p in (project/name).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:files.add(p)
    for pattern in ['README.md','requirements*.txt','*.ps1']:
        files.update(project.glob(pattern))
    source_records=[]
    with zipfile.ZipFile(out/'source_bundle.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            relative=p.relative_to(project).as_posix();archive.write(p,relative)
            source_records.append({'path':relative,'sha256':digest(p),'bytes':p.stat().st_size})
    artifacts=[]
    for directory in ACTIVE:
        folder=project/directory
        if not folder.is_dir():raise FileNotFoundError(folder)
        for p in sorted(folder.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                artifacts.append({'path':p.relative_to(project).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size})
    record={'created_utc':datetime.now(timezone.utc).isoformat(),'project':str(project),'source_zip_sha256':digest(out/'source_bundle.zip'),
            'source_files':source_records,'active_output_directories':ACTIVE,'artifact_files':artifacts,
            'scope':'Source archive plus local-output integrity inventory, not an off-device backup or final contest submission.'}
    (out/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    # Verify archived content, not just ZIP container readability.
    with zipfile.ZipFile(out/'source_bundle.zip') as archive:
        for item in source_records:assert hashlib.sha256(archive.read(item['path'])).hexdigest()==item['sha256']
        assert archive.testzip() is None
    print(json.dumps({'source_files':len(files),'artifact_files':len(artifacts),
                      'artifact_bytes':sum(r['bytes'] for r in artifacts),'source_zip_verified':True},indent=2))


def verify(out):
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'));project=Path(manifest['project'])
    assert digest(out/'source_bundle.zip')==manifest['source_zip_sha256']
    for item in manifest['source_files']+manifest['artifact_files']:
        path=project/item['path']
        if digest(path)!=item['sha256']:raise ValueError('File changed since checkpoint: '+str(path))
    print('Checkpoint source and active output hashes match.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--verify',action='store_true')
    a=p.parse_args()
    if a.verify:verify(a.output)
    else:create(Path(__file__).resolve().parents[1],a.output)
