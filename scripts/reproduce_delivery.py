"""Rebuild the delivered pipeline from source data in a fresh checkout/environment."""
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


def run(source, output):
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    source, output = source.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / 'logs'; logs.mkdir()
    env = dict(os.environ, PYTHONPATH=str(project / 'src'), PYTHONIOENCODING='utf-8', MPLCONFIGDIR=str(output / 'matplotlib_cache'))
    report = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'python': platform.python_version(), 'system': platform.platform(),
              'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'matplotlib', 'openpyxl']},
              'steps': [], 'scope': 'Current delivery rebuilt. Q2 Ridge bootstrap is retrained; historical forest permutation comparisons are reused as versioned evidence tables.'}
    def step(name, *arguments):
        command = [sys.executable, *map(str, arguments)]
        print('Starting ' + name, flush=True); start = time.perf_counter()
        log = logs / (name + '.log')
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        report['steps'].append({'name': name, 'exit_code': result.returncode, 'seconds': round(time.perf_counter() - start, 3), 'log': str(log.relative_to(output))})
        report['status'] = 'running' if result.returncode == 0 else 'failed'
        (output / 'reproduction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        if result.returncode:
            print(log.read_text(encoding='utf-8')[-5000:])
            raise RuntimeError('Reproduction step failed: ' + name)
        print('Completed ' + name, flush=True)
    audit, modeling = output / 'audit', output / 'modeling'
    q1, q3 = modeling / 'q1', modeling / 'q3'
    step('tests', '-m', 'unittest', 'discover', '-s', 'tests', '-q')
    step('reference', 'scripts/fetch_reference.py')
    step('audit', '-m', 'beer_audit', '--source', source, '--output', audit)
    step('q1_q3_retrain', 'scripts/complete_modeling.py', '--source', source, '--audit', audit, '--output', modeling)
    step('model_verification', 'scripts/verify_continuation.py', '--source', source, '--audit', audit, '--output', modeling)
    step('q1_summary', 'scripts/summarize_q1.py', '--q1', q1, '--output', output / 'q1_delivery')
    step('prediction_interface', 'scripts/predict_q1.py', '--model', q1 / 'model.joblib', '--input', source / '1.csv', '--output', output / 'prediction_interface')
    step('q2_bootstrap_retrain', '-m', 'beer_models.explain', '--source', source, '--audit', audit, '--output', output / 'q2_refit')
    step('q2_evidence', 'scripts/complete_q2.py', '--source', source, '--audit', audit, '--exploratory', output / 'q2_refit', '--output', output / 'q2_delivery')
    step('q3_diagnostics', 'scripts/complete_q3.py', '--source', source, '--audit', audit, '--q1', q1, '--q3', q3, '--output', output / 'q3_delivery')
    step('q3_figures', 'scripts/report_q3_delivery.py', '--delivery', output / 'q3_delivery', '--q3', q3, '--audit', audit)
    # Compare scientific tables rather than machine paths, timings or pickle byte streams.
    import numpy as np
    import pandas as pd
    comparisons = [
        (q1 / 'model_comparison.csv', project / 'outputs/final_20261024/q1/model_comparison.csv'),
        (q1 / 'q1_scores_primary45.csv', project / 'outputs/final_20261024/q1/q1_scores_primary45.csv'),
        (q3 / 'generated_profiles.csv', project / 'outputs/final_20261024/q3/generated_profiles.csv'),
        (q3 / 'validation_summary.csv', project / 'outputs/final_20261024/q3/validation_summary.csv'),
        (q3 / 'transfer_results.csv', project / 'outputs/final_20261024/q3/transfer_results.csv'),
        (output / 'q2_refit/bootstrap_frequency.csv', project / 'outputs/q2_exploratory_20261023_v1/bootstrap_frequency.csv'),
        (output / 'q2_delivery/top3_all50.csv', project / 'outputs/q2_delivery_20261024/top3_all50.csv'),
        (output / 'q3_delivery/transfer_sensitivity_summary.csv', project / 'outputs/q3_delivery_20261024/transfer_sensitivity_summary.csv')]
    report['table_comparisons'] = []
    for actual, expected in comparisons:
        pd.testing.assert_frame_equal(pd.read_csv(actual), pd.read_csv(expected), check_exact=False, atol=1e-10, rtol=1e-10)
        report['table_comparisons'].append({'rebuilt': str(actual.relative_to(output)), 'reference': str(expected.relative_to(project)), 'status': 'passed'})
    report['status'] = 'passed'; report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['rebuilt_model_files'] = [str(p.relative_to(output)) for p in modeling.rglob('*.joblib')]
    (output / 'reproduction.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Reproduction passed; all delivered scientific tables agree within 1e-10.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/reproduction_local'))
    args = p.parse_args(); run(args.source, args.output)
