"""Patch the tail (summary/exit) of scripts/q1_official_20261028.py."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / 'scripts' / 'q1_official_20261028.py'
text = PATH.read_text(encoding='utf-8')

START = '    all_pass = bool(checks) and all(bool(value) for value in checks.values())'
END = "\n\nif __name__ == '__main__':"

NEW = '''    all_pass = bool(checks) and all(bool(value) for value in checks.values())
    diagnostics['individual_level'] = {
        'extra_trees': individual_et,
        'q1_topk_nonforced': individual_topk,
        'panel_median_oracle': individual_oracle,
    }
    diagnostics['calibration'] = {'reported': 'nested_linear',
                                  'methods': {item['method']: item for item in improved}}
    diagnostics['metric_calibers'] = calibers.to_dict('records')
    verification = {'all_pass': all_pass, 'checks': checks,
                    'diagnostics': diagnostics,
                    'protocol_id': config['protocol_id']}
    write_json(output_dir / 'verification.json', verification)

    leaky_reference = None
    pre_summary_path = ROOT / 'outputs' / 'q1_decision_20261028_pre_leakfix' / 'summary.json'
    if pre_summary_path.exists():
        pre_summary = json.loads(pre_summary_path.read_text(encoding='utf-8'))
        leaky_reference = {
            name: {'panel_mean_mae': item.get('panel_mean_mae'),
                   'panel_median_mae': item.get('panel_median_mae')}
            for name, item in pre_summary.get('calibration', {}).items()}

    summary = {
        'protocol_id': config['protocol_id'],
        'generated_at': protocol['generated_at'],
        'rows': int(len(submission_long)),
        'grade_distribution': new_metrics['dist'],
        'precap_grade_distribution': precap_metrics['dist'],
        'legacy_grade_distribution': old_metrics['dist'],
        'legacy_integer_median': old_metrics,
        'topk_nonforced': new_metrics,
        'topk_nonforced_precap': precap_metrics,
        'individual_weighted': {'extra_trees_integer_median': individual_et,
                                'q1_topk_nonforced': individual_topk},
        'panel_median_oracle': individual_oracle,
        'calibration': {item['method']: item for item in improved},
        'voided_pre_leakfix_calibration_reference': leaky_reference,
        'leakage_fix': diagnostics['leakage_fix'],
        'parent_child': diagnostics['parent_child'],
        'metric_calibers': calibers.to_dict('records'),
        'non_forced_evidence': diagnostics['non_forced'],
        'verification_all_pass': all_pass,
    }
    write_json(output_dir / 'summary.json', summary)

    print(json.dumps({
        'all_pass': all_pass,
        'checks': checks,
        'dist': new_metrics['dist'],
        'precap_dist': precap_metrics['dist'],
        'parent_child': diagnostics['parent_child'],
        'leakage_fix': diagnostics['leakage_fix'],
        'topk_panel_median_mae': new_metrics['panel_median_mae'],
        'precap_panel_median_mae': precap_metrics['panel_median_mae'],
        'individual_topk': individual_topk,
        'calibration': {item['method']: item['panel_mean_mae'] for item in improved},
        'files': sorted(path.name for path in output_dir.iterdir()),
    }, indent=2, ensure_ascii=False))
    if not all_pass:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit('verification failed: %s' % ', '.join(failed))
    return 0'''

start_index = text.index(START)
end_index = text.index(END)
PATH.write_text(text[:start_index] + NEW + text[end_index:], encoding='utf-8')
print('patched tail', start_index, end_index)
