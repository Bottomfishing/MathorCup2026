"""Patch the verification/hand-off section of scripts/q1_official_20261028.py."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / 'scripts' / 'q1_official_20261028.py'
text = PATH.read_text(encoding='utf-8')

START = '    # hand-off cross-checks (read-only)'
END = '    # full-fit table: read only, must not enter the OOF decision table'

NEW = '''    # hand-off cross-checks (read-only): only the legacy columns are expected to be
    # identical; the new decision column is documented as diverging from the pre-fix
    # hand-off by exactly the parent/child cap.
    handoff = {}
    handoff_tolerance = float(config['verification']['handoff_tolerance'])
    evaluation_handoff = handoff_dir / 'q1__evaluation_long_12500.csv'
    if evaluation_handoff.exists():
        right = pd.read_csv(evaluation_handoff).sort_values(
            ['beer', 'target']).reset_index(drop=True)
        left = evaluation.sort_values(['beer', 'target']).reset_index(drop=True)
        aligned = (left['beer'].tolist() == right['beer'].tolist()
                   and left['target'].tolist() == right['target'].tolist())
        if not aligned:
            checks['q1_handoff_legacy_columns_identical'] = False
            handoff['alignment'] = 'row keys differ'
        else:
            legacy_columns = ['grade_integer_median_old', 'panel_mean_reference',
                              'panel_median_reference', 'probability_rating_ge1',
                              'probability_rating_ge5', 'probability_rating_6']
            diffs = {column: float(np.abs(left[column].to_numpy(float)
                                          - right[column].to_numpy(float)).max())
                     for column in legacy_columns}
            fold_equal = bool((left['outer_fold'].to_numpy(int)
                               == right['outer_fold'].to_numpy(int)).all())
            handoff['legacy_column_max_abs_diff'] = diffs
            handoff['fold_equal'] = fold_equal
            handoff['new_grade_changed_cells_vs_handoff'] = int(
                (left['grade_0_6'].to_numpy(float)
                 != right['grade_0_6'].to_numpy(float)).sum())
            handoff['new_grade_max_abs_diff_vs_handoff'] = float(
                np.abs(left['grade_0_6'].to_numpy(float)
                       - right['grade_0_6'].to_numpy(float)).max())
            checks['q1_handoff_legacy_columns_identical'] = bool(
                max(diffs.values()) < handoff_tolerance and fold_equal)
    else:
        checks['q1_handoff_legacy_columns_identical'] = True
        handoff['evaluation'] = 'handoff table absent; cross-check skipped'

    # The calibrated panel mean is a leak-free recomputation: it must be recorded but
    # is allowed (and expected) to differ from the voided hand-off numbers.
    calibrated_handoff = handoff_dir / 'q1__calibrated_panel_mean_12500.csv'
    if calibrated_handoff.exists():
        right = pd.read_csv(calibrated_handoff).sort_values(
            ['beer', 'target']).reset_index(drop=True)
        left = calibrated.sort_values(['beer', 'target']).reset_index(drop=True)
        handoff['calibrated_mean_max_abs_diff_vs_voided_handoff'] = float(
            np.abs(left['panel_mean_calibrated'].to_numpy(float)
                   - right['panel_mean_calibrated'].to_numpy(float)).max())
        handoff['calibrated_round_equal_vs_voided_handoff'] = bool(
            (left['grade_from_calibrated_mean_round'].to_numpy(int)
             == right['grade_from_calibrated_mean_round'].to_numpy(int)).all())
    checks['q1_no_voided_calibration_rows_in_metrics'] = bool(
        not any('fold-wise linear calibration' in str(method)
                or 'fold-wise isotonic calibration' in str(method)
                for method in metrics['method']))
    diagnostics['handoff'] = handoff

'''

start_index = text.index(START)
end_index = text.index(END)
PATH.write_text(text[:start_index] + NEW + text[end_index:], encoding='utf-8')
print('patched', start_index, end_index)
