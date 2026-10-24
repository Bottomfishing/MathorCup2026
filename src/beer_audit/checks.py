"""Data validation helpers. AI-assisted: OpenAI Codex (GPT-6), 2026-10-23."""
import numpy as np
import pandas as pd

IDS = ['beer', 'beer_id', 'tasting_category_fine']


def repair_utf8_mojibake(value):
    """Repair only reversible UTF-8-as-Latin1 corruption; preserve other strings."""
    if not isinstance(value, str) or not any(c in value for c in ['Î', 'Ã', 'Â']):
        return value
    try:
        return value.encode('latin1').decode('utf-8')
    except (UnicodeError, ValueError):
        return value


def pairwise_spearman(frame):
    """Re-rank within each complete pair, essential when missingness differs."""
    result = frame.rank().corr()
    incomplete = frame.isna().any()
    for i, left in enumerate(frame):
        for right in frame.columns[i:]:
            if incomplete[left] or incomplete[right]:
                pair = frame[[left, right]].dropna() if left != right else frame[[left]].dropna()
                if left == right:
                    value = 1.0 if pair[left].nunique() > 1 else np.nan
                else:
                    value = pair.rank().corr().iloc[0, 1]
                result.loc[left, right] = result.loc[right, left] = value
    return result


def numeric_frame(frame):
    """Reject malformed numeric cells instead of silently making them missing."""
    out = frame.apply(pd.to_numeric, errors='raise')
    if np.isinf(out.to_numpy(dtype=float)).any():
        raise ValueError('Infinite numeric input')
    return out


def align_samples(reference, other):
    """Join on a verified unique key; IDs remain strings throughout."""
    for frame in (reference, other):
        if frame['beer'].isna().any() or frame['beer'].duplicated().any():
            raise ValueError('Missing or duplicate beer key')
    if set(reference['beer']) != set(other['beer']):
        raise ValueError('Sample key sets differ')
    aligned = other.set_index('beer').loc[reference['beer']].reset_index()
    if not np.array_equal(reference['tasting_category_fine'], aligned['tasting_category_fine']):
        raise ValueError('Style mismatch after sample alignment')
    return aligned


def compare_scale(source, calibrated, atol=2e-6, rtol=1e-6):
    """Empirical comparisons only; zero calibrated values are retained as conflicts.

    Positive-reference comparisons distinguish logs. A numerical match does not
    establish measurement units, detection limits, or the origin of discrepancies.
    """
    a, b = np.asarray(source, float), np.asarray(calibrated, float)
    paired = np.isfinite(a) & np.isfinite(b)
    positive = paired & (b > 0)
    result = {'n_pairs': int(paired.sum()), 'n_positive': int(positive.sum()),
              'n_reference_zero': int((paired & (b == 0)).sum())}
    if positive.sum() < 3 or np.unique(b[positive]).size < 3:
        return result | {'candidate': 'undetermined', 'status': 'insufficient_variation'}
    candidates = {'identity': b[positive], 'log10': np.log10(b[positive]),
                  'ln': np.log(b[positive])}
    matches = {}
    for name, expected in candidates.items():
        matches[name] = int(np.isclose(a[positive], expected, atol=atol, rtol=rtol).sum())
        result[name + '_match_fraction'] = matches[name] / int(positive.sum())
        result[name + '_max_abs_error'] = float(np.max(abs(a[positive] - expected)))
    ordered = sorted(matches, key=matches.get, reverse=True)
    best = ordered[0]
    if matches[best] == matches[ordered[1]]:
        return result | {'candidate': 'undetermined', 'status': 'ambiguous'}
    result['candidate'] = best
    result['status'] = ('positive_pairs_match' if matches[best] == positive.sum()
                        else 'partial_match_conflicts' if matches[best] / positive.sum() >= .8
                        else 'unresolved')
    return result


def require_original_score_calibration(metadata):
    """Stop callers from presenting arbitrary rescaling as original grades."""
    if not metadata.get('original_scale_identified', False):
        raise ValueError('Original 0–6 scale is not identified; no inverse calibration allowed')


def square_matrix_checks(frame):
    values = numeric_frame(frame).to_numpy(float)
    same_axes = list(frame.index) == list(frame.columns)
    finite = np.isfinite(values)
    out = {'rows': values.shape[0], 'columns': values.shape[1],
           'same_ordered_axes': same_axes, 'nonfinite': int((~finite).sum()),
           'out_of_bounds': int((finite & (abs(values) > 1 + 1e-8)).sum())}
    if values.shape[0] == values.shape[1] and same_axes:
        out['symmetry_max_abs_error'] = float(np.nanmax(abs(values - values.T)))
        out['diagonal_max_abs_error'] = float(np.nanmax(abs(np.diag(values) - 1)))
    return out
