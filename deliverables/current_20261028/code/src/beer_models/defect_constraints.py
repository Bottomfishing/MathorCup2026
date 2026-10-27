"""Do not require defect compounds to reach a target style's empirical minimum."""
import numpy as np


def project_defects(candidates, source, protected, policy):
    """Preserve or decrease defect coordinates; neither policy may increase them."""
    result = np.asarray(candidates, float).copy()
    source = np.asarray(source, float)
    if not np.isfinite(source[protected]).all():
        raise ValueError('Establish a source bound before projecting missing defects')
    if policy == 'preserve':
        result[:, protected] = source[protected]
    elif policy == 'reduce':
        result[:, protected] = np.minimum(result[:, protected], source[protected])
    else:
        raise ValueError('Unknown defect policy')
    return result


def defect_aware_box(candidates, lower, upper, protected, tolerance=1e-10):
    """Target ranges apply except for defect lower bounds: lower is allowed."""
    lower = np.asarray(lower, float).copy()
    lower[protected] = -np.inf
    candidates = np.asarray(candidates, float)
    return (np.isfinite(candidates).all(axis=1) &
            (candidates >= lower - tolerance).all(axis=1) &
            (candidates <= np.asarray(upper) + tolerance).all(axis=1))


def reference_zero_bound(value, field, calibrated_value):
    """A zero cap for missing, nonnegative diacetyl; never a median-based proof."""
    if np.isfinite(value):
        return float(value), 'observed_coordinate'
    if field == 'X2.3.butanedione' and np.isfinite(calibrated_value) and calibrated_value == 0:
        return 0., 'reference_zero_cap_for_nonnegative_diacetyl'
    raise ValueError('Missing defect requires a supported source bound: ' + field)
