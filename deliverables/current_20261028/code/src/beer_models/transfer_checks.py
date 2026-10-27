"""Interpret coordinate and fitted-proxy checks without claiming measured outcomes."""
import numpy as np


def coordinate_nonincrease(after, cap, tolerance=1e-10):
    """Missing caps are unknown; a fitted median is not a measurement guarantee."""
    if not np.isfinite(after) or not np.isfinite(cap):
        return None
    return bool(after <= cap + tolerance)


def transfer_status(checks, conditional_zero_cap=False):
    if any(value is None for value in checks):
        return 'undetermined_missing_constraint_evidence'
    if not all(checks):
        return 'rejected_by_unified_model_checks'
    if conditional_zero_cap:
        return 'model_candidate_with_reference_zero_assumption'
    return 'model_candidate_under_proxy_constraints'
