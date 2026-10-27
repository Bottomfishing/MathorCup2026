"""Small auditable summaries for fixed-candidate resampling diagnostics."""
import numpy as np


def change_summary(changes, tolerance=1e-10):
    """Rows are refits, columns are fixed candidates; quantiles are not effect CIs."""
    x = np.asarray(changes, float)
    if x.ndim != 2 or not x.shape[0] or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Finite nonempty refit-by-candidate matrix required')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Finite nonnegative tolerance required')
    return dict(pass_fraction=np.mean(x <= tolerance, axis=0),
                median=np.median(x, axis=0), lower=np.quantile(x, .025, axis=0),
                upper=np.quantile(x, .975, axis=0))


def set_overlap(left, right):
    a, b = set(left), set(right)
    if not a or not b:
        raise ValueError('Nonempty selections required')
    return len(a & b) / len(a | b)


def pareto_mask(costs):
    """Both columns are minimized; retain identical nondominated points."""
    x = np.asarray(costs, float)
    if x.ndim != 2 or x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError('Finite two-objective matrix required')
    # Sort by first objective; equal first-coordinate groups need their minimum
    # second coordinate handled together so duplicate best points are retained.
    result = np.zeros(len(x), bool)
    previous = np.inf
    for value in np.unique(x[:, 0]):
        indices = np.flatnonzero(x[:, 0] == value)
        best = x[indices, 1].min()
        result[indices] = (x[indices, 1] == best) & (best < previous)
        previous = min(previous, best)
    return result
