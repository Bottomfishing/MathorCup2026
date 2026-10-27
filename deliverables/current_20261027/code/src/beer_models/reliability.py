"""Randomization and multiple-target evidence calculations."""
import numpy as np


def permute_within_groups(groups, rng):
    groups = np.asarray(groups)
    indices = np.arange(len(groups))
    for group in sorted(set(groups)):
        positions = np.flatnonzero(groups == group)
        indices[positions] = rng.permutation(positions)
    return indices


def monte_carlo_upper_p(observed, null_values):
    observed, null_values = np.asarray(observed), np.asarray(null_values)
    return (1 + (null_values >= observed).sum(axis=0)) / (len(null_values) + 1)


def benjamini_yekutieli(pvalues):
    pvalues = np.asarray(pvalues, float)
    if pvalues.ndim != 1 or not np.isfinite(pvalues).all() or ((pvalues < 0) | (pvalues > 1)).any():
        raise ValueError('Finite p-values in [0,1] required')
    n = len(pvalues)
    if n == 0:
        return pvalues.copy()
    order = np.argsort(pvalues, kind='stable')
    scaled = pvalues[order] * n * np.sum(1 / np.arange(1, n + 1)) / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(scaled[::-1])[::-1]
    result = np.empty(n); result[order] = np.minimum(1, adjusted)
    return result
