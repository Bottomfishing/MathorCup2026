"""Finite-pool constraint certificates; never assert feasibility outside that pool."""
from itertools import combinations
import numpy as np


def subset_counts(passes):
    matrix = np.asarray(passes)
    if matrix.ndim != 2 or matrix.dtype != np.dtype(bool):
        raise ValueError('Expected a Boolean candidate-by-group matrix')
    if matrix.shape[1] > 16:
        raise ValueError('Exhaustive subset enumeration is limited to 16 groups')
    counts = {}
    for size in range(matrix.shape[1]+1):
        for subset in combinations(range(matrix.shape[1]),size):
            counts[subset] = int(matrix[:,subset].all(axis=1).sum())
    return counts


def minimal_blockers(passes):
    counts = subset_counts(passes)
    return [subset for subset,count in counts.items() if count == 0 and
            all(counts[subset[:i]+subset[i+1:]] > 0 for i in range(len(subset)))]


def minimax_diagnostic(probability_deltas, objectives, candidate_ids, tolerance=1e-10):
    delta=np.asarray(probability_deltas,float)
    cost=np.asarray(objectives,float);ids=np.asarray(candidate_ids)
    if (delta.ndim!=2 or not delta.shape[0] or not delta.shape[1] or cost.shape!=(len(delta),)
            or ids.shape!=(len(delta),) or not np.isfinite(delta).all() or not np.isfinite(cost).all()
            or len(np.unique(ids))!=len(ids) or not np.isfinite(tolerance) or tolerance<0):
        raise ValueError('Finite nonempty deltas, aligned costs and unique candidate IDs required')
    maximum=np.maximum(delta,0).max(axis=1);violations=(delta>tolerance).sum(axis=1)
    chosen=int(np.lexsort((ids,cost,violations,maximum))[0])
    return chosen,maximum,violations
