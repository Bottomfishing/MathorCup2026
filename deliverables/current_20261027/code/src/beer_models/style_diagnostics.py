"""Descriptive style distances and finite-grid transfer selection."""
import numpy as np
from scipy.spatial.distance import cdist


def style_centers(generator):
    return {s: generator.robust_.transform(generator.x_[idx]).mean(axis=0)
            for s, idx in generator.indices_.items()}


def nearest_centroid(values, centers):
    styles = sorted(centers)
    distances = np.stack([np.mean((values - centers[s]) ** 2, axis=1) for s in styles])
    return np.asarray(styles)[np.argmin(distances, axis=0)]


def nearest_neighbor_radius(values, quantile=.95):
    if len(values) < 2:
        raise ValueError('At least two reference samples required')
    distance = cdist(values, values) / np.sqrt(values.shape[1])
    np.fill_diagonal(distance, np.inf)
    return float(np.quantile(distance.min(axis=1), quantile))


def select_transfer(distances, changes, feasible, target_like, weight=.7):
    if not 0 <= weight <= 1:
        raise ValueError('Weight must be between zero and one')
    eligible = np.asarray(feasible, bool) & np.asarray(target_like, bool)
    if not eligible.any():
        eligible = np.asarray(feasible, bool)
    pool = np.flatnonzero(eligible)
    if not len(pool):
        return None
    return int(pool[np.argmin(weight * np.asarray(distances)[pool] + (1 - weight) * np.asarray(changes)[pool])])
