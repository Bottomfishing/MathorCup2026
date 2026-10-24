"""Small, explicit modeling protocol for the competition deliverables."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from .baseline import make_pipeline


def relative_grades(reference, values):
    """Columnwise training-reference midrank ECDF; never an inverse panel score."""
    reference, values = np.asarray(reference, float), np.asarray(values, float)
    if reference.ndim != 2 or values.ndim != 2 or reference.shape[1] != values.shape[1]:
        raise ValueError('Reference and values must have matching target columns')
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(values).all():
        raise ValueError('Finite, nonempty reference required')
    result = np.empty(values.shape, int)
    for j in range(values.shape[1]):
        ref = np.sort(reference[:, j])
        p = (np.searchsorted(ref, values[:, j], side='left') +
             np.searchsorted(ref, values[:, j], side='right')) / (2 * len(ref))
        result[:, j] = np.minimum(6, np.floor(7 * p)).astype(int)
    return result


def estimator(name, config, seed):
    settings = dict(config, ridge_alphas=[config['ridge_alpha']])
    base = 'extra_trees' if name == 'extra_trees_equal_targets' else name
    model, _ = make_pipeline(base, settings, seed)
    if base == 'ridge_asinh':
        model.set_params(model__alpha=config['ridge_alpha'])
    if name == 'extra_trees_equal_targets':
        model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    return model


def projected_paths(source, anchors, protected_indices, steps=40):
    """Convex paths in supplied coordinates with four defect coordinates fixed."""
    source, anchors = np.asarray(source, float), np.asarray(anchors, float)
    if not np.isfinite(source).all() or not np.isfinite(anchors).all():
        raise ValueError('Project only complete model-space inputs')
    if steps < 1:
        raise ValueError('Positive path resolution required')
    fractions = np.linspace(1 / steps, 1, steps)
    values = source + fractions[None, :, None] * (anchors[:, None, :] - source)
    values[:, :, protected_indices] = source[protected_indices]
    return values.reshape(-1, len(source)), np.tile(fractions, len(anchors)), np.repeat(np.arange(len(anchors)), steps)
