"""Deterministic Q1 decision layer: non-forced top-k grading and fold-wise calibration.

This module is pure: no file IO, no absolute paths, no dependency on the rest of
``beer_models``.  All functions take/return NumPy arrays or plain Python objects,
validate shapes and domains, and are deterministic for a fixed input order.

The Q1 delivery is a 250 beer x 50 target x 0..6 grade table.  The two official
inference artifacts only expose P(>=1), P(>=5) and P(=6) per cell, so the decision
layer must not pretend to know P(=2..4).  It keeps the integer conditional median
(``base_grades``, capped at 4) as the middle of the scale and only promotes a small
number of cells to 5 or 6, using a **non-forced** cumulative-probability top-k rule:
per (target, outer fold) group the number of promoted cells is derived from the
observed high-grade rate in the *other* folds, so no target is forced to output a
5/6 when the training folds contain none.

Definitions used by the official script (protocol ``q1_decision_20261028``):

* ``r5`` / ``r6`` -- fraction of training-fold cells whose panel-weighted median is
  ``>=5`` / ``==6`` for the same target.
* ``n`` -- number of cells in the group (one outer fold of one target).
* ``k5 = round_half_even(r5 * n)``, ``k6 = round_half_even(r6 * n)``; ``k=0`` means
  the target emits no 5 (or no 6) in that fold.
* promote the top ``k6`` cells by ``p6`` to grade 6, then the top ``k5`` of the
  remaining cells by ``p_ge5`` to grade 5; every other cell keeps
  ``min(base_grade, 4)``.
* ties in a probability are broken by ascending row position inside the group, which
  makes the result independent of the underlying sort implementation.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    'fold_training_rate',
    'fold_training_mean',
    'fold_style_mean',
    'topk_decision',
    'calibrate_panel_mean',
    'isotonic_calibrate',
    'quantile_calibrate',
    'linear_to_grade',
    'high_grade_stats',
    'enforce_parent_child',
    'fit_nested_linear',
    'apply_nested_linear',
    'fit_nested_isotonic',
    'apply_nested_isotonic',
    'fit_nested_quantile',
    'apply_nested_quantile',
    'observed_high_grade_rates',
    'runtime_decision',
]

_GRADE_MIN = 0
_GRADE_MAX = 6


def _as_1d_float(values, name):
    """Return a finite float64 1-D copy of ``values``."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError('%s must be one-dimensional, got shape %r' % (name, array.shape))
    if array.size == 0:
        raise ValueError('%s must not be empty' % name)
    if not np.isfinite(array).all():
        raise ValueError('%s must contain only finite values' % name)
    return array


def _as_1d_int(values, name):
    """Return an int64 1-D array, rejecting non-integral or non-finite input."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError('%s must be one-dimensional, got shape %r' % (name, array.shape))
    if array.size == 0:
        raise ValueError('%s must not be empty' % name)
    numeric = np.asarray(array, dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError('%s must contain only finite values' % name)
    rounded = np.rint(numeric)
    if not np.allclose(numeric, rounded, rtol=0.0, atol=1e-9):
        raise ValueError('%s must be integral' % name)
    return rounded.astype(np.int64)


def _validate_keys(groups, folds, n):
    """Validate aligned group/fold labels and return them as object arrays."""
    group_array = np.asarray(groups, dtype=object)
    fold_array = np.asarray(folds, dtype=object)
    if group_array.ndim != 1 or fold_array.ndim != 1:
        raise ValueError('group_ids and fold_ids must be one-dimensional')
    if group_array.size != n or fold_array.size != n:
        raise ValueError('group_ids/fold_ids must align with the value arrays')
    return group_array, fold_array


def _group_fold_slices(group_array, fold_array):
    """Yield ``(group, fold, train_mask, test_mask)`` for every observed pair."""
    for group in _stable_unique(group_array):
        in_group = group_array == group
        folds = _stable_unique(fold_array[in_group])
        for fold in folds:
            test = in_group & (fold_array == fold)
            train = in_group & (fold_array != fold)
            yield group, fold, train, test


def _stable_unique(values):
    """Unique values in first-appearance order (deterministic, hash independent)."""
    seen = {}
    for value in values:
        seen.setdefault(value, None)
    return list(seen)


def fold_training_rate(labels, group_ids, fold_ids, threshold):
    """Per-cell rate of ``labels >= threshold`` in the training folds of its group.

    Parameters
    ----------
    labels : array-like of float, shape (n,)
        Reference grades (usually the panel-weighted median).
    group_ids, fold_ids : array-like, shape (n,)
        Target label and outer-fold label of every cell.
    threshold : numeric
        Grade threshold defining the event.

    Returns
    -------
    numpy.ndarray of float, shape (n,)
        Each cell receives the rate computed on all cells of the same group that sit
        in a *different* outer fold.  Fails when a group/fold pair has no training cell.
    """
    values = _as_1d_float(labels, 'labels')
    group_array, fold_array = _validate_keys(group_ids, fold_ids, values.size)
    event = (values >= float(threshold)).astype(float)
    rates = np.full(values.size, np.nan)
    for _group, _fold, train, test in _group_fold_slices(group_array, fold_array):
        if not train.any():
            raise ValueError('group %r fold %r has no training cells' % (_group, _fold))
        rates[test] = event[train].mean()
    if not np.isfinite(rates).all():
        raise ValueError('some cells received no training rate')
    return rates


def fold_training_mean(values, group_ids, fold_ids):
    """Per-cell mean of ``values`` over the other folds of the same group."""
    target = _as_1d_float(values, 'values')
    group_array, fold_array = _validate_keys(group_ids, fold_ids, target.size)
    means = np.full(target.size, np.nan)
    for _group, _fold, train, test in _group_fold_slices(group_array, fold_array):
        if not train.any():
            raise ValueError('group %r fold %r has no training cells' % (_group, _fold))
        means[test] = target[train].mean()
    if not np.isfinite(means).all():
        raise ValueError('some cells received no training mean')
    return means


def fold_style_mean(values, style_ids, group_ids, fold_ids):
    """Per-cell fold-wise style mean of ``values`` with a global-mean fallback.

    For each (target, fold) the mean of ``values`` is computed per style on the other
    folds; a style unseen in the training folds falls back to the training global mean.
    """
    target = _as_1d_float(values, 'values')
    styles = np.asarray(style_ids, dtype=object)
    if styles.ndim != 1 or styles.size != target.size:
        raise ValueError('style_ids must be one-dimensional and aligned with values')
    group_array, fold_array = _validate_keys(group_ids, fold_ids, target.size)
    means = np.full(target.size, np.nan)
    for _group, _fold, train, test in _group_fold_slices(group_array, fold_array):
        if not train.any():
            raise ValueError('group %r fold %r has no training cells' % (_group, _fold))
        train_styles = styles[train]
        train_values = target[train]
        global_mean = float(train_values.mean())
        style_mean = {}
        for style in _stable_unique(train_styles):
            style_mean[style] = float(train_values[train_styles == style].mean())
        for row in np.flatnonzero(test):
            means[row] = style_mean.get(styles[row], global_mean)
    if not np.isfinite(means).all():
        raise ValueError('some cells received no style mean')
    return means


def _rank_order(probabilities, positions):
    """Descending probability, ties by ascending position (deterministic)."""
    return np.lexsort((positions, -probabilities))


def topk_decision(p_ge5, p6, base_grades, rate_ge5, rate_6, group_ids):
    """Non-forced cumulative-probability top-k decision over 0..6 grades.

    Parameters
    ----------
    p_ge5, p6 : array-like of float, shape (n,)
        Calibrated probabilities P(grade >= 5) and P(grade == 6) per cell.
    base_grades : array-like of float, shape (n,)
        Integer conditional-median estimate; values above 4 are capped at 4.
    rate_ge5, rate_6 : array-like of float, shape (n,)
        Per-cell training-fold observed rates (see :func:`fold_training_rate`).
    group_ids : array-like, shape (n,)
        Decision group (one outer fold of one target).  ``k5``/``k6`` are constant
        inside a group and equal to ``round(rate * group_size)``.

    Returns
    -------
    numpy.ndarray of int64, shape (n,)
        Grades in ``[0, 6]``.

    Notes
    -----
    * ``k = 0`` suppresses 5 (or 6) for that group; the rule is *not forced*.
    * Grade 6 is assigned first; grade 5 is chosen only among the remaining cells.
    * Sorting is descending probability with ties broken by ascending row position.
    * ``base_grades > 4`` are capped to 4 so the middle of the scale never leaks
      into the tail decision.
    """
    prob5 = _as_1d_float(p_ge5, 'p_ge5')
    prob6 = _as_1d_float(p6, 'p6')
    base = _as_1d_float(base_grades, 'base_grades')
    rates5 = _as_1d_float(rate_ge5, 'rate_ge5')
    rates6 = _as_1d_float(rate_6, 'rate_6')
    size = prob5.size
    for name, array in (('p6', prob6), ('base_grades', base), ('rate_ge5', rates5), ('rate_6', rates6)):
        if array.size != size:
            raise ValueError('%s must align with p_ge5' % name)
    for name, array in (('p_ge5', prob5), ('p6', prob6)):
        if ((array < 0.0) | (array > 1.0)).any():
            raise ValueError('%s must lie in [0, 1]' % name)
    for name, array in (('rate_ge5', rates5), ('rate_6', rates6)):
        if ((array < 0.0) | (array > 1.0)).any():
            raise ValueError('%s must lie in [0, 1]' % name)
    if ((base < _GRADE_MIN) | (base > _GRADE_MAX)).any():
        raise ValueError('base_grades must lie in [0, 6]')
    group_array = np.asarray(group_ids, dtype=object)
    if group_array.ndim != 1 or group_array.size != size:
        raise ValueError('group_ids must be one-dimensional and aligned with the probabilities')

    grades = np.minimum(base, 4).round().astype(np.int64)
    for group, _positions in _group_index(group_array):
        idx = _positions
        n = idx.size
        k6 = int(round(float(rates6[idx][0]) * n))
        k5 = int(round(float(rates5[idx][0]) * n))
        k5 = min(max(k5, 0), n)
        k6 = min(max(k6, 0), n)
        order6 = idx[_rank_order(prob6[idx], np.arange(n))]
        chosen6 = order6[:k6]
        grades[chosen6] = 6
        remaining = np.setdiff1d(idx, chosen6, assume_unique=True)
        if remaining.size and k5:
            order5 = remaining[_rank_order(prob5[remaining], np.arange(remaining.size))]
            chosen5 = order5[:min(k5, remaining.size)]
            grades[chosen5] = 5
    return grades


def _group_index(group_array):
    """Yield ``(group, positions)`` in first-appearance order."""
    for group in _stable_unique(group_array):
        yield group, np.flatnonzero(group_array == group)


def _least_squares_fit(design, response):
    """Return ``(intercept, slope)`` with a constant fallback for degenerate design.

    The minimum-norm least-squares solution is used when the design is rank
    deficient for any reason other than "no slope information at all" (e.g. a
    feature that is constant over the training cells).  This mirrors the
    reference implementation (``numpy.linalg.lstsq`` / scikit-learn
    ``LinearRegression``), which also returns the minimum-norm solution, so
    predictions stay comparable with the frozen handoff tables.  Only a rank-1
    design, i.e. every feature column being redundant with the intercept, falls
    back to the training-fold mean.
    """
    intercept_column = np.ones((design.shape[0], 1))
    matrix = np.hstack([intercept_column, design])
    solution, _residuals, rank, _singular = np.linalg.lstsq(matrix, response, rcond=None)
    if rank < 2:
        # No slope information at all: an intercept-only (mean) fit is the only
        # defensible choice.
        return float(response.mean()), np.zeros(design.shape[1])
    return float(solution[0]), solution[1:].astype(float)


def calibrate_panel_mean(features, panel_mean_ref, group_ids, fold_ids):
    """Fold-wise per-target linear calibration of scores to the panel mean.

    Parameters
    ----------
    features : array-like of float, shape (n,) or (n, k)
        Raw score(s) to be mapped onto the panel-mean scale (e.g. the integer median,
        or the integer median plus a fold-wise style mean).
    panel_mean_ref : array-like of float, shape (n,)
        Panel-weighted mean reference grade.
    group_ids, fold_ids : array-like, shape (n,)
        Target label and outer-fold label of every cell.

    Returns
    -------
    predictions : numpy.ndarray of float, shape (n,)
        Out-of-fold calibrated panel-mean predictions.
    coefficients : dict
        ``groups``, ``folds``, ``intercept``, ``slope`` (shape ``(m, k)``) and
        ``n_train`` for the ``m`` fitted (target, fold) pairs.
    """
    design = np.asarray(features, dtype=float)
    target = _as_1d_float(panel_mean_ref, 'panel_mean_ref')
    if design.ndim == 1:
        design = design[:, None]
    if design.ndim != 2:
        raise ValueError('features must be one- or two-dimensional')
    if design.shape[0] != target.size:
        raise ValueError('features and panel_mean_ref must align')
    if not np.isfinite(design).all():
        raise ValueError('features must contain only finite values')
    group_array, fold_array = _validate_keys(group_ids, fold_ids, target.size)

    predictions = np.full(target.size, np.nan)
    rows = []
    for group, fold, train, test in _group_fold_slices(group_array, fold_array):
        if not train.any():
            raise ValueError('group %r fold %r has no training cells' % (group, fold))
        intercept, slope = _least_squares_fit(design[train], target[train])
        predictions[test] = intercept + design[test] @ slope
        rows.append((group, fold, intercept, slope, int(train.sum())))
    if not np.isfinite(predictions).all():
        raise ValueError('some cells received no calibrated prediction')
    coefficients = {
        'groups': [row[0] for row in rows],
        'folds': [row[1] for row in rows],
        'intercept': np.array([row[2] for row in rows], dtype=float),
        'slope': np.array([row[3] for row in rows], dtype=float),
        'n_train': np.array([row[4] for row in rows], dtype=np.int64),
    }
    return predictions, coefficients


def _pava_isotonic(x, y):
    """Weighted PAVA isotonic (non-decreasing) fit of ``y`` on ``x``.

    Equal ``x`` values are folded into one weighted observation first.  Returns the
    unique ``x`` grid and the fitted non-decreasing values on it.
    """
    order = np.argsort(x, kind='stable')
    x_sorted = x[order]
    y_sorted = y[order]
    unique_x, inverse, counts = np.unique(x_sorted, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=y_sorted, minlength=unique_x.size)
    values = sums / counts

    # Each stack entry is [first unique-x index, number of unique-x cells,
    # total weight, weighted sum]; adjacent entries are pooled on violation.
    stack = []  # type: list[list[float]]
    for index in range(unique_x.size):
        stack.append([index, 1, float(counts[index]), float(values[index]) * float(counts[index])])
        while len(stack) > 1 and (stack[-2][3] / stack[-2][2]) > (stack[-1][3] / stack[-1][2]):
            last = stack.pop()
            stack[-1][1] += last[1]
            stack[-1][2] += last[2]
            stack[-1][3] += last[3]
    fitted_unique = np.empty(unique_x.size, dtype=float)
    for first, width, weight, weighted_sum in stack:
        fitted_unique[int(first):int(first) + int(width)] = weighted_sum / weight
    return unique_x, fitted_unique


def isotonic_calibrate(scores, panel_mean_ref, group_ids, fold_ids):
    """Fold-wise per-target isotonic calibration of scores to the panel mean.

    Missing/constant-score pairs fall back to the training-fold mean.  Prediction
    between knots is linear (``numpy.interp``), outside the knots it is clipped.
    """
    score = _as_1d_float(scores, 'scores')
    target = _as_1d_float(panel_mean_ref, 'panel_mean_ref')
    if score.size != target.size:
        raise ValueError('scores and panel_mean_ref must align')
    group_array, fold_array = _validate_keys(group_ids, fold_ids, target.size)
    predictions = np.full(target.size, np.nan)
    for group, fold, train, test in _group_fold_slices(group_array, fold_array):
        if not train.any():
            raise ValueError('group %r fold %r has no training cells' % (group, fold))
        train_x, train_y = score[train], target[train]
        if np.unique(train_x).size < 2:
            predictions[test] = float(train_y.mean())
            continue
        grid, fitted = _pava_isotonic(train_x, train_y)
        predictions[test] = np.interp(score[test], grid, fitted, left=fitted[0], right=fitted[-1])
    if not np.isfinite(predictions).all():
        raise ValueError('some cells received no isotonic prediction')
    return predictions


def quantile_calibrate(scores, panel_median_ref, group_ids, fold_ids):
    """Fold-wise per-target marginal (quantile) calibration to the panel median.

    Each test score is mapped to its mid-rank quantile among the training scores and
    then to the empirical training panel-median quantile at that level
    (``method='lower'``).  This preserves the target's marginal grade distribution
    without touching the within-target ranking.
    """
    score = _as_1d_float(scores, 'scores')
    target = _as_1d_float(panel_median_ref, 'panel_median_ref')
    if score.size != target.size:
        raise ValueError('scores and panel_median_ref must align')
    group_array, fold_array = _validate_keys(group_ids, fold_ids, target.size)
    predictions = np.full(target.size, np.nan)
    for group, fold, train, test in _group_fold_slices(group_array, fold_array):
        if not train.any():
            raise ValueError('group %r fold %r has no training cells' % (group, fold))
        train_scores = np.sort(score[train])
        train_medians = np.sort(target[train])
        values = score[test]
        left = np.searchsorted(train_scores, values, side='left')
        right = np.searchsorted(train_scores, values, side='right')
        ranks = (left + right) / (2.0 * train_scores.size)
        predictions[test] = np.quantile(train_medians, np.clip(ranks, 0.0, 1.0), method='lower')
    if not np.isfinite(predictions).all():
        raise ValueError('some cells received no quantile prediction')
    return predictions


def linear_to_grade(values, minimum=_GRADE_MIN, maximum=_GRADE_MAX):
    """Round a continuous score to an integer grade and clip to ``[minimum, maximum]``.

    Rounding is ``numpy.rint`` (ties to even), matching the hand-off tables.
    """
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1:
        raise ValueError('values must be one-dimensional')
    if not np.isfinite(scores).all():
        raise ValueError('values must contain only finite values')
    if not (0 <= minimum < maximum <= _GRADE_MAX):
        raise ValueError('grade bounds must satisfy 0 <= minimum < maximum <= 6')
    return np.clip(np.rint(scores), minimum, maximum).astype(np.int64)


def high_grade_stats(pred_grades, panel_median, threshold):
    """Counts/precision/recall for predicting ``grade >= threshold``.

    Truth is ``panel_median >= threshold`` on the same 12,500 cells; this is a
    cell-level panel metric, not a per-taster accuracy.
    """
    grades = _as_1d_float(pred_grades, 'pred_grades')
    truth = _as_1d_float(panel_median, 'panel_median')
    if grades.size != truth.size:
        raise ValueError('pred_grades and panel_median must align')
    event = float(threshold)
    if not np.isfinite(event) or event < 0 or event > _GRADE_MAX:
        raise ValueError('threshold must be finite and within [0, 6]')
    predicted = grades >= event
    observed = truth >= event
    true_positive = int(np.sum(predicted & observed))
    n_pred = int(predicted.sum())
    n_obs = int(observed.sum())
    precision = true_positive / n_pred if n_pred else 0.0
    recall = true_positive / n_obs if n_obs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'threshold': event,
        'n_predicted': n_pred,
        'n_observed': n_obs,
        'true_positive': true_positive,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def enforce_parent_child(grades, parent_indices, child_indices):
    """Deterministically cap child grades at ``<= parent`` for every beer.

    Parameters
    ----------
    grades : array-like of int, shape (n_beers, n_targets)
        Grades already assigned by the decision rule (0..6).
    parent_indices, child_indices : array-like of int, shape (n_pairs,)
        Column positions of the parent and child target of every card pair.

    Returns
    -------
    numpy.ndarray of int64, shape (n_beers, n_targets)
        Same shape with ``grades[:, child] <= grades[:, parent]`` for every pair.

    Notes
    -----
    The card rule is a one-level checkbox hierarchy, but the operation still
    repeats the (deterministic) cap pass until no violation remains, so a chained
    pair definition cannot oscillate.  A grade is only ever lowered, hence the
    result is independent of the number of passes.
    """
    result = np.asarray(grades, dtype=np.int64).copy()
    if result.ndim != 2:
        raise ValueError('grades must be a (n_beers, n_targets) matrix')
    n_beers, n_targets = result.shape
    parents = np.asarray(parent_indices, dtype=np.int64)
    children = np.asarray(child_indices, dtype=np.int64)
    if parents.ndim != 1 or children.ndim != 1 or parents.size != children.size:
        raise ValueError('parent_indices and child_indices must be aligned 1-D arrays')
    if parents.size and (parents.min() < 0 or children.min() < 0
                         or parents.max() >= n_targets or children.max() >= n_targets):
        raise ValueError('parent/child indices must address columns of grades')
    if np.any(parents == children):
        raise ValueError('a target cannot be its own parent')
    for _ in range(min(int(parents.size), 32) + 1):
        before = result.copy()
        for parent, child in zip(parents, children):
            result[:, child] = np.minimum(result[:, child], result[:, parent])
        if np.array_equal(before, result):
            break
    return result


def _calibration_keys(group_ids, fold_ids, size):
    """Validate and normalise calibration keys to ``(str(group), str(fold))``."""
    group_array, fold_array = _validate_keys(group_ids, fold_ids, size)
    return [(str(group), str(fold)) for group, fold in zip(group_array, fold_array)]


def _key_rows(keys):
    """Group row positions by calibration key, preserving input order."""
    positions = {}
    for index, key in enumerate(keys):
        positions.setdefault(key, []).append(index)
    return {key: np.asarray(rows, dtype=np.int64) for key, rows in positions.items()}


def fit_nested_linear(features, reference, group_ids, fold_ids):
    """Fit per-(target, fold) linear calibrators on inner rows only.

    ``features`` rows are inner out-of-fold predictions produced *inside* the
    training folds of the matching outer fold; ``group_ids`` names the target and
    ``fold_ids`` the outer fold that the rows belong to.  Returning a dictionary
    keyed by ``(target, outer_fold)`` keeps the outer validation rows out of the fit.
    """
    design = np.asarray(features, dtype=float)
    target = _as_1d_float(reference, 'reference')
    if design.ndim == 1:
        design = design[:, None]
    if design.ndim != 2 or design.shape[0] != target.size:
        raise ValueError('features must be (n, k) and align with reference')
    if not np.isfinite(design).all():
        raise ValueError('features must contain only finite values')
    keys = _calibration_keys(group_ids, fold_ids, target.size)
    models = {}
    for key, rows in _key_rows(keys).items():
        models[key] = _least_squares_fit(design[rows], target[rows])
    return models


def apply_nested_linear(features, group_ids, fold_ids, models):
    """Apply calibrators fitted by :func:`fit_nested_linear` to outer rows."""
    design = np.asarray(features, dtype=float)
    if design.ndim == 1:
        design = design[:, None]
    if design.ndim != 2 or not np.isfinite(design).all():
        raise ValueError('features must be a finite (n, k) matrix')
    keys = _calibration_keys(group_ids, fold_ids, design.shape[0])
    predictions = np.empty(design.shape[0], dtype=float)
    for row, key in enumerate(keys):
        if key not in models:
            raise ValueError('no calibrator for %r' % (key,))
        intercept, slope = models[key]
        predictions[row] = float(intercept) + design[row] @ np.asarray(slope, float)
    if not np.isfinite(predictions).all():
        raise ValueError('non-finite nested linear prediction')
    return predictions


def fit_nested_isotonic(scores, reference, group_ids, fold_ids):
    """Fit per-(target, outer fold) isotonic calibrators on inner rows only."""
    score = _as_1d_float(scores, 'scores')
    target = _as_1d_float(reference, 'reference')
    if score.size != target.size:
        raise ValueError('scores and reference must align')
    keys = _calibration_keys(group_ids, fold_ids, target.size)
    models = {}
    for key, rows in _key_rows(keys).items():
        train_x, train_y = score[rows], target[rows]
        if np.unique(train_x).size < 2:
            models[key] = {'constant': float(train_y.mean())}
        else:
            grid, fitted = _pava_isotonic(train_x, train_y)
            models[key] = {'grid': grid, 'fitted': fitted}
    return models


def apply_nested_isotonic(scores, group_ids, fold_ids, models):
    """Apply calibrators fitted by :func:`fit_nested_isotonic` to outer rows."""
    score = _as_1d_float(scores, 'scores')
    keys = _calibration_keys(group_ids, fold_ids, score.size)
    predictions = np.empty(score.size, dtype=float)
    for row, key in enumerate(keys):
        if key not in models:
            raise ValueError('no isotonic calibrator for %r' % (key,))
        model = models[key]
        if 'constant' in model:
            predictions[row] = float(model['constant'])
        else:
            grid = np.asarray(model['grid'], float)
            fitted = np.asarray(model['fitted'], float)
            predictions[row] = float(np.interp(score[row], grid, fitted,
                                               left=fitted[0], right=fitted[-1]))
    if not np.isfinite(predictions).all():
        raise ValueError('non-finite nested isotonic prediction')
    return predictions


def fit_nested_quantile(scores, panel_median_ref, group_ids, fold_ids):
    """Fit per-(target, outer fold) marginal quantile maps on inner rows only."""
    score = _as_1d_float(scores, 'scores')
    target = _as_1d_float(panel_median_ref, 'panel_median_ref')
    if score.size != target.size:
        raise ValueError('scores and panel_median_ref must align')
    keys = _calibration_keys(group_ids, fold_ids, target.size)
    models = {}
    for key, rows in _key_rows(keys).items():
        models[key] = {'scores': np.sort(score[rows]), 'medians': np.sort(target[rows])}
    return models


def apply_nested_quantile(scores, group_ids, fold_ids, models):
    """Apply calibrators fitted by :func:`fit_nested_quantile` to outer rows."""
    score = _as_1d_float(scores, 'scores')
    keys = _calibration_keys(group_ids, fold_ids, score.size)
    predictions = np.empty(score.size, dtype=float)
    for row, key in enumerate(keys):
        if key not in models:
            raise ValueError('no quantile calibrator for %r' % (key,))
        train_scores = np.asarray(models[key]['scores'], float)
        train_medians = np.asarray(models[key]['medians'], float)
        left = np.searchsorted(train_scores, score[row], side='left')
        right = np.searchsorted(train_scores, score[row], side='right')
        rank = (left + right) / (2.0 * train_scores.size)
        predictions[row] = float(np.quantile(train_medians, min(max(rank, 0.0), 1.0),
                                             method='lower'))
    return predictions


def observed_high_grade_rates(panel_median):
    """Return per-target high-grade fractions of a panel-median matrix.

    Parameters
    ----------
    panel_median : array-like of int, shape (n_beers, n_targets)
        Integer panel medians on the 0-6 card scale.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        ``(rate_ge5, rate_6)`` fractions in ``[0, 1]`` per target.  These are the
        rates the frozen official decision rule stored in the runtime bundle and
        the same quantities a fold's training slice supplies during evaluation.
    """
    matrix = np.asarray(panel_median)
    if matrix.ndim != 2:
        raise ValueError('panel_median must be two-dimensional, got shape %r'
                         % (matrix.shape,))
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError('panel_median must not be empty')
    values = np.rint(matrix.astype(float))
    if not np.isfinite(values).all():
        raise ValueError('panel_median must be finite')
    if ((values < _GRADE_MIN) | (values > _GRADE_MAX)).any():
        raise ValueError('panel_median must lie on the 0-6 card scale')
    return (values >= 5.0).mean(axis=0), (values >= 6.0).mean(axis=0)


def runtime_decision(p_ge5, p6, base_grades, rate_ge5, rate_6, target_order,
                     parent_indices=None, child_indices=None):
    """Apply the official top-k rule plus the card parent/child cap at inference.

    Parameters
    ----------
    p_ge5, p6, base_grades : array-like of float, shape (n_beers, n_targets)
        Per-cell probabilities and the integer conditional median (capped at 4 by
        the rule itself).
    rate_ge5, rate_6 : array-like of float, shape (n_targets,)
        Training-panel high-grade rates stored in the model bundle.
    target_order : sequence of str, length ``n_targets``
        Column order of the arrays (the bundle's ``targets``).
    parent_indices, child_indices : array-like of int, optional
        Card pairs as column positions; the parent cap is applied after the top-k
        decision, exactly like the frozen official table.

    Returns
    -------
    numpy.ndarray of int64, shape (n_beers, n_targets)
        Grades after the shared :func:`topk_decision` implementation and,
        optionally, :func:`enforce_parent_child`.
    """
    prob5 = np.asarray(p_ge5, dtype=float)
    prob6 = np.asarray(p6, dtype=float)
    base = np.asarray(base_grades, dtype=float)
    if prob5.ndim != 2 or prob5.shape != prob6.shape or prob5.shape != base.shape:
        raise ValueError('p_ge5, p6 and base_grades must share a (n_beers, n_targets) shape')
    n_beers, n_targets = prob5.shape
    order = [str(name) for name in target_order]
    if len(order) != n_targets:
        raise ValueError('target_order must name every column')
    rates5 = _as_1d_float(rate_ge5, 'rate_ge5')
    rates6 = _as_1d_float(rate_6, 'rate_6')
    if rates5.size != n_targets or rates6.size != n_targets:
        raise ValueError('rate arrays must be one value per target')
    if ((rates5 < 0.0) | (rates5 > 1.0)).any() or ((rates6 < 0.0) | (rates6 > 1.0)).any():
        raise ValueError('rate arrays must lie in [0, 1]')
    groups = np.tile(np.asarray(order, dtype=object), n_beers)
    grades = topk_decision(prob5.ravel(), prob6.ravel(), base.ravel(),
                           np.tile(rates5, n_beers), np.tile(rates6, n_beers), groups)
    matrix = grades.reshape(n_beers, n_targets)
    if parent_indices is None and child_indices is None:
        return matrix
    if parent_indices is None or child_indices is None:
        raise ValueError('parent_indices and child_indices must be given together')
    return enforce_parent_child(matrix, parent_indices, child_indices)
