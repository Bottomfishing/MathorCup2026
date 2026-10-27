"""Pure numerical helpers for the Q1 panel-mean redirection experiment (task-9).

The module contains no I/O, no paths and no model fitting.  It only turns the
official weighted 0-6 rating distributions into panel-mean/panel-median truth,
scores candidate predictions against that truth, runs the pre-registered paired
beer bootstrap and applies the pre-registered adoption gates.

Conventions
-----------
``distribution`` is an array whose last axis enumerates grades ``0..6`` with
probability mass (weights summing to one) per ``(beer, target)`` cell.
``panel_mean`` is ``sum_k k * P(grade=k)``; ``panel_median`` is the lower
weighted median (smallest ``k`` with cumulative mass ``>= 0.5``).
Grades predicted from a continuous panel mean are ``clip(rint(x), 0, 6)`` with
banker's rounding (ties to even), matching ``beer_models.q1_decision.linear_to_grade``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "GRADE_VALUES",
    "panel_mean_from_distribution",
    "panel_median_from_distribution",
    "round_grades",
    "absolute_errors",
    "per_beer_mean_error",
    "paired_beer_bootstrap",
    "macro_spearman",
    "pearson_correlation",
    "within_one_rate",
    "exact_rate",
    "panel_mean_metrics",
    "gate_verdicts",
    "target_family",
]

GRADE_VALUES = np.arange(7, dtype=float)


def _as_float_array(values, name="values"):
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError(f"{name} must be an array, got a scalar")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _check_distribution(distribution):
    array = _as_float_array(distribution, "distribution")
    if array.shape[-1] != 7:
        raise ValueError("distribution last axis must enumerate grades 0..6")
    if np.any(array < -1e-12):
        raise ValueError("distribution contains negative probability mass")
    totals = array.sum(axis=-1)
    if np.any(np.abs(totals - 1.0) > 1e-6):
        raise ValueError("distribution rows must sum to one")
    return array


def panel_mean_from_distribution(distribution):
    """Return the weighted panel mean per cell (``...`` shape, grades removed)."""
    array = _check_distribution(distribution)
    return np.tensordot(array, GRADE_VALUES, axes=([-1], [0]))


def panel_median_from_distribution(distribution):
    """Return the lower weighted median grade per cell as float values."""
    array = _check_distribution(distribution)
    cumulative = np.cumsum(array, axis=-1)
    index = np.argmax(cumulative >= 0.5 - 1e-12, axis=-1)
    return index.astype(float)


def round_grades(predictions, lower=0, upper=6):
    """Banker's rounding to integer grades, clipped to ``[lower, upper]``."""
    array = _as_float_array(predictions, "predictions")
    return np.clip(np.rint(array), lower, upper).astype(int)


def _check_pair(predictions, truth):
    predicted = _as_float_array(predictions, "predictions")
    reference = _as_float_array(truth, "truth")
    if predicted.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: predictions {predicted.shape} vs truth {reference.shape}"
        )
    return predicted, reference


def absolute_errors(predictions, truth):
    """Element-wise absolute error."""
    predicted, reference = _check_pair(predictions, truth)
    return np.abs(predicted - reference)


def per_beer_mean_error(predictions, truth):
    """Mean absolute error per beer (rows), averaged over the remaining axes."""
    errors = absolute_errors(predictions, truth)
    if errors.ndim == 1:
        return errors
    return errors.reshape(errors.shape[0], -1).mean(axis=1)


def _average_ranks(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def pearson_correlation(left, right):
    """Pearson correlation of two paired 1-D arrays (``nan`` if degenerate)."""
    a = _as_float_array(left, "left").ravel()
    b = _as_float_array(right, "right").ravel()
    if a.size != b.size:
        raise ValueError("left and right must have the same length")
    if a.size < 3:
        raise ValueError("at least three observations are required")
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = np.sqrt((a_centered ** 2).sum() * (b_centered ** 2).sum())
    if denominator <= 0:
        return float("nan")
    return float((a_centered * b_centered).sum() / denominator)


def macro_spearman(predictions, truth):
    """Mean per-target Spearman rho over columns (``predictions`` shape ``n x m``).

    Columns whose truth is constant (fewer than three distinct values) are
    skipped, mirroring the official Q1 macro-Spearman definition.
    """
    predicted = _as_float_array(predictions, "predictions")
    reference = _as_float_array(truth, "truth")
    if predicted.shape != reference.shape:
        raise ValueError("predictions and truth must have the same shape")
    if predicted.ndim != 2:
        raise ValueError("predictions must be two-dimensional (beers x targets)")
    rhos = []
    for column in range(predicted.shape[1]):
        truth_column = reference[:, column]
        if np.unique(truth_column).size < 3:
            continue
        predicted_ranks = _average_ranks(predicted[:, column])
        truth_ranks = _average_ranks(truth_column)
        predicted_ranks -= predicted_ranks.mean()
        truth_ranks -= truth_ranks.mean()
        denominator = np.sqrt(
            (predicted_ranks ** 2).sum() * (truth_ranks ** 2).sum()
        )
        if denominator > 0:
            rhos.append(float((predicted_ranks * truth_ranks).sum() / denominator))
    if not rhos:
        return float("nan"), 0
    return float(np.mean(rhos)), len(rhos)


def within_one_rate(predicted_grades, panel_median):
    """Share of cells whose predicted integer grade is within one grade."""
    predicted = np.asarray(predicted_grades, dtype=int)
    median = np.asarray(panel_median)
    if predicted.shape != median.shape:
        raise ValueError("predicted_grades and panel_median must have the same shape")
    return float(np.mean(np.abs(predicted - median) <= 1))


def exact_rate(predicted_grades, panel_median):
    """Share of cells whose predicted integer grade equals the panel median."""
    predicted = np.asarray(predicted_grades, dtype=int)
    median = np.asarray(panel_median)
    if predicted.shape != median.shape:
        raise ValueError("predicted_grades and panel_median must have the same shape")
    return float(np.mean(predicted == median))


def panel_mean_metrics(predictions, panel_mean, panel_median, calibrated_reference=None):
    """Standard metric bundle for one candidate and one scope.

    ``predictions``, ``panel_mean`` and ``panel_median`` must share one shape;
    ``calibrated_reference`` (optional, same shape) is the frozen official
    linear-calibrated panel mean used for the "correlation with old column"
    secondary metric.
    """
    predicted = _as_float_array(predictions, "predictions")
    reference = _as_float_array(panel_mean, "panel_mean")
    median = _as_float_array(panel_median, "panel_median")
    if predicted.shape != reference.shape or predicted.shape != median.shape:
        raise ValueError("predictions, panel_mean and panel_median must share a shape")
    rounded = round_grades(predicted)
    rho, n_targets = macro_spearman(predicted, reference)
    metrics = {
        "panel_mean_mae": float(np.mean(np.abs(predicted - reference))),
        "panel_mean_rmse": float(np.sqrt(np.mean((predicted - reference) ** 2))),
        "panel_median_exact": exact_rate(rounded, median),
        "panel_median_within_one": within_one_rate(rounded, median),
        "macro_spearman": rho,
        "macro_spearman_targets": int(n_targets),
        "pearson_with_calibrated_baseline": float("nan"),
    }
    if calibrated_reference is not None:
        metrics["pearson_with_calibrated_baseline"] = pearson_correlation(
            predicted.ravel(), np.asarray(calibrated_reference, dtype=float).ravel()
        )
    return metrics


def paired_beer_bootstrap(differences, styles, n_boot=1000, seed=20261026, alpha=0.05):
    """Paired beer bootstrap of the mean per-beer error improvement.

    ``differences`` holds one value per beer (positive means the candidate is
    better than the baseline).  Beers are resampled with replacement inside
    each tasting style, matching the official style-stratified bootstrap.
    Returns a dict with the point estimate and the percentile interval.
    """
    deltas = _as_float_array(differences, "differences").ravel()
    style_array = np.asarray(styles)
    if style_array.shape[0] != deltas.shape[0]:
        raise ValueError("styles must have one entry per beer")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    unique_styles, style_codes = np.unique(style_array.astype(str), return_inverse=True)
    members = [np.flatnonzero(style_codes == code) for code in range(unique_styles.size)]
    if any(group.size == 0 for group in members):
        raise ValueError("every style group must contain at least one beer")
    generator = np.random.default_rng(seed)
    draws = np.empty(int(n_boot), dtype=float)
    for draw in range(int(n_boot)):
        sample = np.concatenate(
            [
                group[generator.integers(0, group.size, size=group.size)]
                for group in members
            ]
        )
        draws[draw] = deltas[sample].mean()
    return {
        "point": float(deltas.mean()),
        "lower": float(np.quantile(draws, alpha / 2.0)),
        "upper": float(np.quantile(draws, 1.0 - alpha / 2.0)),
        "alpha": float(alpha),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_styles": int(unique_styles.size),
    }


def gate_verdicts(
    pooled_mae,
    repeat_maes,
    bootstrap_lower,
    pooled_within_one,
    baseline_mae,
    minimum_improvement=0.010,
    within_one_floor=0.9119,
):
    """Apply the four pre-registered adoption gates for one candidate."""
    pooled = float(pooled_mae)
    baseline = float(baseline_mae)
    repeat_values = [float(value) for value in np.asarray(repeat_maes).ravel()]
    if not repeat_values:
        raise ValueError("repeat_maes must not be empty")
    gates = {
        "gate_1_point_improvement": bool(pooled <= baseline - float(minimum_improvement)),
        "gate_2_bootstrap": bool(float(bootstrap_lower) > 0.0),
        "gate_3_every_repeat": bool(all(value < baseline for value in repeat_values)),
        "gate_4_within_one": bool(float(pooled_within_one) >= float(within_one_floor)),
    }
    improvement = baseline - pooled
    verdict = {
        "pooled_panel_mean_mae": pooled,
        "repeat_panel_mean_maes": repeat_values,
        "baseline_panel_mean_mae": baseline,
        "point_improvement": improvement,
        "minimum_improvement": float(minimum_improvement),
        "bootstrap_lower": float(bootstrap_lower),
        "pooled_within_one": float(pooled_within_one),
        "within_one_floor": float(within_one_floor),
        "gates": gates,
        "all_pass": bool(all(gates.values())),
    }
    return verdict


def target_family(target):
    """Map a Q1 target name to the Aroma / Flavour / Supplementary family."""
    name = str(target)
    if name.startswith("A_"):
        return "Aroma"
    if name.startswith("F_"):
        return "Flavour"
    return "Supplementary"
