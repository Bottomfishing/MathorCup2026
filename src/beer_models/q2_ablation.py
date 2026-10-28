"""Neighbourhood-weight ablation for the Q2 compound-factors ranking.

The delivered Q2 table scores every candidate compound with

``score = |beta| + w * (adjacency @ |beta|) / degree``

where ``beta`` is the ridge coefficient vector fitted on the 1.csv chemistry
block and ``adjacency`` comes from the 2.csv co-expression matrix.  ``w = 0.5``
is the delivered setting.  This module provides the pure-function pieces needed
to answer the direct question "does the graph neighbourhood term actually buy
anything compared with ``w = 0``?", i.e. two rankings built from exactly the
same samples, preprocessing, ridge fit and bootstrap resampling, differing only
in ``w``.

Everything here is deterministic and free of IO: callers pass numpy arrays and
explicit seeds.  P-values use a two-sided Wilcoxon signed-rank test across the
50 sensory targets; confidence intervals come from a fixed-seed bootstrap over
targets.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

from beer_models.q2_top3 import (
    DEFAULT_ALPHA,
    DEFAULT_GRAPH_THRESHOLD,
    ridge_coefficients,
    adjacency_from_corr,
    neighbourhood_scores,
    top_k_fields,
)

#: Delivered neighbourhood weight; the ablation switches the graph term off.
GRAPH_WEIGHT = 0.5
ABLATION_WEIGHT = 0.0

#: Grouped cross-validation used by the optional predictive check.
PREDICTIVE_FOLDS = 5
PREDICTIVE_SEED = 20261028
SMALL_RIDGE_ALPHA = 1.0

#: Pre-registered thresholds for the "reliable gain" verdict.
MAE_GAIN_THRESHOLD = 0.01
SPEARMAN_GAIN_THRESHOLD = 0.02
STABILITY_JACCARD_DELTA = 0.02
STABILITY_FREQ_DELTA = 0.05
MIN_NON_WORSE_FRACTION = 0.60
SIGNIFICANCE = 0.05


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), used for the fast Spearman computation."""
    return rankdata(values, method="average")


def spearman_matrix(values: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix of the columns of ``values``.

    Missing values are replaced by the column median first (the same imputation
    the Q2 pipeline uses).  Ranking the columns once and taking the Pearson
    correlation of the ranks yields the exact Spearman coefficients with a
    single matrix product, which keeps the per-fold 219 x 219 graph cheap.
    """

    array = np.asarray(values, dtype=float).copy()
    medians = np.nanmedian(array, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    missing = np.isnan(array)
    if missing.any():
        rows, columns = np.where(missing)
        array[rows, columns] = medians[columns]
    ranks = np.apply_along_axis(_rank, 0, array)
    ranks = ranks - ranks.mean(axis=0, keepdims=True)
    scale = ranks.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    ranks = ranks / scale
    correlations = ranks.T @ ranks / float(array.shape[0])
    correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlations, 1.0)
    return correlations


def grouped_folds(
    n_samples: int,
    n_folds: int = PREDICTIVE_FOLDS,
    seed: int = PREDICTIVE_SEED,
) -> np.ndarray:
    """Deterministic beer-level fold assignment (one row per beer)."""

    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if n_samples < n_folds:
        raise ValueError("n_samples must be at least n_folds")
    generator = np.random.default_rng(seed)
    labels = np.empty(n_samples, dtype=int)
    for position, index in enumerate(generator.permutation(n_samples)):
        labels[index] = position % n_folds
    return labels


def standardise_train_test(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute, arcsinh and standardise using statistics from ``train``.

    Both blocks must share the same columns.  This is the leakage-free version
    of :func:`beer_models.q2_top3.preprocess` for held-out prediction: the test
    block is transformed with the training medians, means and scales only.
    """

    train = np.asarray(train, dtype=float).copy()
    test = np.asarray(test, dtype=float).copy()
    medians = np.nanmedian(train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    for block in (train, test):
        missing = np.isnan(block)
        if missing.any():
            rows, columns = np.where(missing)
            block[rows, columns] = medians[columns]
    train = np.arcsinh(train)
    test = np.arcsinh(test)
    means = train.mean(axis=0)
    scales = train.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    return (train - means) / scales, (test - means) / scales


def grouped_cv_predictions(
    X: np.ndarray,
    Y: np.ndarray,
    weight: float,
    folds: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    threshold: float = DEFAULT_GRAPH_THRESHOLD,
    small_alpha: float = SMALL_RIDGE_ALPHA,
) -> tuple[np.ndarray, dict]:
    """Held-out predictions of the "top-3 then small ridge" reader.

    For every fold the neighbourhood graph is rebuilt from the *training* beers
    only (2.csv is a full-sample matrix and must not leak into a generalisation
    comparison), the three compounds per target are selected with the requested
    ``weight``, and a small ridge on those three standardised features predicts
    the held-out 4.csv responses.  Returns the out-of-fold prediction matrix
    and the per-fold per-target selected column indices.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    folds = np.asarray(folds, dtype=int)
    predictions = np.full(Y.shape, np.nan, dtype=float)
    selections: dict[tuple[int, int], tuple[int, ...]] = {}
    n_features = X.shape[1]
    feature_ids = list(range(n_features))
    for fold in np.unique(folds):
        train_rows = folds != fold
        test_rows = ~train_rows
        X_train, X_test = standardise_train_test(X[train_rows], X[test_rows])
        Y_train = Y[train_rows]
        Y_centred = Y_train - Y_train.mean(axis=0)
        beta = ridge_coefficients(X_train, Y_centred, alpha)
        if weight > 0.0:
            adjacency, _ = adjacency_from_corr(spearman_matrix(X[train_rows]), threshold)
            scores = neighbourhood_scores(beta, adjacency, weight)
        else:
            scores = np.abs(beta)
        selected = top_k_fields(scores, feature_ids, 3)
        for target_index in range(Y.shape[1]):
            columns = np.asarray(selected[target_index], dtype=int)
            small = ridge_coefficients(
                X_train[:, columns], Y_centred[:, target_index : target_index + 1], small_alpha
            )
            predictions[test_rows, target_index] = (
                X_test[:, columns] @ small[:, 0] + Y_train[:, target_index].mean()
            )
            selections[(int(fold), int(target_index))] = tuple(int(c) for c in columns)
    return predictions, selections


def predictive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, list[float]]:
    """Per-target MAE and Spearman correlation of out-of-fold predictions."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae: list[float] = []
    rho: list[float] = []
    for target_index in range(y_true.shape[1]):
        mask = ~np.isnan(y_pred[:, target_index])
        truth = y_true[mask, target_index]
        estimate = y_pred[mask, target_index]
        mae.append(float(np.mean(np.abs(truth - estimate))))
        rho.append(float(spearmanr(truth, estimate).statistic))
    return {"mae": mae, "spearman": rho}


def paired_test(
    left: np.ndarray,
    right: np.ndarray,
    seed: int = PREDICTIVE_SEED,
    n_bootstrap: int = 4000,
) -> dict:
    """Paired comparison of two 50-target metric vectors.

    ``difference = left - right``; positive values mean the left variant is
    larger.  Returns the mean/median difference, a two-sided Wilcoxon p-value
    and a fixed-seed bootstrap 95% CI of the mean difference.  Counts of
    left-better / right-better / ties are also reported so callers can judge
    whether a nominal mean gain is broad or driven by a few targets.
    """

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    difference = left - right
    if np.allclose(difference, 0.0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(difference, zero_method="wilcox").pvalue)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, difference.size, (n_bootstrap, difference.size))
    means = difference[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return {
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": p_value,
        "n_left_larger": int(np.sum(difference > 0)),
        "n_right_larger": int(np.sum(difference < 0)),
        "n_ties": int(np.sum(difference == 0)),
        "n_targets": int(difference.size),
    }


def overlap_row(
    target: str,
    graph_top3: list[str],
    no_graph_top3: list[str],
    graph_scores: np.ndarray,
    no_graph_scores: np.ndarray,
    graph_freq: dict[str, float],
    no_graph_freq: dict[str, float],
    fields: list[str],
    graph_tiers: dict[str, int],
    no_graph_tiers: dict[str, int],
) -> dict:
    """Per-target agreement between the two rankings.

    ``jaccard`` / ``overlap_n`` describe the selected sets, ``rank_spearman``
    compares the full 219-compound importance order, and ``top3_rank_spearman``
    compares only the compounds selected by both variants.  Frequency and
    evidence-tier deltas are reported as descriptive differences -- Jaccard is
    a similarity measure and does not establish which ranking is correct.
    """

    graph_set = set(graph_top3)
    no_graph_set = set(no_graph_top3)
    overlap = graph_set & no_graph_set
    union = graph_set | no_graph_set
    graph_ranks = _rank_magnitudes(graph_scores, fields)
    no_graph_ranks = _rank_magnitudes(no_graph_scores, fields)
    graph_all = np.array([graph_ranks[field] for field in fields])
    no_graph_all = np.array([no_graph_ranks[field] for field in fields])
    shared = sorted(overlap, key=fields.index)
    graph_shared = np.array([graph_ranks[field] for field in shared])
    no_graph_shared = np.array([no_graph_ranks[field] for field in shared])
    if shared and not np.allclose(graph_shared, no_graph_shared):
        top3_rho = float(spearmanr(graph_shared, no_graph_shared).statistic)
    else:
        top3_rho = 1.0 if shared else 0.0
    return {
        "target": target,
        "jaccard": len(overlap) / len(union) if union else 0.0,
        "overlap_n": len(overlap),
        "overlap_fields": "; ".join(shared),
        "rank_spearman": float(spearmanr(graph_all, no_graph_all).statistic),
        "top3_rank_spearman": top3_rho,
        "graph_mean_freq": float(np.mean([graph_freq.get(field, 0.0) for field in graph_top3])),
        "no_graph_mean_freq": float(np.mean([no_graph_freq.get(field, 0.0) for field in no_graph_top3])),
        "freq_difference": float(
            np.mean([graph_freq.get(field, 0.0) for field in graph_top3])
            - np.mean([no_graph_freq.get(field, 0.0) for field in no_graph_top3])
        ),
        "graph_A": int(sum(graph_tiers.get(field, 0) == 1 for field in graph_top3)),
        "graph_B": int(sum(graph_tiers.get(field, 0) == 2 for field in graph_top3)),
        "graph_C": int(sum(graph_tiers.get(field, 0) == 3 for field in graph_top3)),
        "no_graph_A": int(sum(no_graph_tiers.get(field, 0) == 1 for field in no_graph_top3)),
        "no_graph_B": int(sum(no_graph_tiers.get(field, 0) == 2 for field in no_graph_top3)),
        "no_graph_C": int(sum(no_graph_tiers.get(field, 0) == 3 for field in no_graph_top3)),
    }


def _rank_magnitudes(scores: np.ndarray, fields: list[str]) -> dict[str, float]:
    """Per-field average rank of the importance magnitudes (1 = most important)."""

    magnitudes = np.abs(np.asarray(scores, dtype=float)).ravel()
    order = np.argsort(-magnitudes, kind="stable")
    ranks = np.empty(magnitudes.size, dtype=float)
    ranks[order] = np.arange(1, magnitudes.size + 1)
    return {field: float(ranks[index]) for index, field in enumerate(fields)}


def stability_row(
    variant: str,
    slot_frequencies: list[float],
    bootstrap_jaccard_means: list[float],
    n_targets: int,
) -> dict:
    """Summary of bootstrap stability for one variant."""

    frequencies = np.asarray(slot_frequencies, dtype=float)
    per_target_mean = frequencies.reshape(n_targets, 3).mean(axis=1)
    return {
        "variant": variant,
        "n_top3_rows": int(frequencies.size),
        "freq_mean": float(frequencies.mean()),
        "freq_median": float(np.median(frequencies)),
        "freq_min": float(frequencies.min()),
        "freq_max": float(frequencies.max()),
        "n_slots_freq_ge_0_5": int(np.sum(frequencies >= 0.5)),
        "n_targets_mean_freq_ge_0_5": int(np.sum(per_target_mean >= 0.5)),
        "bootstrap_jaccard_mean": float(np.mean(bootstrap_jaccard_means)),
        "bootstrap_jaccard_min": float(np.min(bootstrap_jaccard_means)),
    }


def ablation_verdict(
    mae_stats: dict,
    spearman_stats: dict,
    stability_stats: dict,
    mae_threshold: float = MAE_GAIN_THRESHOLD,
    spearman_threshold: float = SPEARMAN_GAIN_THRESHOLD,
    jaccard_delta: float = STABILITY_JACCARD_DELTA,
    freq_delta: float = STABILITY_FREQ_DELTA,
    min_non_worse: float = MIN_NON_WORSE_FRACTION,
    significance: float = SIGNIFICANCE,
) -> dict:
    """Pre-registered decision rule for "does the graph term buy a reliable gain".

    ``mae_stats['gain_mean']`` is ``MAE(no graph) - MAE(graph)`` so a positive
    value means the delivered graph variant predicts held-out responses better;
    ``spearman_stats['gain_mean']`` is ``rho(graph) - rho(no graph)``.
    ``stability_stats`` carries the two bootstrap summaries for both variants.

    A predictive gain only counts as *reliable* when the mean improvement is at
    least the pre-registered threshold, the paired Wilcoxon p-value is below
    0.05 **and** at least 60% of the 50 targets are not worse (so a few large
    wins cannot carry the claim).  The stability criterion asks whether the
    graph variant is more reproducible by at least one of the two pre-registered
    margins.  The overall verdict is ``reliable_gain`` only when both axes show
    a reliable improvement, ``partial_gain`` when exactly one does, and
    ``no_reliable_gain`` otherwise.
    """

    mae_reliable = (
        mae_stats["gain_mean"] >= mae_threshold
        and mae_stats["p_value"] < significance
        and mae_stats["graph_non_worse_fraction"] >= min_non_worse
    )
    rho_reliable = (
        spearman_stats["gain_mean"] >= spearman_threshold
        and spearman_stats["p_value"] < significance
        and spearman_stats["graph_non_worse_fraction"] >= min_non_worse
    )
    predictive_reliable = bool(mae_reliable or rho_reliable)
    jaccard_gain = (
        stability_stats["graph_jaccard_mean"] - stability_stats["no_graph_jaccard_mean"]
    )
    freq_gain = stability_stats["graph_freq_mean"] - stability_stats["no_graph_freq_mean"]
    stability_reliable = bool(jaccard_gain >= jaccard_delta or freq_gain >= freq_delta)
    if predictive_reliable and stability_reliable:
        verdict = "reliable_gain"
    elif predictive_reliable or stability_reliable:
        verdict = "partial_gain"
    else:
        verdict = "no_reliable_gain"
    return {
        "verdict": verdict,
        "predictive_reliable": predictive_reliable,
        "stability_reliable": stability_reliable,
        "criteria": {
            "mae_gain_threshold": mae_threshold,
            "spearman_gain_threshold": spearman_threshold,
            "stability_jaccard_delta": jaccard_delta,
            "stability_freq_delta": freq_delta,
            "min_non_worse_fraction": min_non_worse,
            "significance": significance,
            "mae_reliable": bool(mae_reliable),
            "spearman_reliable": bool(rho_reliable),
        },
        "observed": {
            "mae_gain_mean": mae_stats["gain_mean"],
            "mae_p_value": mae_stats["p_value"],
            "mae_graph_non_worse_fraction": mae_stats["graph_non_worse_fraction"],
            "spearman_gain_mean": spearman_stats["gain_mean"],
            "spearman_p_value": spearman_stats["p_value"],
            "spearman_graph_non_worse_fraction": spearman_stats["graph_non_worse_fraction"],
            "jaccard_gain": float(jaccard_gain),
            "freq_gain": float(freq_gain),
        },
    }
