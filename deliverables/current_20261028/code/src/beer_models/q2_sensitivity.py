"""Q2 parameter-sensitivity analysis (pure functions, no file IO, no absolute paths).

The delivered Q2 table ranks the 219 individual-analyte candidates per sensory
dimension with ``Ridge + co-expression-graph neighbourhood weighting`` at
``|rho| >= 0.5`` and Louvain resolution 1.0. This module quantifies how far that
Top-3 table moves when the two graph parameters are changed:

* co-expression threshold ``|rho|`` over ``THRESHOLD_GRID``;
* Louvain ``resolution`` over ``RESOLUTION_GRID`` (plus a wider curve for the
  community-count plot).

The pre-registered stability criteria live here so that the script, the tests and
the written protocol all share one definition (no post-hoc thresholds):

* threshold criterion: every boundary variant in ``BOUNDARY_THRESHOLDS`` reaches
  ``jaccard_mean >= 0.5`` and ``>= 40/50`` dimensions keep ``>= 2/3`` Top-3
  compounds (treat the main table as the baseline);
* resolution criterion: every required resolution keeps the community count in
  ``COMMUNITY_RANGE`` and the largest module below ``MAX_LARGEST_SHARE`` of nodes.
"""
import numpy as np
import networkx as nx
from scipy.stats import kendalltau, spearmanr

from .q2_top3 import DEFAULT_GRAPH_THRESHOLD, DEFAULT_LOUVAIN_SEED, jaccard

THRESHOLD_GRID = (0.40, 0.45, 0.50, 0.55, 0.60)
BOUNDARY_THRESHOLDS = (0.45, 0.50, 0.55)
RESOLUTION_GRID = (0.8, 1.0, 1.2)
RESOLUTION_CURVE = tuple(round(0.6 + 0.1 * step, 1) for step in range(11))
DEFAULT_RESOLUTION = 1.0
STABLE_MIN_JACCARD = 0.5
STABLE_MIN_TARGETS = 40
COMMUNITY_RANGE = (40, 80)
MAX_LARGEST_SHARE = 0.20
BASELINE_VARIANT = {'threshold': DEFAULT_GRAPH_THRESHOLD, 'resolution': DEFAULT_RESOLUTION}


def weighted_adjacency(rho, threshold=DEFAULT_GRAPH_THRESHOLD):
    """Symmetric weighted graph: ``|rho|`` on edges above the threshold, else 0.

    Louvain runs on this matrix (edge weight = |rho|); the binary
    ``q2_top3.adjacency_from_corr`` matrix is what the neighbourhood score uses.
    """
    corr = np.asarray(rho, dtype=float).copy()
    if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
        raise ValueError('rho must be square')
    np.fill_diagonal(corr, 0.0)
    weights = np.abs(corr)
    return np.where(weights >= threshold, weights, 0.0)


def community_partition(weighted, resolution=DEFAULT_RESOLUTION, seed=DEFAULT_LOUVAIN_SEED):
    """Louvain partition of a weighted graph, relabelled by decreasing community size.

    This mirrors ``q2_top3.louvain_labels`` (same relabelling and tie-breaking)
    but exposes the resolution parameter, which the delivered module keeps fixed.
    Returns ``(labels, community_sizes)``.
    """
    weights = np.asarray(weighted, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError('weighted adjacency must be square')
    graph = nx.Graph()
    graph.add_nodes_from(range(weights.shape[0]))
    for i, j in zip(*np.triu_indices_from(weights, k=1)):
        if weights[i, j] > 0:
            graph.add_edge(int(i), int(j), weight=float(weights[i, j]))
    communities = nx.algorithms.community.louvain_communities(graph, seed=seed, resolution=resolution)
    communities = sorted(communities, key=len, reverse=True)
    labels = np.full(weights.shape[0], -1, dtype=int)
    for cid, members in enumerate(communities):
        for node in members:
            labels[node] = cid
    return labels, [len(community) for community in communities]


def frequency_table(replicates, fields, n_bootstrap):
    """Bootstrap selection frequency of every candidate field (not only the Top-3)."""
    counts = {}
    for replicate in replicates:
        for field in replicate:
            counts[field] = counts.get(field, 0) + 1
    return {field: counts.get(field, 0) / n_bootstrap for field in fields}


def overlap_metrics(baseline, variant):
    """Per-dimension Top-3 overlap of a variant against the baseline selection.

    ``baseline``/``variant`` map target -> ordered list of field names. Returns
    the per-target rows and the aggregate stability summary.
    """
    rows = []
    for target in baseline:
        base_set, var_set = set(baseline[target]), set(variant[target])
        rows.append({'target': target, 'jaccard': jaccard(base_set, var_set),
                     'overlap': len(base_set & var_set), 'identical': base_set == var_set})
    values = [row['jaccard'] for row in rows]
    overlaps = [row['overlap'] for row in rows]
    return rows, {'n_targets': len(rows),
                  'jaccard_mean': float(np.mean(values)),
                  'jaccard_median': float(np.median(values)),
                  'jaccard_min': float(np.min(values)),
                  'overlap_mean': float(np.mean(overlaps)),
                  'n_targets_jaccard_ge_0_5': int(sum(value >= 0.5 for value in values)),
                  'n_targets_overlap_ge_2': int(sum(value >= 2 for value in overlaps)),
                  'n_targets_identical': int(sum(row['identical'] for row in rows))}


def rank_correlations(baseline_scores, variant_scores, baseline_sets, variant_sets, targets):
    """Ranking stability per dimension between baseline and variant.

    ``importance_spearman`` correlates the full candidate importance vector
    (219 candidates); ``kendall_tau_top3`` compares the order of the compounds the
    two Top-3 sets share (nan when fewer than two compounds are shared).
    """
    rows = []
    for j, target in enumerate(targets):
        rho = float(spearmanr(baseline_scores[:, j], variant_scores[:, j]).statistic)
        base_rank = {field: rank for rank, field in enumerate(baseline_sets[j])}
        var_rank = {field: rank for rank, field in enumerate(variant_sets[j])}
        shared = [field for field in base_rank if field in var_rank]
        tau = float('nan')
        if len(shared) >= 2:
            tau = float(kendalltau([base_rank[field] for field in shared],
                                   [var_rank[field] for field in shared]).statistic)
        rows.append({'target': target, 'importance_spearman': rho,
                     'kendall_tau_top3': tau, 'n_shared': len(shared)})
    return rows


def frequency_rank_correlation(baseline_frequency, variant_frequency, targets, fields):
    """Per-dimension Spearman correlation of the 219 candidate bootstrap frequencies."""
    rows = []
    for target in targets:
        base = [baseline_frequency[target].get(field, 0.0) for field in fields]
        variant = [variant_frequency[target].get(field, 0.0) for field in fields]
        value = float(spearmanr(base, variant).statistic)
        rows.append({'target': target, 'frequency_spearman': value})
    return rows


def threshold_verdict(rows, thresholds=BOUNDARY_THRESHOLDS,
                      min_jaccard=STABLE_MIN_JACCARD, min_targets=STABLE_MIN_TARGETS):
    """Pre-registered threshold-stability criterion, evaluated on boundary variants."""
    by_threshold = {round(float(row['threshold']), 4): row for row in rows}
    details = {}
    for threshold in thresholds:
        row = by_threshold[round(float(threshold), 4)]
        passed = row['jaccard_mean'] >= min_jaccard and row['n_targets_overlap_ge_2'] >= min_targets
        details[f'{threshold:.2f}'] = {'jaccard_mean': float(row['jaccard_mean']),
                                       'jaccard_median': float(row['jaccard_median']),
                                       'n_targets_overlap_ge_2': int(row['n_targets_overlap_ge_2']),
                                       'n_targets_jaccard_ge_0.5': int(row['n_targets_jaccard_ge_0_5']),
                                       'pass': bool(passed)}
    total_targets = int(next(iter(by_threshold.values()))['n_targets'])
    return {'criterion': f'for thresholds {list(thresholds)}: jaccard_mean >= {min_jaccard} and '
                         f'at least {min_targets}/{total_targets} dimensions with overlap >= 2 of 3',
            'passed': all(item['pass'] for item in details.values()),
            'details': details}


def resolution_verdict(rows, grids=RESOLUTION_GRID, community_range=COMMUNITY_RANGE,
                       max_largest_share=MAX_LARGEST_SHARE):
    """Pre-registered resolution-stability criterion on the required grid."""
    by_resolution = {round(float(row['resolution']), 4): row for row in rows}
    low, high = community_range
    details = {}
    for resolution in grids:
        row = by_resolution[round(float(resolution), 4)]
        passed = low <= row['n_communities'] <= high and row['largest_share'] < max_largest_share
        details[f'{resolution:.1f}'] = {'n_communities': int(row['n_communities']),
                                        'largest_community': int(row['largest_community']),
                                        'largest_share': float(row['largest_share']),
                                        'pass': bool(passed)}
    stable = [float(row['resolution']) for row in rows
              if low <= row['n_communities'] <= high and row['largest_share'] < max_largest_share]
    return {'criterion': f'for resolutions {list(grids)}: {low} <= communities <= {high} and '
                         f'largest community < {max_largest_share:.0%} of nodes',
            'passed': all(item['pass'] for item in details.values()),
            'details': details,
            'stable_resolution_range': [min(stable), max(stable)] if stable else None}


def evidence_tier_counts(rows):
    """A/B/C evidence tiers and high/medium/low confidence counts of a detail table."""
    tiers = {tier: int(sum(row['chemical_evidence'] == tier for row in rows)) for tier in ('A', 'B', 'C')}
    confidence = {level: int(sum(row['confidence'] == level for row in rows))
                  for level in ('high', 'medium', 'low')}
    return {'tiers': tiers, 'confidence': confidence}
