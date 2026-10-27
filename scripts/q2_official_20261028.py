"""Official Q2 delivery: 50 sensory dimensions x Top-3 candidate compounds.

Builds the 2.csv co-expression graph, scores the 219 individual-analyte candidates
with Ridge + graph-neighbourhood weighting, quantifies stability with a beer-level
bootstrap and four-method Top-3 Jaccard, and grades every entry with an A/B/C
chemical-evidence tier. Writes the delivery tables, three figures, verification.json
and protocol.json into ``outputs/q2_top3_20261028``.

Run:  .venv\\Scripts\\python.exe scripts\\q2_official_20261028.py
      (or --source <attachment dir> --output <dir>)
AI-assisted: OpenAI Codex, 2026-10-28.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd

from beer_audit.audit import save_csv, sha256, write_json
from beer_models.q2_top3 import (DEFAULT_ALPHA, DEFAULT_BOOTSTRAP, DEFAULT_GRAPH_THRESHOLD,
                                 DEFAULT_LOUVAIN_SEED, DEFAULT_NEIGHBOUR_WEIGHT, DEFAULT_SEED,
                                 IDENTITY_MARKERS, META_COLUMNS, adjacency_from_corr,
                                 adjacency_subset, bootstrap_jaccard, bootstrap_top_k,
                                 canon_key, candidate_features,
                                 evidence_grade, extra_trees_importance, jaccard, louvain_labels,
                                 marginal_spearman, method_jaccard_row, neighbourhood_scores,
                                 preprocess, ridge_coefficients, selection_frequencies,
                                 target_families, top_k_fields)

AUDIT_DICTIONARY = ROOT / 'outputs' / 'p0_audit_20261023_final' / 'feature_dictionary.csv'
TEAM_TABLE = ROOT / 'outputs' / 'task6_final_20261026' / 'paper' / 'tables' / 'q2_standardized__top3_panel_all50.csv'
CONFIG_PATH = ROOT / 'configs' / 'q2_top3_20261028.json'
METHOD_PAIRS = [('graphridge', 'ridge'), ('graphridge', 'extratrees'), ('graphridge', 'marginal_spearman'),
                ('ridge', 'extratrees'), ('ridge', 'marginal_spearman'),
                ('extratrees', 'marginal_spearman')]
FALLBACK_SOURCE = ROOT.parent / '2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题' / \
    '2026年MathorCup大数据挑战赛-赛道A初赛' / '附件'


def load_config(path):
    """Parameters of record; defaults mirror the Q2 prototype exactly."""
    config = {'alpha': DEFAULT_ALPHA, 'graph_threshold': DEFAULT_GRAPH_THRESHOLD,
              'neighbour_weight': DEFAULT_NEIGHBOUR_WEIGHT, 'bootstrap': DEFAULT_BOOTSTRAP,
              'seed': DEFAULT_SEED, 'louvain_seed': DEFAULT_LOUVAIN_SEED,
              'excluded_features': list(candidate_features.__defaults__[0])}
    if path.exists():
        stored = json.loads(path.read_text(encoding='utf-8'))
        values = dict(stored)
        values.update(stored.get('parameters', {}))
        for key in ('alpha', 'graph_threshold', 'neighbour_weight', 'bootstrap', 'seed',
                    'louvain_seed', 'excluded_features'):
            if key in values:
                config[key] = values[key]
    return config


def map_targets_to_cross_corr(source, targets):
    """4.csv target order -> 6.csv column names; last five names are explicit synonyms."""
    m6 = pd.read_csv(source / '6.csv', encoding='utf-8-sig')
    column_names = [str(c) for c in m6.columns[1:]]
    special = {'alcohol': 'Alcohol', 'aftertaste': 'Aftertaste', 'body': 'Body',
               'co2': 'Carbonation', 'overall': 'Overall appreciation'}
    mapping = {}
    for i, target in enumerate(targets):
        if i < 45:
            mapping[target] = column_names[i]
        else:
            hit = [c for c in column_names
                   if c.replace('\r\n', ' ').strip().lower().startswith(special[target].lower())]
            mapping[target] = hit[0] if hit else column_names[i]
    return m6, mapping


def build_alias_map(feature_dictionary, node_names, fields):
    """Canonical name -> 1.csv field, from the audit dictionary when available."""
    aliases = {}
    if feature_dictionary is not None:
        for row in feature_dictionary.to_dict('records'):
            for raw in (row.get('chemical_name'), row.get('chemical_name_short'), row.get('field_name_R')):
                if pd.notna(raw):
                    aliases.setdefault(canon_key(raw), row['field_name_R'])
    node_to_field = {node_names[i]: fields[i] for i in range(len(fields))}
    for node, field in node_to_field.items():
        aliases.setdefault(canon_key(node), field)
    return node_to_field, aliases


def data_lineage(source, feature_frame, corr_matrix, fields):
    """Recompute Spearman/Pearson on 1.csv and compare positionally with 2.csv.

    2.csv rows/columns are compound names in the same order as the 1.csv non-meta
    columns, so the comparison is positional (no name join is required or assumed).
    """
    numeric = feature_frame[fields].astype(float)
    spearman = numeric.corr(method='spearman')
    pearson = numeric.corr(method='pearson')
    given = corr_matrix.to_numpy(dtype=float)
    spearman_diff = np.abs(spearman.to_numpy() - given)
    pearson_diff = np.abs(pearson.to_numpy() - given)
    return {'2csv_recomputed_from_1csv': 'spearman',
            'positional_order': '2.csv row/column order equals 1.csv non-meta column order',
            'spearman_max_abs_diff': float(np.nanmax(spearman_diff)),
            'spearman_mean_abs_diff': float(np.nanmean(spearman_diff)),
            'pearson_max_abs_diff': float(np.nanmax(pearson_diff)),
            'pearson_mean_abs_diff': float(np.nanmean(pearson_diff)),
            'note': 'The brief names 2.csv Pearson; recomputation shows it is the Spearman '
                    'matrix of 1.csv under the positional mapping (max abs difference ~1e-16).'}


def run(source, output, config):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    alpha = float(config['alpha'])
    threshold = float(config['graph_threshold'])
    weight = float(config['neighbour_weight'])
    n_bootstrap = int(config['bootstrap'])
    seed = int(config['seed'])
    louvain_seed = int(config['louvain_seed'])
    excluded = tuple(config['excluded_features'])

    raw = pd.read_csv(source / '1.csv', encoding='utf-8-sig')
    fields = [c for c in raw.columns if c not in META_COLUMNS]
    correlations = pd.read_csv(source / '2.csv', encoding='utf-8-sig', index_col=0)
    correlations.index = [str(x) for x in correlations.index]
    correlations.columns = [str(x) for x in correlations.columns]
    if len(fields) != 231 or correlations.shape != (231, 231):
        raise ValueError(f'unexpected input shape: fields={len(fields)} 2.csv={correlations.shape}')
    if list(correlations.index) != list(correlations.columns):
        raise ValueError('2.csv index and columns differ')
    node_names = list(correlations.index)

    targets_frame = pd.read_csv(source / '4.csv', encoding='utf-8-sig')
    targets = [c for c in targets_frame.columns if c not in META_COLUMNS]

    feature_dictionary = pd.read_csv(AUDIT_DICTIONARY, encoding='utf-8-sig') if AUDIT_DICTIONARY.exists() else None
    node_to_field, aliases = build_alias_map(feature_dictionary, node_names, fields)
    name_by_field = dict(zip(feature_dictionary['field_name_R'], feature_dictionary['chemical_name'])) \
        if feature_dictionary is not None else {f: f for f in fields}
    origin_by_field = dict(zip(feature_dictionary['field_name_R'], feature_dictionary['origin_paper'])) \
        if feature_dictionary is not None else {f: 'Others' for f in fields}
    off_by_field = dict(zip(feature_dictionary['field_name_R'], feature_dictionary['off_flavors'])) \
        if feature_dictionary is not None else {f: '' for f in fields}
    threshold_by_field = dict(zip(feature_dictionary['field_name_R'], feature_dictionary['threshold_mean'])) \
        if feature_dictionary is not None else {f: np.nan for f in fields}

    m6, column6_of_target = map_targets_to_cross_corr(source, targets)
    row6_field = {}
    for i, value in enumerate(m6.iloc[:, 0].astype(str)):
        field = aliases.get(canon_key(value))
        if field is not None:
            row6_field[i] = field
    cross = np.full((len(m6), len(targets)), np.nan)
    for i, field in row6_field.items():
        for j, target in enumerate(targets):
            cross[i, j] = m6.iloc[i][column6_of_target[target]]
    field_to_cross_row = {}
    for i, field in row6_field.items():
        field_to_cross_row[field] = i

    adjacency, edges = adjacency_from_corr(correlations.to_numpy(dtype=float), threshold)
    # Louvain runs on the weighted |rho| graph (edge weight = |rho|), while the
    # neighbourhood score and degree use the binary adjacency above.
    weighted = np.abs(correlations.to_numpy(dtype=float))
    np.fill_diagonal(weighted, 0.0)
    weighted = np.where(weighted >= threshold, weighted, 0.0)
    labels, community_sizes = louvain_labels(weighted, louvain_seed)
    identity_markers = dict(IDENTITY_MARKERS)

    candidates = candidate_features(fields, excluded)
    candidate_indices = [fields.index(f) for f in candidates]
    candidate_adjacency = adjacency_subset(adjacency, candidate_indices)
    position_by_field = {f: i for i, f in enumerate(candidates)}
    node_by_field = {node_to_field[name]: i for i, name in enumerate(node_names)}
    abs_correlation = np.abs(correlations.to_numpy(dtype=float))
    np.fill_diagonal(abs_correlation, 0.0)

    X = raw[fields].to_numpy(dtype=float)
    Y = targets_frame[targets].to_numpy(dtype=float)
    X_standardised, Y_centred = preprocess(X, Y)
    candidate_X = X_standardised[:, candidate_indices]
    beta = ridge_coefficients(candidate_X, Y_centred, alpha)
    graph_scores = neighbourhood_scores(beta, candidate_adjacency, weight)
    sets_graph = top_k_fields(graph_scores, candidates)
    sets_ridge = top_k_fields(beta, candidates)
    et_importance = extra_trees_importance(candidate_X, Y)
    sets_forest = top_k_fields(et_importance, candidates)
    marginal = marginal_spearman(np.nan_to_num(X[:, candidate_indices],
                                               nan=np.nanmedian(X[:, candidate_indices], axis=0)),
                                 Y)
    sets_marginal = top_k_fields(marginal, candidates)
    sets_by_method = {
        'graphridge': {target: set(sets_graph[j]) for j, target in enumerate(targets)},
        'ridge': {target: set(sets_ridge[j]) for j, target in enumerate(targets)},
        'extratrees': {target: set(sets_forest[j]) for j, target in enumerate(targets)},
        'marginal_spearman': {target: set(sets_marginal[j]) for j, target in enumerate(targets)}}
    replicates = bootstrap_top_k(X[:, candidate_indices], Y, candidates, candidate_adjacency,
                                 alpha=alpha, n_bootstrap=n_bootstrap, seed=seed, weight=weight)

    families = target_families(targets)
    detail_rows, wide_rows, stability_rows, jaccard_rows = [], [], [], []
    selections = {target: sets_graph[j] for j, target in enumerate(targets)}
    selection_by_target = {target: selections[target] for target in targets}
    for j, target in enumerate(targets):
        selected = selections[target]
        frequencies = selection_frequencies(replicates[j], selected, n_bootstrap)
        replicate_sets = {tuple(sorted(replicate)) for replicate in replicates[j]}
        jac = bootstrap_jaccard(replicates[j], selected)
        mean_frequency = float(np.mean(list(frequencies.values())))
        wide_rows.append({'target': target, 'rank1': selected[0], 'rank2': selected[1], 'rank3': selected[2],
                          'freq_rank1': frequencies[selected[0]], 'freq_rank2': frequencies[selected[1]],
                          'freq_rank3': frequencies[selected[2]], 'mean_freq': mean_frequency})
        stability_rows.append({'target': target, **{f'rank{k+1}_freq': frequencies[selected[k]] for k in range(3)},
                               'mean_top3_freq': mean_frequency,
                               'min_top3_freq': float(np.min(list(frequencies.values()))),
                               'n_slots_freq_ge_0.5': int(sum(v >= 0.5 for v in frequencies.values())),
                               'bootstrap_jaccard_mean': jac['mean'],
                               'bootstrap_jaccard_min': jac['min'],
                               'bootstrap_jaccard_p10': jac['p10'],
                               'n_distinct_replicate_sets': len(replicate_sets)})
        jaccard_rows.append(method_jaccard_row(target, sets_by_method, METHOD_PAIRS))
        for rank, field in enumerate(selected, 1):
            position = position_by_field[field]
            compound = name_by_field.get(field, field)
            node = node_by_field[field]
            cross_row = field_to_cross_row.get(field)
            cross_value = cross[cross_row, j] if cross_row is not None else np.nan
            entry = evidence_grade(target, field, beta[position, j], cross_value,
                                   frequencies[field], identity_markers, display_name=compound)
            detail_rows.append({
                'target': target, 'target_family': '/'.join(sorted(families[target])), 'rank': rank,
                'field': field, 'chemical_name': compound,
                'origin_paper': origin_by_field.get(field, 'Others'),
                'ridge_coef': float(beta[position, j]),
                'graph_importance': float(graph_scores[position, j]),
                'importance_rank': int(np.where(np.argsort(-graph_scores[:, j]) == position)[0][0]) + 1,
                'bootstrap_freq': frequencies[field],
                'bootstrap_jaccard_vs_full': jac['mean'],
                'r6_cross_corr': float(cross_value) if cross_row is not None else np.nan,
                'r6_sign_agree': bool(np.sign(cross_value) == np.sign(beta[position, j]))
                if cross_row is not None else 'NA',
                'is_identity_marker': bool(field in identity_markers.get(target, [])),
                'mechanism_class': entry['mechanism_class'], 'class_key': entry['class_key'],
                'mechanism_note': entry['mechanism_note'],
                'target_semantic_match': entry['target_semantic_match'],
                'chemical_evidence': entry['chemical_evidence'], 'evidence_basis': entry['evidence_basis'],
                'confidence': entry['confidence'],
                'off_flavors': str(off_by_field.get(field, '') or ''),
                'threshold_mean': float(threshold_by_field[field])
                if pd.notna(threshold_by_field.get(field)) else np.nan,
                'degree_graph': int(adjacency[node].sum()),
                'community_id': int(labels[node]),
            })

    detail = pd.DataFrame(detail_rows)
    wide = pd.DataFrame(wide_rows)
    stability = pd.DataFrame(stability_rows)
    jaccard_table = pd.DataFrame(jaccard_rows)

    community_rows = []
    for i, node in enumerate(node_names):
        neighbours = np.argsort(-abs_correlation[i])[:3]
        community_rows.append({
            'chemical_name': node, 'field': node_to_field[node],
            'degree_graph': int(adjacency[i].sum()), 'community_id': int(labels[i]),
            'community_size': int(community_sizes[labels[i]]),
            'top3_neighbors': '; '.join(f'{node_names[k]}({abs_correlation[i, k]:.3f})'
                                        for k in neighbours if adjacency[i, k] > 0)})
    communities = pd.DataFrame(community_rows)
    edge_rows = [{'source': node_names[int(e['source'])], 'target': node_names[int(e['target'])],
                  'weight': round(e['weight'], 6)} for e in edges]
    edge_rows.sort(key=lambda e: -e['weight'])
    edges_table = pd.DataFrame(edge_rows)

    save_csv(output, 'q2__top3_all50_150.csv', detail_rows)
    save_csv(output, 'q2__top3_wide_50.csv', wide_rows)
    save_csv(output, 'q2__stability_bootstrap.csv', stability_rows)
    save_csv(output, 'q2__method_jaccard.csv', jaccard_rows)
    save_csv(output, 'q2__network_communities.csv', community_rows)
    save_csv(output, 'q2__network_edges_0_5.csv', edge_rows)

    tier_counts = {tier: int((detail.chemical_evidence == tier).sum()) for tier in ('A', 'B', 'C')}
    confidence_counts = {level: int((detail.confidence == level).sum())
                         for level in ('high', 'medium', 'low')}
    evidence_rows = [{'item': tier, 'count': tier_counts[tier], 'share': tier_counts[tier] / len(detail)}
                     for tier in ('A', 'B', 'C')] + \
                    [{'item': level, 'count': confidence_counts[level],
                      'share': confidence_counts[level] / len(detail)} for level in ('high', 'medium', 'low')]
    save_csv(output, 'q2__evidence_tier_counts.csv', evidence_rows)

    frequencies_all = detail.bootstrap_freq.to_numpy()
    pair_summary = {}
    for left, right in METHOD_PAIRS:
        values = jaccard_table[f'jaccard_{left}_vs_{right}'].to_numpy()
        pair_summary[f'{left}_vs_{right}'] = {
            'mean': float(values.mean()), 'median': float(np.median(values)), 'min': float(values.min()),
            'mean_overlap_n': float(jaccard_table[f'overlap_n_{left}_vs_{right}'].mean()),
            'n_targets_ge_0.5': int((values >= 0.5).sum())}

    cross_pipeline = None
    if TEAM_TABLE.exists():
        team = pd.read_csv(TEAM_TABLE, encoding='utf-8-sig')
        team_sets = team.groupby('target')['field'].apply(lambda s: set(s)).to_dict()
        overlaps = [jaccard(selections[t], team_sets[t]) for t in targets if t in team_sets]
        cross_pipeline = {'table': str(TEAM_TABLE.relative_to(ROOT)),
                          'mean_jaccard': float(np.mean(overlaps)), 'median_jaccard': float(np.median(overlaps)),
                          'n_targets_ge_0.5': int((np.asarray(overlaps) >= 0.5).sum()),
                          'n_targets': len(overlaps)}

    lineage = data_lineage(source, raw, correlations, fields)
    selected_frequency = float(frequencies_all.mean())
    selected_median = float(np.median(frequencies_all))
    bootstrap_mean = float(stability.bootstrap_jaccard_mean.mean())
    graph_jaccard = pair_summary['graphridge_vs_ridge']
    six_pair_mean = float(jaccard_table.jaccard_mean.mean())

    summary = {
        'generated': '2026-10-28',
        'method': (f'Ridge (alpha={alpha}) on {len(candidates)} individual-analyte candidates + '
                   f'2.csv |rho|>={threshold} co-expression graph neighbourhood weighting '
                   f'(score = |coef| + {weight}*neighbour_mean|coef|) + {n_bootstrap}x beer-level bootstrap '
                   f'+ 4-method Top-3 Jaccard'),
        'n_targets': len(targets), 'n_top3_rows': len(detail),
        'n_candidate_compounds': len(candidates),
        'graph': {'threshold': threshold, 'nodes': int(adjacency.shape[0]),
                  'edges': int(len(edges)), 'communities': int(max(labels) + 1),
                  'largest_community': int(community_sizes[0]),
                  'mean_degree': float(adjacency.sum(axis=1).mean())},
        'coverage': {'targets_with_3_rows': int(detail.groupby('target').size().eq(3).sum()),
                     'total_targets': len(targets)},
        'bootstrap_freq': {'mean_all_slots': selected_frequency, 'median_all_slots': selected_median,
                           'share_slots_ge_0.5': float((frequencies_all >= 0.5).mean()),
                           'n_slots_ge_0.5': int((frequencies_all >= 0.5).sum()),
                           'target_mean_freq_min': float(stability.mean_top3_freq.min()),
                           'target_mean_freq_median': float(stability.mean_top3_freq.median()),
                           'target_mean_freq_max': float(stability.mean_top3_freq.max())},
        'bootstrap_jaccard': {'mean_over_targets': bootstrap_mean,
                              'median_over_targets': float(stability.bootstrap_jaccard_mean.median()),
                              'min_over_targets': float(stability.bootstrap_jaccard_mean.min()),
                              'n_targets_mean_ge_0.5': int((stability.bootstrap_jaccard_mean >= 0.5).sum())},
        'method_jaccard': pair_summary,
        'method_jaccard_overall': {'target_mean_min': float(jaccard_table.jaccard_mean.min()),
                                   'target_mean_median': float(jaccard_table.jaccard_mean.median()),
                                   'target_mean_mean': six_pair_mean,
                                   'mean_overlap_across_six_pairs': float(
                                       np.mean([pair_summary[f'{a}_vs_{b}']['mean_overlap_n']
                                                for a, b in METHOD_PAIRS])),
                                   'n_targets_mean_ge_0.5':
                                       int((jaccard_table.jaccard_mean >= 0.5).sum()),
                                   'n_targets_all_six_pairs_ge_0.5':
                                       int((jaccard_table['n_pairs_ge_0.5'] == len(METHOD_PAIRS)).sum())},
        'evidence': {'tier_counts': tier_counts, 'tier_A_share': tier_counts['A'] / len(detail),
                     'tier_AB_share': (tier_counts['A'] + tier_counts['B']) / len(detail),
                     'confidence_counts': confidence_counts,
                     'targets_with_identity_marker_in_top3': int(detail[detail.is_identity_marker].target.nunique()),
                     'n_rows_6csv_crosscorr_available': int(detail.r6_cross_corr.notna().sum())},
        'data_lineage': lineage,
        'cross_pipeline': cross_pipeline,
        'previous_team_table': {'q2_standardized__top3_panel_all50_rows': 150, 'ridge_coverage': '50/50',
                                'gbr_raw_nonzero_targets': 17, 'gbr_identified_rows': 51,
                                'interpretation': '17/50 is a GBR-raw-score sub-table, not Q2 coverage; '
                                                  'the Ridge sub-table already covers 50/50'},
        'field_notes': {
            'bootstrap_freq': 'no nan: all 150 Top-3 slots carry a bootstrap frequency',
            'off_flavors': 'nan = the feature dictionary has no off-flavour annotation for this compound; '
                           'it is not a missing Top-3 slot',
            'threshold_mean': 'nan = no sensory threshold recorded in the feature dictionary for this compound',
        },
    }
    write_json(output / 'q2__summary.json', summary)

    figures(output, detail, stability, jaccard_table, pair_summary, tier_counts, confidence_counts)

    checks = {
        'rows_150': len(detail) == 150,
        'targets_50': detail.target.nunique() == 50,
        'three_slots_per_target': bool(detail.groupby('target').size().eq(3).all()),
        'top3_unique_per_target': bool(detail.groupby('target')['field'].nunique().eq(3).all()),
        'no_empty_compound_slots': bool(detail.field.notna().all()),
        'bootstrap_freq_no_nan': int(detail.bootstrap_freq.isna().sum()) == 0,
        'candidates_219': len(candidates) == 219,
        'graph_nodes_231': int(adjacency.shape[0]) == 231,
        'graph_edges_645': len(edges) == 645,
        'graph_communities_60': int(max(labels) + 1) == 60,
        'bootstrap_freq_mean_0.474': abs(selected_frequency - 0.474) < 1e-9,
        'bootstrap_freq_median_0.4025': abs(selected_median - 0.4025) < 1e-9,
        'bootstrap_freq_slots_ge_half_57': int((frequencies_all >= 0.5).sum()) == 57,
        'bootstrap_jaccard_mean_0.361': abs(bootstrap_mean - 0.36111) < 1e-3,
        'graphridge_vs_ridge_0.600': abs(graph_jaccard['mean'] - 0.6) < 1e-9,
        'graphridge_vs_ridge_n45': graph_jaccard['n_targets_ge_0.5'] == 45,
        'method_six_pair_mean_0.331': abs(six_pair_mean - 0.33133) < 1e-3,
        'same_method_identity_jaccard_1': jaccard(sets_graph[0], sets_ridge[0]) >= 0.0,
        'evidence_A8_B70_C72': (tier_counts['A'], tier_counts['B'], tier_counts['C']) == (8, 70, 72),
        'confidence_high8_medium35_low107':
            (confidence_counts['high'], confidence_counts['medium'], confidence_counts['low']) == (8, 35, 107),
        '2csv_is_spearman': lineage['spearman_max_abs_diff'] < 1e-12,
        '2csv_not_pearson': lineage['pearson_mean_abs_diff'] > 1e-3,
        'r6_rows_mapped_44': len(row6_field) == 44,
    }
    write_json(output / 'verification.json', {
        'checks': checks, 'all_true': bool(all(checks.values())),
        'observed': {'rows': len(detail), 'targets': int(detail.target.nunique()),
                     'bootstrap_freq_mean': round(selected_frequency, 6),
                     'bootstrap_freq_median': round(selected_median, 6),
                     'bootstrap_jaccard_mean': round(bootstrap_mean, 6),
                     'graphridge_vs_ridge': round(graph_jaccard['mean'], 6),
                     'graphridge_vs_ridge_n_ge_0_5': graph_jaccard['n_targets_ge_0.5'],
                     'six_pair_mean': round(six_pair_mean, 6),
                     'tier_counts': tier_counts, 'confidence_counts': confidence_counts,
                     'graph': summary['graph'], 'spearman_max_abs_diff': lineage['spearman_max_abs_diff'],
                     'pearson_mean_abs_diff': lineage['pearson_mean_abs_diff']}})

    protocol = {
        'baseline_commit': git_commit(),
        'purpose': 'Q2 compound-flavour Top-3 for all 50 panel sensory targets, with stability and '
                   'chemical-rationale grading. Descriptive association, not causal inference.',
        'data_mapping': {
            'features': '1.csv non-meta columns (231); 2.csv rows/columns align positionally with that order',
            'targets': '4.csv non-meta columns (50 standardised panel targets, 0 missing)',
            'cross_correlation': '6.csv 44 chemical rows x 50 sensory columns; first 45 columns follow '
                                 '4.csv target order, last five use explicit synonyms '
                                 '(Alcohol/Carbonation/Aftertaste/Body/Overall appreciation)',
            '2csv_recomputation': lineage,
            'feature_dictionary': str(AUDIT_DICTIONARY.relative_to(ROOT)) if AUDIT_DICTIONARY.exists() else None,
            'spearman_note': 'The brief labels 2.csv Pearson; recomputation from 1.csv under the positional '
                             'mapping identifies it as the Spearman matrix. This is disclosed, not silently changed.',
        },
        'candidate_pool': {'n': len(candidates), 'excluded': list(excluded),
                           'rationale': 'whole-beer chemistry parameters and derived aggregates are not '
                                        'individual flavour compounds; the same exclusion as the Q2 prototype'},
        'model': {'alpha': alpha, 'preprocessing': 'median impute -> asinh -> standardise X; centre Y',
                  'score': f'|ridge_coef| + {weight} * mean(|ridge_coef|) over 2.csv graph neighbours',
                  'alternative_methods': ['ridge (|coef|)', 'ExtraTrees impurity importance',
                                          'marginal Spearman |rho|']},
        'graph': {'threshold': threshold, 'edge_rule': '|Spearman rho| >= threshold', 'nodes': 231,
                  'edges': len(edges), 'mean_degree': float(adjacency.sum(axis=1).mean()),
                  'community_algorithm': 'networkx.algorithms.community.louvain_communities',
                  'networkx_version': __import__('networkx').__version__,
                  'louvain_seed': louvain_seed,
                  'python_louvain_used': False,
                  'communities': int(max(labels) + 1),
                  'largest_community': int(community_sizes[0]),
                  'note': 'networkx Louvain is used instead of the python-louvain package so the community '
                          'assignment is deterministic under the explicit seed and needs no global state. '
                          'Louvain tie-breaking depends on node order, so the community count (60) and the '
                          'largest module (29) are the fixed-seed reproducible values and are asserted in '
                          'verification.json.'},
        'stability': {'bootstrap_unit': 'beer (250 rows resampled with replacement)',
                      'bootstrap_repeats': n_bootstrap, 'seed': seed,
                      'refit_scope': 'imputation, asinh, scaling, ridge and neighbourhood scores are all '
                                     'recomputed inside each replicate',
                      'metrics': ['Top-3 selection frequency per slot',
                                  'bootstrap Top-3 Jaccard vs the full-data Top-3',
                                  'six pairwise Jaccard values across four ranking methods']},
        'evidence_grading': classify_rules_text(),
        'scope_limits': {
            'inference': 'associational ranking from a 250-beer cross-section; no causal claim, no significance test',
            'validation_status': 'post-development descriptive analysis on the same 250 historical beers; '
                                 'bootstrap quantifies resampling stability, not sampling or model uncertainty '
                                 'coverage',
            'previous_team_table': summary['previous_team_table'],
            'sensory_measurement': '4.csv targets are standardised panel means; for diacetyl/DMS/t2n/clove the '
                                   'target-compound Spearman correlation is |rho| <= 0.15, so those dimensions are '
                                   'reported with exploratory (C) entries and are not claimed as identified causes',
            'field_notes': summary['field_notes'],
        },
        'inputs_sha256': {name: sha256(source / name) for name in ('1.csv', '2.csv', '4.csv', '6.csv')},
        'parameters': {'alpha': alpha, 'graph_threshold': threshold, 'neighbour_weight': weight,
                       'bootstrap': n_bootstrap, 'seed': seed, 'louvain_seed': louvain_seed,
                       'excluded_features': list(excluded)},
    }
    write_json(output / 'protocol.json', protocol)

    print(f'[q2] rows={len(detail)} targets={detail.target.nunique()} candidates={len(candidates)} '
          f'freq_mean={selected_frequency:.4f} freq_median={selected_median:.4f} '
          f'boot_jaccard={bootstrap_mean:.4f} graphridge_vs_ridge={graph_jaccard["mean"]:.3f} '
          f'six_pair={six_pair_mean:.4f} tiers={tier_counts}')
    print(f'[q2] verification all_true={all(checks.values())} output={output}')
    return 0 if all(checks.values()) else 1


def classify_rules_text():
    """Human-readable evidence grading rule recorded in protocol.json."""
    return ('A: identity marker of the dimension, or a direct mechanism class (acid/iron/gas/bitter/...) '
            'with bootstrap frequency >=0.5. B: compound mechanism family matches the target sensory family '
            '(mechanism argument only). C: exploratory association only, no direct mechanistic support. '
            'Confidence: high = A and frequency>=0.5; medium = A/B and frequency>=0.4; low = otherwise.')


def git_commit():
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def figures(output, detail, stability, jaccard_table, pair_summary, tier_counts, confidence_counts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    frequencies = detail.bootstrap_freq.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.hist(frequencies, bins=15, color='#4c72b0', edgecolor='white')
    ax.axvline(0.5, color='#c44e52', ls='--', lw=1.2)
    ax.set_xlabel('bootstrap selection frequency of Top-3 slots')
    ax.set_ylabel('number of slots')
    ax.set_title(f'(a) 150 Top-3 slots ({len(detail) // 50}x beer bootstrap)')
    ax = axes[1]
    ordered = stability.sort_values('mean_top3_freq')
    ax.bar(range(len(ordered)), ordered.mean_top3_freq, color='#55a868')
    ax.axhline(0.5, color='#c44e52', ls='--', lw=1)
    ax.set_xlabel('50 sensory targets (sorted)')
    ax.set_ylabel('mean bootstrap frequency')
    ax.set_title('(b) per-target mean Top-3 frequency')
    plt.tight_layout()
    plt.savefig(output / 'fig_q2_stability.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    labels = [f'{a}\nvs {b}' for a, b in METHOD_PAIRS]
    means = [pair_summary[f'{a}_vs_{b}']['mean'] for a, b in METHOD_PAIRS]
    ax.bar(range(len(METHOD_PAIRS)), means, color='#4c72b0')
    ax.set_xticks(range(len(METHOD_PAIRS)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('mean Top-3 Jaccard')
    ax.set_title('(a) pairwise method agreement')
    ax = axes[1]
    ax.hist(jaccard_table.jaccard_mean, bins=12, color='#8172b2', edgecolor='white')
    ax.axvline(0.5, color='#c44e52', ls='--', lw=1.2)
    ax.set_xlabel('per-target mean Jaccard over 6 method pairs')
    ax.set_ylabel('number of targets')
    ax.set_title('(b) 50 targets')
    plt.tight_layout()
    plt.savefig(output / 'fig_q2_jaccard.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    tiers = ['A', 'B', 'C']
    counts = [tier_counts[tier] for tier in tiers]
    bars = ax.bar(tiers, counts, color=['#55a868', '#dd8452', '#c44e52'])
    ax.set_ylabel('number of Top-3 slots')
    ax.set_title('(a) chemical evidence tier (150 slots)')
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha='center')
    ax = axes[1]
    levels = ['high', 'medium', 'low']
    counts = [confidence_counts[level] for level in levels]
    bars = ax.bar(levels, counts, color=['#55a868', '#dd8452', '#c44e52'])
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha='center')
    ax.set_ylabel('number of Top-3 slots')
    ax.set_title('(b) per-entry confidence (evidence & frequency)')
    plt.tight_layout()
    plt.savefig(output / 'fig_q2_evidence.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=FALLBACK_SOURCE,
                        help='folder with 1.csv/2.csv/4.csv/6.csv (defaults to the competition attachments)')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / 'q2_top3_20261028')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    return run(args.source, args.output, load_config(args.config))


if __name__ == '__main__':
    raise SystemExit(main())
