"""Q2 parameter-sensitivity package: graph threshold, Louvain resolution, ranking stability.

The delivered Q2 table (``outputs/q2_top3_20261028``) is not re-computed here: it
is read back, its Top-3 selections are reproduced from the default parameters, and
the same code path is then re-run over a pre-registered parameter grid:

* graph threshold ``|rho|`` in ``q2_sensitivity.THRESHOLD_GRID``;
* Louvain ``resolution`` in ``q2_sensitivity.RESOLUTION_GRID`` (plus a wider curve
  for the community-count figure).

Outputs go to ``outputs/q2_sensitivity_20261028``: sweep tables, per-dimension
overlap/stability tables, two figures, summary.json, verification.json and
protocol.json. The pre-registered stability criteria are defined once in
``src/beer_models/q2_sensitivity.py`` so this script, the tests and the protocol
cannot drift apart.

Run:  .venv\\Scripts\\python.exe scripts\\q2_sensitivity_20261028.py
      (or --source <attachment dir> --output <dir>)
AI-assisted: OpenAI Codex, 2026-10-28.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from beer_audit.audit import save_csv, sha256, write_json  # noqa: E402
from beer_models.q2_sensitivity import (BASELINE_VARIANT, BOUNDARY_THRESHOLDS,  # noqa: E402
                                        COMMUNITY_RANGE, MAX_LARGEST_SHARE, RESOLUTION_CURVE,
                                        RESOLUTION_GRID, STABLE_MIN_JACCARD, STABLE_MIN_TARGETS,
                                        THRESHOLD_GRID, community_partition, evidence_tier_counts,
                                        frequency_rank_correlation, frequency_table,
                                        overlap_metrics, rank_correlations, resolution_verdict,
                                        threshold_verdict, weighted_adjacency)
from beer_models.q2_top3 import (DEFAULT_ALPHA, DEFAULT_BOOTSTRAP, DEFAULT_EXTRA_TREES,  # noqa: E402
                                 DEFAULT_LOUVAIN_SEED, DEFAULT_NEIGHBOUR_WEIGHT, DEFAULT_SEED,
                                 IDENTITY_MARKERS, META_COLUMNS, adjacency_from_corr,
                                 adjacency_subset, bootstrap_top_k, canon_key,
                                 candidate_features, evidence_grade, louvain_labels,
                                 neighbourhood_scores, preprocess, ridge_coefficients,
                                 target_families, top_k_fields)

AUDIT_DICTIONARY = ROOT / 'outputs' / 'p0_audit_20261023_final' / 'feature_dictionary.csv'
MAIN_TABLE = ROOT / 'outputs' / 'q2_top3_20261028' / 'q2__top3_all50_150.csv'
FALLBACK_SOURCE = ROOT.parent / '2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题' / \
    '2026年MathorCup大数据挑战赛-赛道A初赛' / '附件'
DEFAULT_OUTPUT = ROOT / 'outputs' / 'q2_sensitivity_20261028'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Q2 parameter-sensitivity package')
    parser.add_argument('--source', default=str(FALLBACK_SOURCE), help='attachment directory (1.csv/2.csv/4.csv/6.csv)')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT), help='output directory')
    parser.add_argument('--bootstrap', type=int, default=DEFAULT_BOOTSTRAP, help='beer-level bootstrap repeats')
    return parser.parse_args(argv)


def map_targets_to_cross_corr(source, targets):
    """4.csv target order -> 6.csv column names; the last five names are explicit synonyms."""
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


def load_context(source):
    """Read the inputs and build the shared (parameter-independent) matrices."""
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
    node_to_field = {node_names[i]: fields[i] for i in range(len(fields))}

    targets_frame = pd.read_csv(source / '4.csv', encoding='utf-8-sig')
    targets = [c for c in targets_frame.columns if c not in META_COLUMNS]

    dictionary = pd.read_csv(AUDIT_DICTIONARY, encoding='utf-8-sig') if AUDIT_DICTIONARY.exists() else None
    aliases = {}
    if dictionary is not None:
        for row in dictionary.to_dict('records'):
            for value in (row.get('chemical_name'), row.get('chemical_name_short'), row.get('field_name_R')):
                if pd.notna(value):
                    aliases.setdefault(canon_key(value), row['field_name_R'])
    for node, field in node_to_field.items():
        aliases.setdefault(canon_key(node), field)
    name_by_field = dict(zip(dictionary['field_name_R'], dictionary['chemical_name'])) \
        if dictionary is not None else {field: field for field in fields}

    m6, column6_of_target = map_targets_to_cross_corr(source, targets)
    cross = np.full((len(m6), len(targets)), np.nan)
    field_to_cross_row = {}
    for i, value in enumerate(m6.iloc[:, 0].astype(str)):
        field = aliases.get(canon_key(value))
        if field is not None:
            field_to_cross_row[field] = i
            for j, target in enumerate(targets):
                cross[i, j] = m6.iloc[i][column6_of_target[target]]

    candidates = candidate_features(fields)
    candidate_indices = [fields.index(field) for field in candidates]
    X = raw[fields].to_numpy(dtype=float)
    Y = targets_frame[targets].to_numpy(dtype=float)
    return {'source': source, 'fields': fields, 'correlations': correlations, 'node_names': node_names,
            'node_to_field': node_to_field, 'targets': targets, 'candidates': candidates,
            'candidate_indices': candidate_indices, 'X': X, 'Y': Y, 'name_by_field': name_by_field,
            'cross': cross, 'field_to_cross_row': field_to_cross_row,
            'families': target_families(targets), 'identity_markers': dict(IDENTITY_MARKERS)}


def score_variant(context, threshold, resolution, n_bootstrap, seed=DEFAULT_SEED,
                  alpha=DEFAULT_ALPHA, weight=DEFAULT_NEIGHBOUR_WEIGHT):
    """Full chain for one (threshold, resolution) variant, including the bootstrap."""
    rho = context['correlations'].to_numpy(dtype=float)
    adjacency, edges = adjacency_from_corr(rho, threshold)
    weighted = weighted_adjacency(rho, threshold)
    labels, community_sizes = community_partition(weighted, resolution=resolution, seed=DEFAULT_LOUVAIN_SEED)
    candidate_indices = context['candidate_indices']
    candidates = context['candidates']
    candidate_adjacency = adjacency_subset(adjacency, candidate_indices)

    X_standardised, Y_centred = preprocess(context['X'], context['Y'])
    candidate_X = X_standardised[:, candidate_indices]
    beta = ridge_coefficients(candidate_X, Y_centred, alpha)
    scores = neighbourhood_scores(beta, candidate_adjacency, weight)
    selections = top_k_fields(scores, candidates)
    sets = {target: list(selections[j]) for j, target in enumerate(context['targets'])}

    replicates = bootstrap_top_k(candidate_X, context['Y'], candidates, candidate_adjacency,
                                 alpha=alpha, n_bootstrap=n_bootstrap, seed=seed, weight=weight)
    frequency = {target: frequency_table(replicates[j], candidates, n_bootstrap)
                 for j, target in enumerate(context['targets'])}
    return {'threshold': float(threshold), 'resolution': float(resolution),
            'adjacency': adjacency, 'edges': edges, 'weighted': weighted, 'labels': labels,
            'community_sizes': community_sizes, 'beta': beta, 'scores': scores,
            'sets': sets, 'replicates': replicates, 'frequency': frequency}


def evidence_rows(context, variant):
    """A/B/C grade for every Top-3 row of a variant (same grading as the main table)."""
    rows = []
    for j, target in enumerate(context['targets']):
        for rank, field in enumerate(variant['sets'][target], 1):
            position = context['candidates'].index(field)
            cross_row = context['field_to_cross_row'].get(field)
            cross_value = context['cross'][cross_row, j] if cross_row is not None else np.nan
            entry = evidence_grade(target, field, variant['beta'][position, j], cross_value,
                                   variant['frequency'][target][field], context['identity_markers'],
                                   display_name=context['name_by_field'].get(field, field))
            rows.append({'target': target, 'rank': rank, 'field': field,
                         'chemical_name': context['name_by_field'].get(field, field),
                         'graph_importance': float(variant['scores'][position, j]),
                         'bootstrap_freq': variant['frequency'][target][field],
                         'chemical_evidence': entry['chemical_evidence'],
                         'confidence': entry['confidence']})
    return rows


def figure_threshold(rows, output):
    """Figure 1: Top-3 Jaccard mean/median and 2/3-overlap count vs |rho| threshold."""
    thresholds = [row['threshold'] for row in rows]
    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.plot(thresholds, [row['jaccard_mean'] for row in rows], 'o-', label='mean Jaccard vs main table')
    axis.plot(thresholds, [row['jaccard_median'] for row in rows], 's--', label='median Jaccard')
    axis.axhline(STABLE_MIN_JACCARD, color='firebrick', linestyle=':', linewidth=1,
                 label=f'pre-registered mean >= {STABLE_MIN_JACCARD}')
    axis.set_xlabel('co-expression graph threshold |rho|')
    axis.set_ylabel('Top-3 Jaccard vs main table')
    axis.set_ylim(0, 1.05)
    axis.set_title('Q2 sensitivity: Top-3 stability across graph thresholds')
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / 'fig_q2_threshold_jaccard.png', dpi=160)
    plt.close(fig)


def figure_resolution(rows, output):
    """Figure 2: community count and largest-module share vs Louvain resolution."""
    resolutions = [row['resolution'] for row in rows]
    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.plot(resolutions, [row['n_communities'] for row in rows], 'o-', color='tab:blue',
              label='community count')
    low, high = COMMUNITY_RANGE
    axis.axhspan(low, high, color='tab:blue', alpha=0.08, label=f'community range {low}-{high}')
    axis.set_xlabel('Louvain resolution')
    axis.set_ylabel('number of communities')
    axis.grid(alpha=0.3)
    twin = axis.twinx()
    twin.plot(resolutions, [row['largest_share'] for row in rows], 's--', color='tab:red',
              label='largest community share')
    twin.axhline(MAX_LARGEST_SHARE, color='firebrick', linestyle=':', linewidth=1)
    twin.set_ylabel('largest community / nodes')
    twin.set_ylim(0, 0.6)
    lines = axis.get_lines() + twin.get_lines()[:1] + [twin.lines[-1]]
    labels = [line.get_label() for line in lines]
    axis.legend(lines, labels, fontsize=8, loc='upper left')
    axis.set_title('Q2 sensitivity: co-expression communities vs Louvain resolution')
    fig.tight_layout()
    fig.savefig(output / 'fig_q2_resolution_communities.png', dpi=160)
    plt.close(fig)


def write_report(path, summary, threshold_rows, resolution_rows):
    """Short human-readable verdict, so the package can be cited without the JSON."""
    verdict = summary['verdict']
    lines = ['# Q2 参数敏感性包（图阈值 / 社区分辨率 / 排名稳定性）', '',
             f"- 预注册阈值判据：{verdict['threshold_criterion']}",
             f"- 判定：阈值 {'稳定（不敏感）' if verdict['threshold_stable'] else '敏感'}；"
             f"分辨率 {'稳健' if verdict['resolution_stable'] else '不稳健'}",
             '',
             '## 阈值扫描（主表 = 0.50）', '',
             '| |rho| 阈值 | 边数 | 社区数 | 最大社区 | Jaccard 均值 | Jaccard 中位 | 重合≥2/3 维数 |',
             '|---|---|---|---|---|---|---|---|']
    for row in threshold_rows:
        lines.append(f"| {row['threshold']:.2f} | {row['n_edges']} | {row['n_communities']} | "
                     f"{row['largest_community']} | {row['jaccard_mean']:.3f} | "
                     f"{row['jaccard_median']:.3f} | {row['n_targets_overlap_ge_2']}/50 |")
    lines += ['', '## 分辨率扫描', '', '| resolution | 社区数 | 最大社区 | 最大社区占比 | 判据 |',
              '|---|---|---|---|---|']
    for row in resolution_rows:
        if row['resolution'] in RESOLUTION_CURVE:
            passed = '✓' if (COMMUNITY_RANGE[0] <= row['n_communities'] <= COMMUNITY_RANGE[1]
                             and row['largest_share'] < MAX_LARGEST_SHARE) else '✗'
            lines.append(f"| {row['resolution']:.1f} | {row['n_communities']} | "
                         f"{row['largest_community']} | {row['largest_share']:.1%} | {passed} |")
    lines += ['', '判据：社区数在 40–80 且最大社区 < 20% 节点。',
              '', '复现：`.venv\\Scripts\\python.exe scripts\\q2_sensitivity_20261028.py`', '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def run(source, output, n_bootstrap):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    (output / 'figures').mkdir(parents=True, exist_ok=True)
    context = load_context(source)
    targets = context['targets']
    candidates = context['candidates']
    rho = context['correlations'].to_numpy(dtype=float)

    baseline = score_variant(context, BASELINE_VARIANT['threshold'], BASELINE_VARIANT['resolution'],
                             n_bootstrap)

    # --- threshold sweep (Louvain resolution fixed at the baseline value) -------------
    threshold_rows, overlap_rows, top3_rows, tier_rows = [], [], [], []
    stability_by_target = {target: {'target': target} for target in targets}
    for threshold in THRESHOLD_GRID:
        variant = score_variant(context, threshold, BASELINE_VARIANT['resolution'], n_bootstrap)
        per_target, aggregate = overlap_metrics(baseline['sets'], variant['sets'])
        ranks = rank_correlations(baseline['scores'], variant['scores'],
                                  [baseline['sets'][target] for target in targets],
                                  [variant['sets'][target] for target in targets], targets)
        frequency_ranks = frequency_rank_correlation(baseline['frequency'], variant['frequency'],
                                                     targets, candidates)
        rows = evidence_rows(context, variant)
        tiers = evidence_tier_counts(rows)
        threshold_rows.append({'threshold': float(threshold), 'n_edges': len(variant['edges']),
                               'n_communities': int(max(variant['labels']) + 1),
                               'largest_community': int(variant['community_sizes'][0]),
                               'mean_degree': float(variant['adjacency'].sum(axis=1).mean()),
                               **aggregate,
                               'importance_spearman_mean': float(np.nanmean([r['importance_spearman'] for r in ranks])),
                               'kendall_tau_top3_mean': float(np.nanmean([r['kendall_tau_top3'] for r in ranks])),
                               'frequency_spearman_mean': float(np.nanmean([r['frequency_spearman'] for r in frequency_ranks])),
                               **{f'tier_{tier}': tiers['tiers'][tier] for tier in ('A', 'B', 'C')},
                               **{f'confidence_{level}': tiers['confidence'][level]
                                  for level in ('high', 'medium', 'low')}})
        for row, rank, freq_rank in zip(per_target, ranks, frequency_ranks):
            overlap_rows.append({'threshold': float(threshold), 'target': row['target'],
                                 'top3_jaccard': row['jaccard'], 'overlap_count': row['overlap'],
                                 'identical_set': row['identical'],
                                 'importance_spearman': rank['importance_spearman'],
                                 'kendall_tau_top3': rank['kendall_tau_top3'],
                                 'frequency_spearman': freq_rank['frequency_spearman']})
            stability_by_target[row['target']][f'jaccard_{threshold:.2f}'] = row['jaccard']
            stability_by_target[row['target']][f'overlap_{threshold:.2f}'] = row['overlap']
            stability_by_target[row['target']][f'importance_spearman_{threshold:.2f}'] = rank['importance_spearman']
            stability_by_target[row['target']][f'frequency_spearman_{threshold:.2f}'] = freq_rank['frequency_spearman']
        for target, fields_selected in variant['sets'].items():
            for rank, field in enumerate(fields_selected, 1):
                top3_rows.append({'threshold': float(threshold), 'target': target, 'rank': rank,
                                  'field': field,
                                  'chemical_name': context['name_by_field'].get(field, field)})
        tier_rows.append({'threshold': float(threshold),
                          **{f'tier_{tier}': tiers['tiers'][tier] for tier in ('A', 'B', 'C')},
                          **{f'confidence_{level}': tiers['confidence'][level]
                             for level in ('high', 'medium', 'low')}})

    # --- resolution sweep (threshold fixed at the baseline value) ---------------------
    baseline_weighted = weighted_adjacency(rho, BASELINE_VARIANT['threshold'])
    resolution_rows, top3_resolution_rows = [], []
    for resolution in RESOLUTION_CURVE:
        labels, community_sizes = community_partition(baseline_weighted, resolution=resolution,
                                                      seed=DEFAULT_LOUVAIN_SEED)
        n_nodes = int(baseline_weighted.shape[0])
        resolution_rows.append({'resolution': float(resolution),
                                'n_communities': int(len(community_sizes)),
                                'largest_community': int(community_sizes[0]),
                                'largest_share': float(community_sizes[0] / n_nodes),
                                'n_nodes': n_nodes,
                                'in_required_grid': bool(round(float(resolution), 4) in
                                                         {round(value, 4) for value in RESOLUTION_GRID}),
                                'passes_criterion': bool(COMMUNITY_RANGE[0] <= len(community_sizes) <= COMMUNITY_RANGE[1]
                                                         and community_sizes[0] / n_nodes < MAX_LARGEST_SHARE)})
        # the resolution only changes communities, not the ranking; record the selected sets
        for target, fields_selected in baseline['sets'].items():
            for rank, field in enumerate(fields_selected, 1):
                top3_resolution_rows.append({'resolution': float(resolution), 'target': target,
                                             'rank': rank, 'field': field})

    threshold_stability = threshold_verdict(threshold_rows)
    resolution_stability = resolution_verdict(resolution_rows)
    verdict = {'threshold_stable': threshold_stability['passed'],
               'threshold_criterion': threshold_stability['criterion'],
               'threshold_details': threshold_stability['details'],
               'resolution_stable': resolution_stability['passed'],
               'resolution_criterion': resolution_stability['criterion'],
               'resolution_details': resolution_stability['details'],
               'resolution_stable_range': resolution_stability['stable_resolution_range'],
               'headline': ('主结论对图阈值与社区分辨率均不敏感，Top-3 主表不是调参产物'
                            if threshold_stability['passed'] and resolution_stability['passed']
                            else '至少一项判据未通过，按预注册口径如实报告敏感项')}

    # --- baseline identity with the delivered main table ------------------------------
    identity = {'main_table': str(MAIN_TABLE), 'compared_rows': 0, 'mismatches': []}
    if MAIN_TABLE.exists():
        main = pd.read_csv(MAIN_TABLE, encoding='utf-8-sig')
        main_selection = {target: [] for target in targets}
        for row in main.sort_values(['target', 'rank']).itertuples():
            if row.target in main_selection:
                main_selection[row.target].append(str(row.field))
        compared = 0
        mismatches = []
        for target in targets:
            compared += len(main_selection[target])
            if list(main_selection[target]) != baseline['sets'][target]:
                mismatches.append(target)
        identity.update({'compared_rows': compared, 'mismatch_targets': mismatches,
                         'identical': not mismatches and compared == 150})

    # --- tables and figures ----------------------------------------------------------
    save_csv(output, 'q2__threshold_sweep.csv', threshold_rows)
    save_csv(output, 'q2__resolution_sweep.csv', resolution_rows)
    save_csv(output, 'q2__variant_overlap.csv', overlap_rows)
    save_csv(output, 'q2__stability_by_target.csv', list(stability_by_target.values()))
    save_csv(output, 'q2__top3_by_variant.csv', top3_rows)
    save_csv(output, 'q2__evidence_tiers.csv', tier_rows)
    save_csv(output, 'q2__top3_by_resolution.csv', top3_resolution_rows)
    figure_threshold(threshold_rows, output / 'figures')
    figure_resolution(resolution_rows, output / 'figures')

    checks = {
        'baseline_identical_to_main_table': bool(identity.get('identical', False)),
        'baseline_rows_150': identity.get('compared_rows') == 150,
        'threshold_grid_complete': [round(row['threshold'], 2) for row in threshold_rows] ==
                                   [round(value, 2) for value in THRESHOLD_GRID],
        'resolution_grid_complete': all(any(round(row['resolution'], 1) == round(value, 1)
                                            for row in resolution_rows) for value in RESOLUTION_GRID),
        'all_threshold_variants_150_rows': all(sum(row['n_targets'] for row in [threshold_rows[i]]) == 50
                                               for i in range(len(threshold_rows))) and
                                            all(row['n_targets'] == 50 for row in threshold_rows),
        'evidence_tiers_for_all_thresholds': len(tier_rows) == len(THRESHOLD_GRID),
        'figures_written': (output / 'figures' / 'fig_q2_threshold_jaccard.png').stat().st_size > 0 and
                           (output / 'figures' / 'fig_q2_resolution_communities.png').stat().st_size > 0,
        'verdict_recorded': 'threshold_stable' in verdict and 'resolution_stable' in verdict,
    }

    summary = {
        'generated': '2026-10-28',
        'purpose': 'pre-registered sensitivity evidence for the delivered Q2 50xTop-3 table',
        'main_table': str(MAIN_TABLE),
        'baseline_variant': BASELINE_VARIANT,
        'grids': {'thresholds': list(THRESHOLD_GRID), 'boundary_thresholds': list(BOUNDARY_THRESHOLDS),
                  'resolutions_required': list(RESOLUTION_GRID),
                  'resolutions_curve': list(RESOLUTION_CURVE)},
        'criteria': {'threshold': threshold_stability['criterion'],
                     'resolution': resolution_stability['criterion'],
                     'stable_min_jaccard': STABLE_MIN_JACCARD, 'stable_min_targets': STABLE_MIN_TARGETS,
                     'community_range': list(COMMUNITY_RANGE), 'max_largest_share': MAX_LARGEST_SHARE},
        'verdict': verdict,
        'baseline_identity': identity,
        'threshold_sweep': threshold_rows,
        'resolution_sweep': resolution_rows,
        'bootstrap': {'unit': 'beer', 'repeats': n_bootstrap, 'seed': DEFAULT_SEED,
                      'note': 'all threshold variants reuse the same seeded beer resamples, so the '
                              'comparison is paired and isolates the graph parameter'},
        'method': (f'Ridge (alpha={DEFAULT_ALPHA}) on {len(candidates)} candidates + graph neighbourhood '
                   f'weighting (weight={DEFAULT_NEIGHBOUR_WEIGHT}) with |rho| threshold and Louvain '
                   f'resolution varied; Louvain seed={DEFAULT_LOUVAIN_SEED}'),
    }
    write_json(output / 'summary.json', summary)

    protocol = {
        'baseline_commit': _git_commit(),
        'purpose': 'Q2 parameter-sensitivity evidence package; the delivered main table is not modified',
        'pre_registered_criteria': {
            'threshold': threshold_stability['criterion'],
            'resolution': resolution_stability['criterion'],
            'note': 'criteria were fixed before the sweep and live in src/beer_models/q2_sensitivity.py'},
        'grids': summary['grids'],
        'parameters': {'alpha': DEFAULT_ALPHA, 'neighbour_weight': DEFAULT_NEIGHBOUR_WEIGHT,
                       'bootstrap': n_bootstrap, 'seed': DEFAULT_SEED, 'louvain_seed': DEFAULT_LOUVAIN_SEED,
                       'extra_trees_params_unused': DEFAULT_EXTRA_TREES},
        'baseline_reproduction': {'main_table': str(MAIN_TABLE),
                                  'check': 'default parameters (threshold 0.50, resolution 1.0) reproduce '
                                           'the main Q2 Top-3 selection row by row',
                                  'result': identity},
        'data_mapping': {'2csv_is': 'spearman',
                         'positional_order': '2.csv row/column order equals 1.csv non-meta column order',
                         'note': 'the brief names 2.csv Pearson; recomputation of 1.csv gives the Spearman '
                                 'matrix (max abs difference ~1e-16)'},
        'graph': {'weighted_edge_rule': '|rho| >= threshold (edge weight = |rho|)',
                  'community_algorithm': 'networkx.algorithms.community.louvain_communities',
                  'resolution_parameter': 'varied over the sweep, fixed at 1.0 in the main table',
                  'seed': DEFAULT_LOUVAIN_SEED},
        'scope_limits': {'inference': 'parameter sensitivity of an associational ranking; no causal claim',
                         'main_table_untouched': 'q2_top3.py defaults and outputs/q2_top3_20261028 are '
                                                 'not modified by this package'},
        'inputs_sha256': {name: sha256(source / name) for name in ('1.csv', '2.csv', '4.csv', '6.csv')},
        'reproduction': '.venv\\Scripts\\python.exe scripts\\q2_sensitivity_20261028.py',
    }
    write_json(output / 'protocol.json', protocol)

    verification = {'checks': checks, 'all_pass': all(checks.values()),
                    'observed': {'baseline_identity': identity,
                                 'threshold_jaccard_mean': {f'{row["threshold"]:.2f}': row['jaccard_mean']
                                                            for row in threshold_rows},
                                 'resolution_communities': {f'{row["resolution"]:.1f}': row['n_communities']
                                                            for row in resolution_rows}}}
    write_json(output / 'verification.json', verification)
    write_report(output / 'q2__sensitivity_report.md', summary, threshold_rows, resolution_rows)

    print(f'[q2-sensitivity] baseline_identical={identity.get("identical", False)} '
          f'thresholds={[round(row["threshold"], 2) for row in threshold_rows]} '
          f'jaccard_mean={[round(row["jaccard_mean"], 3) for row in threshold_rows]} '
          f'communities={[row["n_communities"] for row in resolution_rows]}')
    print(f'[q2-sensitivity] threshold_stable={threshold_stability["passed"]} '
          f'resolution_stable={resolution_stability["passed"]} all_pass={all(checks.values())} '
          f'output={output}')
    return 0 if all(checks.values()) else 1


def _git_commit():
    """Short HEAD hash when available, else 'unknown'."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=str(ROOT),
                                capture_output=True, text=True, check=False)
        return result.stdout.strip() or 'unknown'
    except OSError:
        return 'unknown'


def main(argv=None):
    args = parse_args(argv)
    return run(args.source, args.output, args.bootstrap)


if __name__ == '__main__':
    sys.exit(main())
