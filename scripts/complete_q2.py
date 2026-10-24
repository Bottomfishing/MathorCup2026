"""Freeze descriptive Q2 Top-3 and join statistical, network and chemical evidence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import networkx as nx
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256, save_csv, write_json
from beer_audit.checks import repair_utf8_mojibake
from beer_models.baseline import load_data, make_pipeline
from beer_models.explain import top_indices
from beer_models.evidence import resolve_compound, annotate_relation


def run(source, reference, output):
    audit = Path('outputs/p0_audit_20261023_final')
    previous = Path('outputs/q2_exploratory_20261023_v1')
    permutation = Path('outputs/q2_permutation_20261024_v2')
    within = Path('outputs/q2_within_style_20261024_v3')
    review = Path('outputs/q2_review_20261024_v3')
    config_path = Path('configs/q2_evidence.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    manifest = json.loads((previous / 'run_manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['source_sha256'].items():
        assert sha256(source / name) == digest, name
    downloads = json.loads(Path('data/reference/zenodo10653704/download_manifest.json').read_text(encoding='utf-8'))
    expected = next(r['sha256'] for r in downloads if r['file'] == reference.name)
    assert sha256(reference) == expected
    ids, xf, yf, primary = load_data(source, audit)
    old = pd.read_csv(previous / 'top3_candidates.csv')
    frequency = pd.read_csv(previous / 'bootstrap_frequency.csv')
    selections = pd.read_csv(previous / 'bootstrap_selections.csv')
    assert len(selections) == 200 * len(yf.columns) * 3
    assert not selections.duplicated(['repeat', 'target', 'field']).any()
    assert selections.groupby(['repeat', 'target']).size().eq(3).all()
    assert set(selections['repeat']) == set(range(200))
    counts = selections.groupby(['target', 'field']).size()
    independently_counted = np.array([counts.get((r.target, r.field), 0) for r in frequency.itertuples()])
    np.testing.assert_array_equal(independently_counted, frequency['count'])
    np.testing.assert_allclose(frequency.frequency, independently_counted / 200, rtol=0, atol=1e-12)
    fdict = frequency.set_index(['target', 'field'])
    excluded = set(json.loads((previous / 'config.json').read_text(encoding='utf-8'))['exclude_from_compound_top3'])
    eligible = np.array([i for i, field in enumerate(xf) if field not in excluded])
    # One deterministic fit checks the frozen ranking, not another tuning or bootstrap run.
    model, _ = make_pipeline('ridge_asinh', {'ridge_alphas': [1000.]}, 20261023)
    model.set_params(model__alpha=1000.).fit(xf.to_numpy(float), yf.to_numpy(float))
    coefficients = model.named_steps['model'].coef_
    for j, target in enumerate(yf):
        selected = old.loc[old.target.eq(target)].sort_values('rank')
        expected_fields = xf.columns[top_indices(coefficients[j], eligible)].tolist()
        assert selected.field.tolist() == expected_fields, target
        np.testing.assert_allclose(selected.ridge_signed_coef, coefficients[j, [xf.columns.get_loc(f) for f in expected_fields]], atol=1e-12)
    meta = pd.read_csv(audit / 'feature_dictionary.csv').set_index('field_name_R')
    evidence = pd.read_excel(reference, sheet_name='Supplementary Table S1')
    evidence.columns = evidence.columns.str.strip(); evidence['excel_row'] = np.arange(len(evidence)) + 2
    ledger = []; resolved = {}
    for field in sorted(pd.read_csv(review / 'compound_evidence_ledger.csv').field):
        name = repair_utf8_mojibake(meta.loc[field, 'chemical_name'])
        hit, method = resolve_compound(field, name, evidence, config['aliases'])
        resolved[field] = hit
        ledger.append({'field': field, 'chemical_name': name, 'reference_compound': str(hit.Compound).strip(),
            'name_resolution': method, 'compound_class': str(hit['Compound class']).strip(), 'origin': str(hit.Origin).strip(),
            'flavor_description_present': pd.notna(hit.Flavor), 'source_excel_row': int(hit.excel_row),
            'source_url': 'https://zenodo.org/records/10653704', 'source_sheet': 'Supplementary Table S1',
            'source_sha256': expected, 'alias_basis': config['aliases'].get(field, {}).get('basis', 'Exact normalized reference name'),
            'alias_support_url': config['aliases'].get(field, {}).get('support_url', 'https://zenodo.org/records/10653704')})
    # Rebuild graph from the supplied correlation matrix, not from copied degree columns.
    network = pd.read_csv(source / '2.csv', index_col=0)
    aliases = {repair_utf8_mojibake(meta.loc[f, 'chemical_name']): f for f in xf}
    assert set(aliases) == set(network.columns)
    network = network.rename(index=aliases, columns=aliases).loc[xf.columns, xf.columns]
    a = network.to_numpy(float); np.testing.assert_allclose(a, a.T, atol=1e-12)
    graph = nx.Graph(); graph.add_nodes_from(xf.columns)
    i, j = np.where(np.triu(abs(a) >= .6, k=1))
    graph.add_edges_from((xf.columns[left], xf.columns[right]) for left, right in zip(i, j))
    component = {}
    for group in nx.connected_components(graph):
        for field in group:
            component[field] = len(group)
    old_degree = pd.read_csv(previous / 'network_nodes.csv').query('threshold == 0.6').set_index('field')
    for field in xf:
        assert graph.degree[field] == old_degree.loc[field, 'degree']
    global_imp = pd.read_csv(permutation / 'analytes_only/individual_importance.csv').set_index(['target', 'field'])
    within_imp = pd.read_csv(within / 'importance.csv').set_index(['target', 'field'])
    compare = pd.read_csv(review / 'method_comparison.csv').set_index('target')
    ridge_metrics = pd.read_csv('outputs/p1_robust_20261023_v2/metrics_by_target.csv')
    ridge_metrics = ridge_metrics.loc[ridge_metrics.model.eq('ridge_asinh')].set_index('target')
    result = []
    for record in old.to_dict('records'):
        target, field = record['target'], record['field']; hit = resolved[field]
        relation, matched, explanation = annotate_relation(target, field, hit.Flavor, hit.Origin, record['ridge_signed_coef'], config)
        gp, wp = global_imp.loc[(target, field)], within_imp.loc[(target, field)]
        frequency_value = fdict.loc[(target, field), 'frequency']
        np.testing.assert_allclose(record['bootstrap_frequency'], frequency_value, atol=1e-12)
        neighbors = sorted(graph.neighbors(field), key=lambda f: -abs(network.loc[field, f]))[:3]
        result.append({'target': target, 'primary': target in primary, 'rank': record['rank'], 'field': field,
            'chemical_name': record['chemical_name'], 'ridge_signed_coef': record['ridge_signed_coef'],
            'bootstrap_frequency': frequency_value, 'selected_in_at_least_half_refits': frequency_value >= .5,
            'marginal_spearman': record['marginal_spearman'], 'ridge_oof_target_r2': ridge_metrics.loc[target, 'r2'],
            'forest_global_delta_mae': gp.delta_mae, 'forest_within_style_delta_mae': wp.delta_mae,
            'forest_global_fixed_model_resample_frequency': gp.validation_resample_top3_frequency,
            'forest_within_style_fixed_model_resample_frequency': wp.validation_resample_top3_frequency,
            'network_degree_0_6': graph.degree[field], 'network_component_size_0_6': component[field],
            'network_neighbors_top3': ';'.join(neighbors),
            'network_neighbor_rho': ';'.join(str(network.loc[field, f]) for f in neighbors),
            'chemical_evidence': relation, 'matched_reference_terms': matched, 'chemical_rationale': explanation,
            'reference_excel_row': int(hit.excel_row), 'reference_source_url': 'https://zenodo.org/records/10653704',
            'causal_effect_established': False})
    result = pd.DataFrame(result)
    target_rows = []
    for target in yf:
        table = result.loc[result.target.eq(target)].sort_values('rank')
        agreement = compare.loc[target]
        target_rows.append({'target': target, 'primary': target in primary,
            'top3_fields': ';'.join(table.field), 'top3_names': ';'.join(table.chemical_name),
            'minimum_refit_frequency': table.bootstrap_frequency.min(), 'median_refit_frequency': table.bootstrap_frequency.median(),
            'n_with_reference_support': int(table.chemical_evidence.ne('target_specific_evidence_insufficient').sum()),
            'n_nonpositive_within_style_forest_importance': int((table.forest_within_style_delta_mae <= 0).sum()),
            **agreement.to_dict()})
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output, 'top3_all50.csv', result)
    save_csv(output, 'top3_primary45.csv', result.loc[result.primary])
    save_csv(output, 'target_summary.csv', target_rows)
    save_csv(output, 'compound_name_evidence.csv', ledger)
    save_csv(output, 'network_edges_0_6.csv', [{'source': xf.columns[left], 'target': xf.columns[right], 'rho': a[left, right]} for left, right in zip(i, j)])
    markers = []
    for target, fields in config['identity_markers'].items():
        for field in fields:
            markers.append({'target': target, 'field': field, 'chemical_name': repair_utf8_mojibake(meta.loc[field, 'chemical_name']),
                'in_frozen_top3': bool(((result.target == target) & (result.field == field)).any()),
                'role': 'named_target_or_acid_marker_for_interpretation_not_a_replacement_ranking'})
    save_csv(output, 'named_marker_crosscheck.csv', markers)
    summary = {'status': 'passed', 'selection_rule': config['selection_rule'],
        'rows_all50': len(result), 'rows_primary45': int(result.primary.sum()), 'resolved_compound_names': len(ledger),
        'explicit_aliases': len(config['aliases']), 'refit_frequency_median': float(result.bootstrap_frequency.median()),
        'n_below_half_refit_frequency': int((result.bootstrap_frequency < .5).sum()),
        'chemical_evidence_counts': result.chemical_evidence.value_counts().to_dict(),
        'n_negative_coefficients_with_reference_support': int(((result.ridge_signed_coef < 0) & result.chemical_evidence.ne('target_specific_evidence_insufficient')).sum()),
        'n_nonpositive_within_style_forest_importance': int((result.forest_within_style_delta_mae <= 0).sum()),
        'network_nodes': len(graph), 'network_edges_0_6': graph.number_of_edges(),
        'mean_jaccard': {c: float(compare[c].mean()) for c in compare if c.endswith('jaccard')},
        'source_sha256': manifest['source_sha256'], 'reference_sha256': expected, 'config_sha256': sha256(config_path),
        'checks': ['source and reference hashes', 'one-fit coefficient and frozen ranking reproduction',
                   '30000 bootstrap selections recounted', '82 unique exact or explicit name matches',
                   'supplied graph reconstructed and node degrees compared'],
        'scope': 'Descriptive full-data associations with target-specific evidence annotations. Descriptor matching is a documented manual-lexicon heuristic, not independent mechanistic validation. Refitting bootstrap and fixed-model validation bootstrap are distinct.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('data/reference/zenodo10653704/Supplemental Files and Figure source files.xlsx'))
    p.add_argument('--output', type=Path, default=Path('outputs/q2_delivery_20261024'))
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.reference, args.output)
