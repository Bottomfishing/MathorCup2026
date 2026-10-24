"""Check selection scope and within-style evidence without tuning another model."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_audit.checks import repair_utf8_mojibake
from beer_models.baseline import load_data, make_pipeline
from beer_models.explain import top_indices, stratified_draw
from beer_models.reliability import permute_within_groups, monte_carlo_upper_p, benjamini_yekutieli


def run(source, output):
    config_path = Path('configs/q2_reliability_protocol.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    original_config = json.loads(Path('configs/q2_exploratory_v1.json').read_text(encoding='utf-8'))
    previous = Path('outputs/q2_exploratory_20261023_v1')
    audit = Path('outputs/p0_audit_20261023_final')
    ids, xf, yf, primary = load_data(source, audit)
    x, y = xf.to_numpy(float), yf.to_numpy(float); styles = ids.tasting_category_fine.to_numpy()
    excluded = set(original_config['exclude_from_compound_top3'])
    strict = np.array([i for i, c in enumerate(xf) if c not in excluded])
    family_index = xf.columns.get_loc(config['family_indicator'])
    expanded = np.array([i for i, c in enumerate(xf) if c not in excluded or i == family_index])
    settings = {'ridge_alphas': [config['ridge_alpha']]}
    model, _ = make_pipeline('ridge_asinh', settings, config['seed'])
    model.set_params(model__alpha=config['ridge_alpha']).fit(x, y)
    coefficients = model.named_steps['model'].coef_
    z = model[:-1].transform(x); z -= z.mean(axis=0); centered_y = y - y.mean(axis=0)
    # X and alpha are fixed under target permutations: factorize the Ridge system once.
    projection = cho_solve(cho_factor(z.T @ z + config['ridge_alpha'] * np.eye(z.shape[1])), z.T)
    np.testing.assert_allclose((projection @ centered_y).T, coefficients, rtol=0, atol=1e-12)
    null_strict = np.empty((config['within_style_permutations'], len(yf.columns)))
    null_expanded = np.empty_like(null_strict); rng = np.random.default_rng(config['seed'])
    for repeat in range(config['within_style_permutations']):
        permutation = permute_within_groups(styles, rng)
        assert np.array_equal(styles[permutation], styles)
        permuted = abs((projection @ centered_y[permutation]).T)
        null_strict[repeat] = permuted[:, strict].max(axis=1)
        null_expanded[repeat] = permuted[:, expanded].max(axis=1)
        if (repeat + 1) % 500 == 0:
            print(f'Within-style null {repeat + 1}/{len(null_strict)}', flush=True)
    observed_max = abs(coefficients[:, strict]).max(axis=1)
    target_p = monte_carlo_upper_p(observed_max, null_strict)
    target_q = benjamini_yekutieli(target_p)
    family_target_p = monte_carlo_upper_p(abs(coefficients[:, expanded]).max(axis=1), null_expanded)
    family_target_q = benjamini_yekutieli(family_target_p)
    metadata = pd.read_csv(audit / 'feature_dictionary.csv').set_index('field_name_R')
    old_top = pd.read_csv('outputs/q2_delivery_20261024/top3_all50.csv')
    old_selections = pd.read_csv(previous / 'bootstrap_selections.csv')
    selection_lookup = {(repeat, target): g.sort_values('rank').field.tolist()
                        for (repeat, target), g in old_selections.groupby(['repeat', 'target'])}
    counts = np.zeros_like(coefficients, int); sign_counts = np.zeros_like(coefficients, int)
    rng = np.random.default_rng(config['bootstrap_seed'])
    for repeat in range(config['bootstrap_repeats']):
        draw = stratified_draw(styles, rng)
        boot, _ = make_pipeline('ridge_asinh', settings, config['bootstrap_seed'])
        boot.set_params(model__alpha=config['ridge_alpha']).fit(x[draw], y[draw])
        c = boot.named_steps['model'].coef_
        sign_counts += np.sign(c) == np.sign(coefficients)
        for j, target in enumerate(yf):
            assert xf.columns[top_indices(c[j], strict)].tolist() == selection_lookup[repeat, target]
            counts[j, top_indices(c[j], expanded)] += 1
    original_rows = []; family_rows = []; summaries = []
    for j, target in enumerate(yf):
        old = old_top.loc[old_top.target.eq(target)].sort_values('rank')
        chosen = top_indices(coefficients[j], expanded)
        original_fields = set(old.field); new_fields = set(xf.columns[chosen])
        summaries.append({'target': target, 'primary': target in primary,
            'within_style_max_statistic_p': target_p[j], 'across_50_targets_BY_q': target_q[j],
            'passes_BY_0_05': target_q[j] <= .05, 'strict_family_top3_jaccard': len(original_fields & new_fields) / len(original_fields | new_fields),
            'family_within_style_max_statistic_p': family_target_p[j],
            'family_across_50_targets_BY_q': family_target_q[j], 'family_passes_BY_0_05': family_target_q[j] <= .05,
            'chemical_family_indicator_selected': family_index in chosen,
            'ridge_oof_r2': old.ridge_oof_target_r2.iloc[0]})
        for row in old.to_dict('records'):
            k = xf.columns.get_loc(row['field'])
            row['sign_agreement_all_refits'] = sign_counts[j, k] / config['bootstrap_repeats']
            row['within_style_selection_max_p'] = float(monte_carlo_upper_p(abs(coefficients[j, k]), null_strict[:, j]))
            row['target_BY_q'] = target_q[j]
            row['chemical_scope'] = 'original_individual_analyte_candidate_scope'
            original_rows.append(row)
        for rank, k in enumerate(chosen):
            field = xf.columns[k]
            family_rows.append({'target': target, 'primary': target in primary, 'rank': rank + 1, 'field': field,
                'chemical_name': repair_utf8_mojibake(metadata.loc[field, 'chemical_name']), 'reference_unit': metadata.loc[field, 'unit'],
                'ridge_signed_coef': coefficients[j, k], 'bootstrap_frequency': counts[j, k] / config['bootstrap_repeats'],
                'sign_agreement_all_refits': sign_counts[j, k] / config['bootstrap_repeats'],
                'within_style_selection_max_p': float(monte_carlo_upper_p(abs(coefficients[j, k]), null_expanded[:, j])),
                'target_family_scope_BY_q': family_target_q[j],
                'chemical_scope': 'chemical_family_IBU_indicator' if k == family_index else 'individual_analyte_candidate',
                'source_interpretation': ('Iso-alpha acids labelled by source; IBU assay does not identify one molecule or certify exact concentration.' if k == family_index else 'Same individual-analyte scope as original ranking')})
    original_rows = pd.DataFrame(original_rows); family_rows = pd.DataFrame(family_rows); summaries = pd.DataFrame(summaries)
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output, 'original_top3_reliability.csv', original_rows)
    save_csv(output, 'family_scope_top3.csv', family_rows)
    save_csv(output, 'target_diagnostics.csv', summaries)
    save_csv(output, 'null_max_strict.csv', pd.DataFrame(null_strict, columns=yf.columns).assign(repeat=np.arange(len(null_strict))))
    save_csv(output, 'null_max_family.csv', pd.DataFrame(null_expanded, columns=yf.columns).assign(repeat=np.arange(len(null_expanded))))
    markers = pd.read_csv('outputs/q2_delivery_20261024/named_marker_crosscheck.csv')
    marker_rows = []
    for row in markers.itertuples():
        j, k = yf.columns.get_loc(row.target), xf.columns.get_loc(row.field)
        marker_rows.append({'target': row.target, 'field': row.field, 'chemical_name': row.chemical_name,
            'absolute_coefficient_rank_in_strict_scope': int(np.flatnonzero(top_indices(coefficients[j], strict, len(strict)) == k)[0] + 1),
            'signed_coefficient': coefficients[j, k],
            'within_style_selection_max_p': float(monte_carlo_upper_p(abs(coefficients[j, k]), null_strict[:, j])),
            'interpretation': 'Named marker is not forced into a statistical Top-3.'})
    save_csv(output, 'named_marker_diagnostics.csv', marker_rows)
    summary = {'status': 'passed', 'config': config, 'config_sha256': sha256(config_path),
        'source_sha256': {n: sha256(source / n) for n in ['1.csv', '4.csv', '8.xlsx']},
        'strict_candidates': len(strict), 'family_scope_candidates': len(expanded),
        'targets_with_family_indicator_in_top3': summaries.loc[summaries.chemical_family_indicator_selected, 'target'].tolist(),
        'n_targets_BY_0_05': int(summaries.passes_BY_0_05.sum()),
        'targets_BY_0_05': summaries.loc[summaries.passes_BY_0_05, 'target'].tolist(),
        'n_family_scope_targets_BY_0_05': int(summaries.family_passes_BY_0_05.sum()),
        'family_scope_targets_BY_0_05': summaries.loc[summaries.family_passes_BY_0_05, 'target'].tolist(),
        'original_top3_candidates_max_p_le_0_05': int(original_rows.within_style_selection_max_p.le(.05).sum()),
        'bootstrap_strict_selections_reproduced': len(old_selections),
        'checks': ['closed-form coefficient projection equals fitted Ridge', 'all permutation indices stay within style',
                   'all 30000 original bootstrap selections reproduced before applying expanded scope'],
        'interpretation': 'Permutation evidence is conditional on within-style exchangeability; neither small p-values, sign consistency nor family indicators prove direct sensory or causal effects.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('outputs/q2_reliability_20261024'))
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.output)
