"""Select the validated Q1 decision and deliver all 52 scorecard items with provenance."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from beer_audit.audit import save_csv, sha256, write_json
from beer_models.baseline import mask_observed
from beer_models.final_model import relative_grades
from beer_models.ordinal import central_grade_interval, grade_decisions
from beer_models.scorecard import below_threshold_grade, extend_scorecard
from complete_modeling import load_q1_artifacts


def run(source, q1, ordinal, output):
    config_path = Path('configs/q1_task2_protocol.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    old_summary = json.loads((ordinal / 'summary.json').read_text(encoding='utf-8'))
    ids, xf, yf, base, splits = load_q1_artifacts(source, Path('outputs/p0_audit_20261023_final'), q1)
    assert sha256(ordinal / 'model.joblib') == old_summary['model_sha256']
    assert sha256(q1 / 'model.joblib') == old_summary['base_continuous_model_sha256']
    assert old_summary['source_sha256'] == {n: sha256(source / n) for n in ['1.csv', '4.csv', '8.xlsx']}
    bundle = joblib.load(ordinal / 'model.joblib')
    assert bundle['grade_method'] == config['main_panel_grade_method']
    assert bundle['features'] == list(xf) and bundle['targets'] == list(yf)
    scope_path = Path(config['scorecard_scope']); scope = pd.read_csv(scope_path).fillna('')
    assert len(scope) == 52 and scope.acceptance_scope.eq('main_content').sum() == 47
    assert set(scope.loc[scope.panel_target_available, 'target_id']) == set(yf)
    thresholds = pd.read_excel(source / '8.xlsx', sheet_name=config['threshold_sheet'])
    hit = thresholds.loc[thresholds.field_name_R.str.strip().eq(config['chemical_field'])]
    assert len(hit) == 1 and hit.unit.iloc[0] == config['threshold_unit']
    threshold = float(hit[config['threshold_statistic']].iloc[0])
    protocol = dict(config, reference_threshold_mg_L=threshold, reference_excel_row=int(hit.index[0])+2,
        aroma_flavour_thresholds_separately_calibrated=False,
        original_absolute_score_calibration_identified=False)
    bundle = dict(bundle, scorecard_scope=scope.to_dict('records'), scorecard_protocol=protocol)
    x, y = xf.to_numpy(float), yf.to_numpy(float); beers = ids.beer.astype(str).to_numpy()
    old = pd.read_csv(ordinal / 'oof_predictions.csv', dtype={'beer': str})
    assert len(old) == 12500 and not old.duplicated(['beer', 'target']).any()
    def matrix(field):
        return old.pivot(index='beer', columns='target', values=field).loc[beers, yf.columns].to_numpy()
    probabilities = np.stack([matrix(f'probability_grade_{k}') for k in range(7)], axis=2)
    decisions = grade_decisions(probabilities); truth = matrix('observed_relative_grade')
    recorded_median = matrix('median_relative_grade').astype(int)
    # CSV round-tripping can move a CDF of exactly .5 by one floating-point ULP.
    # Preserve the recorded action only after verifying it still minimizes loss.
    risk = np.sum(probabilities[:, :, :, None] * abs(np.arange(7)[:, None]-np.arange(7)[None, :]), axis=2)
    recorded_risk = np.take_along_axis(risk, recorded_median[:, :, None], axis=2)[:, :, 0]
    assert np.max(recorded_risk-risk.min(axis=2)) <= 1e-12
    median_boundary_ties = int((decisions['conditional_median'] != recorded_median).sum())
    decisions['conditional_median'] = recorded_median
    np.testing.assert_array_equal(decisions['conditional_mode'], matrix('mode_relative_grade'))
    naive = np.empty_like(truth)
    for train, test in splits:
        np.testing.assert_array_equal(relative_grades(y[train], y[test]), truth[test])
        naive[test] = np.quantile(relative_grades(y[train], y[train]), .5, axis=0, method='inverted_cdf')
    predictions = {'training_median': naive, 'continuous_then_grade': matrix('baseline_relative_grade'), **decisions}
    parents = []
    for row in scope.loc[scope.parent_formula_cell.ne('')].itertuples():
        parent = scope.loc[scope.template_score_formula_cell.eq(row.parent_formula_cell), 'target_id'].iloc[0]
        parents.append((list(yf).index(parent), list(yf).index(row.target_id)))
    clipped = decisions['conditional_median'].copy()
    for parent, child in parents:
        clipped[:, child] = np.minimum(clipped[:, child], clipped[:, parent])
    predictions['conditional_median_child_clipped_diagnostic'] = clipped
    metrics = []
    for name, values in predictions.items():
        for label, mask in [('panel_primary45', base['primary']), ('panel_all50', np.ones(50, bool))]:
            error = abs(values[:, mask]-truth[:, mask])
            metrics.append({'method': name, 'scope': label, 'n_labelled_scores': int(error.size),
                'grade_mae': error.mean(), 'exact_accuracy': (error == 0).mean(), 'within_one_accuracy': (error <= 1).mean(),
                'original_panel_score_recovered': False})
    hierarchy = []
    for name, values in [('constructed_observed_relative_grades', truth), *predictions.items()]:
        violation = np.column_stack([values[:, child] > values[:, parent] for parent, child in parents])
        hierarchy.append({'method': name, 'n_child_parent_pairs': int(violation.size),
            'child_above_parent_count': int(violation.sum()), 'fraction': violation.mean(),
            'interpretation': 'Separately ranked marginals; this is not proof of an impossible raw individual scorecard'})
    old_cont = pd.read_csv(q1 / 'oof_predictions.csv', dtype={'beer': str})
    continuous = old_cont.loc[old_cont.model.eq('nested_selected')].pivot(index='beer', columns='target', values='prediction').loc[beers, yf.columns].to_numpy()
    lower, upper = central_grade_interval(probabilities)
    panel = pd.DataFrame({'beer': np.repeat(beers, 50), 'target': np.tile(yf.columns, len(x)),
        'primary': np.tile(base['primary'], len(x)), 'grade_method': config['main_panel_grade_method'],
        'relative_grade_0_6': decisions['conditional_median'].ravel(), 'observed_relative_grade': truth.ravel(),
        'standardized_prediction': continuous.ravel(), 'observed_feature_count': np.repeat(np.isfinite(x).sum(axis=1), 50),
        'missing_feature_fraction': np.repeat(1-np.isfinite(x).mean(axis=1), 50),
        'relative_grade_forecast_lower90': lower.ravel(), 'relative_grade_forecast_upper90': upper.ravel(),
        'grade_max_class_probability': probabilities.max(axis=2).ravel(), 'grade_probability_calibrated': False})
    frame = pd.concat([ids[['beer']].reset_index(drop=True), xf.reset_index(drop=True)], axis=1)
    long, primary, all_items = extend_scorecard(panel, frame, bundle, 'outer_fold_estimate_not_full_fit')
    # Check the chemical-coordinate assumption independently against attachment 8.
    cal = pd.read_excel(source / '8.xlsx', sheet_name='Calibrated chemical dataset').set_index('beer')
    cal.index = cal.index.astype(str); ref = cal.loc[beers, config['chemical_field']].to_numpy(float)
    chemical = list(xf).index(config['chemical_field']); measured_log = x[:, chemical]
    paired = np.isfinite(measured_log) & (ref > 0)
    np.testing.assert_allclose(measured_log[paired], np.log10(ref[paired]), rtol=0, atol=1e-6)
    concentration = np.power(10., measured_log)
    sensitivity = []
    for multiplier in config['threshold_sensitivity_multipliers']:
        t = threshold * multiplier; values = below_threshold_grade(measured_log, t)
        sensitivity.append({'threshold_multiplier': multiplier, 'threshold_mg_L': t,
            'below_threshold_zero_count': int(np.isfinite(values).sum()),
            'at_or_above_threshold_intensity_unidentified_count': int(np.isnan(values).sum()),
            'source_samples': len(x), 'scope': 'declared_threshold_scenario_not_a_confidence_interval_or_human_accuracy'})
    reviews = pd.read_csv(source / '7.csv', dtype={'beer': str}).set_index('beer').loc[beers]
    auxiliary = []
    for target, field in [('A_acetaldehyde', 'aroma_acetaldehyde'), ('F_acetaldehyde', 'taste_acetaldehyde')]:
        r = reviews[field]; floor = r.mode().iloc[0]
        auxiliary.append({'target': target, 'forum_field': field, 'nonmissing': int(r.notna().sum()),
            'modal_value_count': int(r.eq(floor).sum()), 'nonmodal_value_count': int(r.ne(floor).sum()),
            'spearman_with_acetaldehyde_coordinate': pd.Series(r.to_numpy()).corr(pd.Series(measured_log), method='spearman'),
            'treated_as_panel_label': False})
    original_missing = pd.read_csv(ordinal / 'missingness.csv')
    selected_missing = original_missing.loc[original_missing.method.eq('conditional_median')].copy()
    selected_missing['evaluation_scope'] = '45_panel_relative_targets_only_no_proxy_ground_truth'
    stress = []
    old_config = old_summary['config']
    for rate in old_config['missing_rates']:
        for repeat in range(old_config['missing_repeats']):
            total_missing = 0; changed = 0
            for fold, (train, test) in enumerate(splits):
                masked, _ = mask_observed(x[test], rate, old_config['seed']+1000+fold*100+repeat)
                missing = ~np.isfinite(masked[:, chemical]); total_missing += int(missing.sum())
                log_value = np.where(missing, np.nanmedian(x[train, chemical]), masked[:, chemical])
                after = below_threshold_grade(log_value, threshold)
                changed += int(np.sum(after != below_threshold_grade(measured_log[test], threshold)))
            stress.append({'additional_missing_rate': rate, 'repeat': repeat, 'source_samples': len(x),
                'chemical_inputs_imputed_training_fold_only': total_missing, 'proxy_rule_changed_count': changed,
                'interpretation': 'Rule consistency only; no panel label exists to measure accuracy'})
    score_semantics = scope.copy()
    score_semantics['grade_method'] = np.where(score_semantics.panel_target_available,
        config['main_panel_grade_method'], 'below_reference_threshold_assumption')
    score_semantics['absolute_card_score_identified'] = False
    output.mkdir(parents=True, exist_ok=False)
    for name, data in [('scores_all52_long', long), ('estimated_grades_primary47', primary),
                       ('estimated_grades_all52', all_items), ('score_semantics', score_semantics),
                       ('model_comparison', metrics), ('hierarchy_diagnostics', hierarchy),
                       ('threshold_sensitivity', sensitivity), ('auxiliary_evidence', auxiliary),
                       ('selected_missingness', selected_missing), ('acetaldehyde_missingness', stress)]:
        save_csv(output, name+'.csv', data)
    save_csv(output, 'selected_metrics_by_target.csv', pd.read_csv(ordinal / 'metrics_by_target.csv').query("method == 'conditional_median'"))
    save_csv(output, 'selected_metrics_by_observed_grade.csv', pd.read_csv(ordinal / 'metrics_by_observed_grade.csv').query("method == 'conditional_median'"))
    joblib.dump(bundle, output / 'model.joblib', compress=3)
    write_json(output / 'summary.json', {'status': 'completed', 'config': config, 'config_sha256': sha256(config_path),
        'scope_sha256': sha256(scope_path), 'source_sha256': {n: sha256(source / n) for n in ['1.csv', '3.xlsx', '4.csv', '7.csv', '8.xlsx']},
        'parent_q1_model_sha256': sha256(q1 / 'model.joblib'), 'parent_ordinal_model_sha256': sha256(ordinal / 'model.joblib'),
        'parent_ordinal_oof_sha256': sha256(ordinal / 'oof_predictions.csv'), 'model_sha256': sha256(output / 'model.joblib'),
        'score_items': 52, 'primary_items': 47, 'panel_targets': 50, 'labelled_primary_targets': 45,
        'extra_targets': config['extra_targets'], 'reference_threshold_mg_L': threshold,
        'reference_threshold_excel_row': protocol['reference_excel_row'],
        'acetaldehyde_concentration_range_mg_L': [float(concentration.min()), float(concentration.max())],
        'acetaldehyde_positive_calibration_pairs': int(paired.sum()),
        'acetaldehyde_zero_reference_beers': beers[ref == 0].tolist(),
        'main_complete_numeric_rows': int(long.estimated_grade_0_6.notna().sum()), 'rows': len(long),
        'new_predictive_models_trained': 0, 'original_absolute_card_scale_identified': False,
        'csv_median_boundary_ties_preserving_verified_recorded_action': median_boundary_ties,
        'main_panel_selection': 'Conditional median: lowest pre-existing primary45 OOF MAE; clipping worsens this metric and does not recover raw-card semantics.',
        'limitations': ['Panel grades remain relative estimates, not original 0-6 intensity recovery.',
            'The two zero grades require the reference threshold to apply to both modalities; no expert labels or separate modality calibration exists.',
            'A threshold alone identifies no above-threshold 1-6 intensity; such future inputs receive no invented point grade.',
            'Do not include 500 proxy estimates in panel accuracy metrics. Historical OOF folds informed exploration.']})
    print(pd.DataFrame(metrics).query("scope == 'panel_primary45'").to_string(index=False))
    print(f'Delivered {len(long)} estimates, 47 primary and 5 supplementary items; absolute card scale remains unidentified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--q1', type=Path, default=Path('outputs/final_20261024/q1'))
    parser.add_argument('--ordinal', type=Path, default=Path('outputs/q1_ordinal_delivery_20261024'))
    parser.add_argument('--output', type=Path, default=Path('outputs/q1_task2_20261025'))
    args = parser.parse_args()
    run(args.source, args.q1, args.ordinal, args.output)
