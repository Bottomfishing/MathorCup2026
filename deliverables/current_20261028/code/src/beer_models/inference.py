"""Q1 inference shared by the current CLI and the historical compatibility CLI."""
import joblib
import numpy as np
import pandas as pd
from beer_audit.audit import save_csv, write_json, sha256
from .predict import predict_frame
from .scorecard import predict_scorecard_frame


def _official_grade_frame(frame, bundle):
    """Apply the bundle's frozen Q1 decision rule to fresh inputs.

    Returns ``(grades, p_ge5, p6, legacy)`` where ``grades`` is the official
    integer matrix produced by :func:`beer_models.q1_decision.runtime_decision`
    -- the same function the official evaluation script uses.
    """
    from beer_audit.checks import numeric_frame
    from beer_models.q1_decision import runtime_decision
    from beer_models.tail_risk import base_probabilities
    decision = bundle['q1_decision']
    values = numeric_frame(frame.reindex(columns=bundle['features'])).to_numpy(float)
    model = bundle['raw_panel_model']
    legacy = np.asarray(model.predict(values)['grade'], dtype=float)
    probabilities = base_probabilities(model.model, values)
    p_ge5 = probabilities[:, :, 5:].sum(axis=2)
    p6 = probabilities[:, :, 6]
    pairs = decision['parent_child_pairs']
    grades = runtime_decision(
        p_ge5, p6, legacy,
        np.asarray(decision['rate_ge5'], dtype=float),
        np.asarray(decision['rate_6'], dtype=float),
        decision['targets'], [int(pair[0]) for pair in pairs],
        [int(pair[1]) for pair in pairs])
    return grades, p_ge5, p6, legacy


def _attach_official_decision(long, frame, bundle):
    """Add official top-k grades and probabilities while keeping legacy columns."""
    decision = bundle['q1_decision']
    grades, p_ge5, p6, legacy = _official_grade_frame(frame, bundle)
    targets = decision['targets']
    beer = (frame.beer.astype('string') if 'beer' in frame else
            pd.Series([f'row_{i+1:06d}' for i in range(len(frame))], index=frame.index))
    grid = pd.DataFrame({
        'beer_key': np.repeat(beer.astype(str).to_numpy(), len(targets)),
        'target': np.tile(targets, len(frame)),
        'grade_official': grades.ravel(),
        'grade_legacy': legacy.ravel(),
        'probability_rating_ge5': p_ge5.ravel(),
        'probability_rating_6': p6.ravel()})
    merged = long.assign(beer_key=long.beer.astype(str)).merge(
        grid, on=['beer_key', 'target'], how='left', validate='one_to_one', sort=False)
    scored = merged['panel_target_available'].astype(bool)
    if merged.loc[scored, ['grade_official', 'grade_legacy']].isna().all(axis=1).any():
        raise ValueError('official decision grid does not cover every scored cell')
    legacy_column = decision.get('legacy_grade_column', 'grade_legacy_integer_median')
    merged[legacy_column] = merged['grade_legacy']
    merged['estimated_grade_0_6'] = merged['grade_official']
    merged['estimated_grade_0_6'] = merged['estimated_grade_0_6'].astype('Int64')
    merged = merged.drop(columns=['beer_key', 'grade_official', 'grade_legacy'])
    scope = pd.DataFrame(bundle['scorecard_scope'])
    wide = merged.pivot(index='beer', columns='target', values='estimated_grade_0_6')
    wide = wide.loc[beer.astype(str).tolist(), scope['target_id'].astype(str).tolist()].reset_index()
    primary = wide[['beer', *scope.loc[scope.acceptance_scope.eq('main_content'), 'target_id']]]
    return merged, primary, wide


def run(model, input_path, output, experimental_tail=False):
    bundle = joblib.load(model)
    if experimental_tail:
        if 'experimental_point_model' not in bundle:
            raise ValueError('This bundle has no explicit experimental tail decision')
        bundle = dict(bundle)
        bundle.update(raw_panel_model=bundle['experimental_point_model'],
            selected_model='experimental_selective_tail_not_adopted',
            score_status='experimental_selective_tail_not_adopted',
            point_semantics='selective_exact_tail_utility_not_conditional_median',
            point_deployment_status='experimental_not_adopted_tail_stability_gate_failed',
            grade_method='q1_selective_tail_decision', tail_point_qualified_on_historical_folds=False)
    frame = pd.read_csv(input_path, dtype=str)
    if bundle.get('bundle_kind') in ['r2_raw_panel', 'r2b_raw_panel']:
        from beer_models.raw_panel import predict_raw_frame
        long, primary, wide, samples = predict_raw_frame(frame, bundle)
        if bundle['bundle_kind'] == 'r2b_raw_panel':
            from beer_models.tail_risk import risk_frame
            risks = risk_frame(frame, bundle, long.beer.drop_duplicates().to_numpy())
            long = long.merge(risks, on=['beer', 'target'], how='left', validate='one_to_one', sort=False)
            long['safety_certified'] = False
        if 'q1_decision' in bundle:
            long, primary, wide = _attach_official_decision(long, frame, bundle)
        output.mkdir(parents=True, exist_ok=False)
        save_csv(output, 'scores_all52_long.csv', long)
        save_csv(output, 'estimated_grades_primary47.csv', primary)
        save_csv(output, 'estimated_grades_all52.csv', wide)
        save_csv(output, 'sample_quality.csv', samples)
        info = {
            'samples': len(frame), 'targets': 52, 'labelled_panel_targets': len(bundle['targets']),
            'model': bundle['selected_model'], 'model_sha256': sha256(model), 'input_sha256': sha256(input_path),
            'score_scale': 'original_0_6_individual_rating_scale',
            'grade_semantics': bundle.get('point_semantics', 'integer_balanced_panel_conditional_median_with_parent_cap'),
            'grade_method': bundle.get('grade_method', 'r2_raw_panel_median'), 'grade_probability_calibrated': False,
            'missing_columns_imputed': [f for f in bundle['features'] if f not in frame],
            'unlabelled_card_targets': bundle['protocol']['unlabelled_card_targets'],
            'raw_source_sha256': bundle['raw_source_sha256'],
            'qualified_on_historical_grouped_folds': bundle['qualified_on_historical_grouped_folds'],
            'input_scale': 'same_supplied_coordinates_as_attachment_1_no_automatic_log_transform',
            'evaluation_use': 'full_training_model_predictions_are_not_OOF_accuracy_estimates'}
        if bundle['bundle_kind'] == 'r2b_raw_panel':
            info.update(tail_point_qualified=bundle['tail_point_qualified_on_historical_folds'],
                point_deployment_status=bundle['point_deployment_status'],
                risk_meaning=bundle['r2b_protocol']['risk_meaning'],
                risk_calibration_method='inner_OOF_sigmoid_then_order_projection_or_training_prior',
                risk_probabilities_universally_validated=False, safety_certified=False)
        if 'q1_decision' in bundle:
            decision = bundle['q1_decision']
            info.update(grade_method=bundle['grade_method'],
                grade_semantics=bundle['point_semantics'],
                official_grade_column=decision.get('official_grade_column', 'estimated_grade_0_6'),
                legacy_grade_column=decision.get('legacy_grade_column', 'grade_legacy_integer_median'),
                probability_columns=['probability_rating_ge5', 'probability_rating_6'],
                decision_rule=dict(rule=decision['rule'], forced=bool(decision['forced']),
                    middle_cap=int(decision['middle_cap']), rate_source=decision['rate_source'],
                    implementation=decision['implementation'],
                    implementation_sha256=decision['implementation_sha256'],
                    n_parent_child_pairs=len(decision['parent_child_pairs'])))
        if 'q1_target_protocol' in bundle:
            info.update(grade_method='q1_target_distribution_median',
                point_validation_status=bundle['q1_target_protocol']['validation_status'],
                risk_provenance=bundle['risk_provenance'],
                point_model_distribution_status='uncalibrated_comparison_only_separate_from_deployed_risk')
        if 'q1_selective_protocol' in bundle:
            info.update(grade_method=bundle['grade_method'],
                point_validation_status=bundle['q1_selective_protocol']['development_disclosure'],
                experimental_tail=bool(experimental_tail), official_point_replacement=False if experimental_tail else
                    bundle['tail_point_qualified_on_historical_folds'])
        write_json(output / 'prediction_info.json', info)
        print(f'Predicted {len(frame)} samples on raw 0-6 scale; output: {output}')
        return
    complete_card = 'scorecard_protocol' in bundle
    if complete_card:
        long, wide, all_items, samples = predict_scorecard_frame(frame, bundle)
    else:
        long, wide, samples = predict_frame(frame, bundle)
    output.mkdir(parents=True, exist_ok=False)
    if complete_card:
        save_csv(output, 'scores_all52_long.csv', long)
        save_csv(output, 'estimated_grades_primary47.csv', wide)
        save_csv(output, 'estimated_grades_all52.csv', all_items)
    else:
        save_csv(output, 'scores_all50_long.csv', long)
        save_csv(output, 'relative_grades_primary45.csv', wide)
    save_csv(output, 'sample_quality.csv', samples)
    info = {'samples': len(frame), 'targets': long.target.nunique(),
        'labelled_panel_targets': len(bundle['targets']), 'complete_scorecard_view': complete_card,
        'model': bundle['selected_model'], 'model_sha256': sha256(model), 'input_sha256': sha256(input_path),
        'missing_columns_imputed': [f for f in bundle['features'] if f not in frame],
        'ignored_input_columns': [f for f in frame if f not in bundle['features'] and f != 'beer'],
        'input_scale': 'same_supplied_coordinates_as_attachment_1_no_automatic_log_transform',
        'score_scale': 'provided_standardized_panel_score',
        'grade_semantics': ('per_row_panel_relative_or_threshold_scenario_not_raw_card_scores' if complete_card
                            else 'relative_midrank_ECDF_of_full_training_targets_not_raw_panel_rating'),
        'grade_method': bundle.get('grade_method', 'continuous_prediction_then_training_ECDF'),
        'grade_interval_kind': ('uncalibrated_equal_tail_forecast_interval' if 'grade_model' in bundle else 'not_available_for_continuous_only_bundle'),
        'grade_interval_nominal_level': .9 if 'grade_model' in bundle else None,
        'grade_probability_calibrated': False if 'grade_model' in bundle else None,
        'grade_interval_scope': 'panel_relative_targets_only_no_interval_for_unlabelled_proxies',
        'scorecard_protocol': bundle.get('scorecard_protocol'),
        'evaluation_use': 'full_training_model_predictions_are_not_OOF_accuracy_estimates'}
    if 'selected_grade_model' in bundle:
        info.update(task2_official_default=bundle['task2_official_default'],
            task2_qualified_on_historical_partition=bundle['task2_qualified'],
            task2_protocol=bundle['task2_protocol'])
    write_json(output / 'prediction_info.json', info)
    print(f'Predicted {len(frame)} samples; output: {output}')
