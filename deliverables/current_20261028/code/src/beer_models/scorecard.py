"""A complete scorecard view with explicit, different evidence for panel and proxy items."""
import numpy as np
import pandas as pd
from beer_audit.checks import numeric_frame
from .predict import predict_frame


def below_threshold_grade(log10_concentration, threshold):
    """Only infer zero under a stated detection-threshold assumption.

    A detection threshold does not identify any of intensity levels 1 through 6.
    Above-threshold point estimates therefore remain unavailable.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('A finite positive reference threshold is required')
    values = np.asarray(log10_concentration, float)
    if not np.isfinite(values).all():
        raise ValueError('Resolve missing chemical inputs using the fitted training imputer')
    return np.where(values < np.log10(threshold), 0., np.nan)


def extend_scorecard(panel, frame, bundle, prediction_role):
    """Extend 50 panel estimates with two threshold scenarios, never fake panel labels."""
    scope = pd.DataFrame(bundle['scorecard_scope']).set_index('target_id')
    protocol = bundle['scorecard_protocol']
    values = numeric_frame(frame.reindex(columns=bundle['features'])).to_numpy(float)
    beers = panel.beer.drop_duplicates().astype(str).to_numpy()
    if len(beers) != len(frame):
        raise ValueError('Panel rows and input samples do not align')
    input_beers = frame.beer.astype(str).to_numpy() if 'beer' in frame else np.array([f'row_{i+1:06d}' for i in range(len(frame))])
    if not np.array_equal(beers, input_beers):
        raise ValueError('Panel sample order differs from chemical inputs')
    chemical = bundle['features'].index(protocol['chemical_field'])
    supplied = values[:, chemical]
    imputed = ~np.isfinite(supplied)
    fitted_median = bundle['grade_model'].named_steps['impute'].statistics_[chemical]
    log_values = np.where(imputed, fitted_median, supplied)
    proxy_grades = below_threshold_grade(log_values, protocol['reference_threshold_mg_L'])

    result = panel.copy()
    result['panel_primary45'] = result.primary
    result['estimated_grade_0_6'] = result.relative_grade_0_6
    result['grade_kind'] = 'panel_relative_estimate'
    result['score_status'] = 'relative_grade_absolute_card_scale_unidentified'
    result['prediction_role'] = prediction_role
    result['original_card_score_recovered'] = False
    result['chemical_value_imputed'] = pd.NA
    result['chemical_log10_coordinate'] = np.nan
    result['reference_threshold_mg_L'] = np.nan
    result['panel_target_available'] = True
    extra = []
    for target in protocol['extra_targets']:
        extra.append(pd.DataFrame({'beer': beers, 'target': target, 'primary': True, 'panel_primary45': False,
            'estimated_grade_0_6': proxy_grades, 'relative_grade_0_6': np.nan,
            'standardized_prediction': np.nan, 'grade_method': 'below_reference_threshold_assumption',
            'grade_kind': 'chemical_threshold_proxy_no_panel_label',
            'score_status': np.where(np.isfinite(proxy_grades), 'below_reference_threshold_assumed_zero',
                                      'at_or_above_threshold_intensity_unidentified'),
            'prediction_role': 'mechanistic_scenario_not_panel_OOF', 'original_card_score_recovered': False,
            'panel_target_available': False, 'chemical_value_imputed': imputed,
            'chemical_log10_coordinate': log_values, 'reference_threshold_mg_L': protocol['reference_threshold_mg_L'],
            'observed_feature_count': np.isfinite(values).sum(axis=1),
            'missing_feature_fraction': 1 - np.isfinite(values).sum(axis=1) / len(bundle['features'])}))
    result = pd.concat([result, *extra], ignore_index=True)
    if set(result.target) != set(scope.index):
        raise ValueError('Scorecard coverage differs from the frozen scope')
    result['primary'] = result.target.map(scope.acceptance_scope.eq('main_content'))
    result['estimated_grade_0_6'] = result.estimated_grade_0_6.astype('Int64')
    sample_order = {beer: i for i, beer in enumerate(beers)}
    target_order = {target: i for i, target in enumerate(scope.index)}
    result = result.sort_values(['beer', 'target'], key=lambda col: col.map(sample_order if col.name == 'beer' else target_order)).reset_index(drop=True)
    wide = result.pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[beers, scope.index].reset_index()
    primary = wide[['beer', *scope.index[scope.acceptance_scope.eq('main_content')]]]
    return result, primary, wide


def predict_scorecard_frame(frame, bundle):
    panel, _, samples = predict_frame(frame, bundle)
    long, primary, all_items = extend_scorecard(panel, frame, bundle, 'full_training_model_inference_not_OOF')
    return long, primary, all_items, samples
