"""Predict standardized sensory scores and training-reference relative grades."""
import numpy as np
import pandas as pd
from beer_audit.checks import numeric_frame
from .final_model import relative_grades
from .ordinal import grade_decisions, seven_level_probabilities, central_grade_interval


def predict_frame(frame, bundle):
    if frame.empty:
        raise ValueError('Input contains no samples')
    features, targets = bundle['features'], bundle['targets']
    if 'beer' in frame:
        beer = frame.beer.astype('string')
        if beer.isna().any() or beer.str.strip().eq('').any() or beer.duplicated().any():
            raise ValueError('beer must be nonempty and unique')
    else:
        beer = pd.Series([f'row_{i+1:06d}' for i in range(len(frame))], index=frame.index)
    # Missing columns and missing cells use the already fitted training imputer.
    values = numeric_frame(frame.reindex(columns=features)).to_numpy(float)
    observed = np.isfinite(values).sum(axis=1)
    if (observed == 0).any():
        raise ValueError('No observed model features for samples: ' + ', '.join(beer[observed == 0].astype(str)))
    prediction = bundle['model'].predict(values)
    has_distribution = 'grade_model' in bundle or 'selected_grade_model' in bundle
    if 'selected_grade_model' in bundle:
        probabilities = seven_level_probabilities(bundle['selected_grade_model'], values)
    elif 'grade_model' in bundle:
        grade_model = bundle['grade_model']
        imputed = grade_model.named_steps['impute'].transform(values)
        probabilities = seven_level_probabilities(grade_model.named_steps['model'], imputed)
    if has_distribution:
        grades = grade_decisions(probabilities)['conditional_median']
        lower, upper = central_grade_interval(probabilities)
    else:
        grades = relative_grades(bundle['reference_y'], prediction)
    long = pd.DataFrame({'beer': np.repeat(beer.to_numpy(), len(targets)),
        'target': np.tile(targets, len(frame)), 'primary': np.tile(bundle['primary'], len(frame)),
        'grade_method': bundle.get('grade_method', 'continuous_prediction_then_training_ECDF'),
        'standardized_prediction': prediction.ravel(), 'relative_grade_0_6': grades.ravel(),
        'observed_feature_count': np.repeat(observed, len(targets)),
        'missing_feature_fraction': np.repeat(1 - observed / len(features), len(targets))})
    if has_distribution:
        long['relative_grade_forecast_lower90'] = lower.ravel()
        long['relative_grade_forecast_upper90'] = upper.ravel()
        long['grade_max_class_probability'] = probabilities.max(axis=2).ravel()
        long['grade_probability_calibrated'] = False
    primary_targets = np.asarray(targets)[bundle['primary']].tolist()
    wide = long.pivot(index='beer', columns='target', values='relative_grade_0_6').loc[beer, primary_targets].reset_index()
    samples = pd.DataFrame({'beer': beer.to_numpy(), 'observed_feature_count': observed,
                            'missing_feature_fraction': 1 - observed / len(features)})
    return long, wide, samples
