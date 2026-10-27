"""Locally supported block transfer with explicit chemical/proxy evidence layers."""
import numpy as np
from scipy.spatial.distance import cdist
from .raw_panel import median_grades
from .tail_risk import base_probabilities, survival, binary_predict
from .ordered_models import cumulative_probabilities


def decode(parameters, source, neighbors, mutable, minimum_progress=.1):
    u = np.asarray(parameters, float)
    if u.ndim != 2 or u.shape[1] != len(neighbors)+1 or not np.isfinite(u).all() or ((u < 0)|(u > 1)).any():
        raise ValueError('Expected bounded progress and nonnegative local weights')
    weights = u[:, 1:].copy()
    totals = weights.sum(1, keepdims=True)
    weights = np.divide(weights, totals, out=np.full_like(weights, 1/len(neighbors)), where=totals > 0)
    progress = minimum_progress+(1-minimum_progress)*u[:, 0]
    result = np.broadcast_to(source, (len(u), len(source))).copy()
    endpoint = weights@np.asarray(neighbors)[:, mutable]
    result[:, mutable] += progress[:, None]*(endpoint-np.asarray(source)[mutable])
    return result, progress, weights


def structural_metrics(x, source, robust, reference_z, target_center, support_radius, classifiers, target_style):
    z, zs = robust.transform(x), robust.transform(np.asarray(source)[None])[0]
    target = np.sqrt(np.square(z-target_center).mean(1))
    initial = float(np.sqrt(np.square(zs-target_center).mean()))
    change = np.sqrt(np.square(z-zs).mean(1))
    nearest = cdist(z, reference_z).min(1)/np.sqrt(z.shape[1])
    style = {name: model.predict(x) == target_style for name, model in classifiers.items()}
    return dict(target_distance=target, source_target_distance=np.full(len(x), initial), change_distance=change,
        distance_improvement=1-target/max(initial, 1e-12), support_distance=nearest,
        support_pass=nearest <= support_radius+1e-10, **{name+'_style_pass': value for name, value in style.items()})


def defect_predictions(bundle, x, targets):
    """Reuse the existing six-cut projection; evaluate only needed ET-based risks."""
    risk = bundle['risk_panel_model']
    indices = [bundle['targets'].index(t) for t in targets]
    if any(risk.choices[j] != 'extra_trees' for j in indices):
        raise ValueError('Frozen Q3 protocol expects the five official ET risk sources')
    point = bundle['raw_panel_model']
    if getattr(point, 'penalties', 'unsupported') is not None:
        raise ValueError('Use the unchanged official median, not an experimental tail model')
    probabilities = base_probabilities(risk.models['extra_trees'], x)
    uncalibrated = probabilities[:, indices]
    scores = survival(uncalibrated)
    e = risk.protocol['calibration_epsilon']
    logits = np.log(np.clip(scores, e, 1-e)/np.clip(1-scores, e, 1-e))
    adjusted = np.empty_like(scores)
    for k, j in enumerate(indices):
        for cut, calibrator in enumerate(risk.calibrators[j]):
            adjusted[:, k, cut] = binary_predict(calibrator, logits[:, k, cut:cut+1])
    calibrated = cumulative_probabilities(np.clip(adjusted, 0, 1))
    return dict(grade=median_grades(probabilities)[:, indices], uncalibrated=uncalibrated, calibrated=calibrated)


def guard_metrics(before, after, tolerance=1e-10):
    delta_grade = after['grade']-before['grade']
    unc = survival(after['uncalibrated'])[:, :, [0, 4, 5]]-survival(before['uncalibrated'])[:, :, [0, 4, 5]]
    cal = survival(after['calibrated'])[:, :, [0, 4, 5]]-survival(before['calibrated'])[:, :, [0, 4, 5]]
    grade = (delta_grade <= 0).all(1)
    supported = (unc[:, :4, 0] <= tolerance).all(1)
    nonzero5 = (unc[:, :, 0] <= tolerance).all(1)
    both_nonzero = nonzero5&(cal[:, :, 0] <= tolerance).all(1)
    thirty = grade&(unc <= tolerance).all((1, 2))&(cal <= tolerance).all((1, 2))
    return dict(grade_pass=grade, supported_nonzero_pass=supported, five_nonzero_pass=nonzero5,
        both_nonzero_pass=both_nonzero, thirty_guard_pass=thirty,
        sensory_violation_count=(delta_grade > 0).sum(1)+(unc[:, :4, 0] > tolerance).sum(1),
        supported_positive_delta=np.maximum(unc[:, :4, 0], 0).sum(1)), dict(grade=delta_grade, uncalibrated=unc, calibrated=cal)


def rank_candidates(table):
    return np.lexsort((table['objective'], table['supported_positive_delta'],
                       table['sensory_violation_count'], table['structural_violation_count']))
