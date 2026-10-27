"""Sparse block designs and joint refit constraints; no brewing certification."""
import numpy as np
from .raw_panel import median_grades
from .tail_risk import base_probabilities


def sparse_decode(u, source, neighbors, mutable, group_ids, reducible, lower, n_groups=6):
    u = np.asarray(u, float)
    n_neighbors = len(neighbors)
    if u.ndim != 2 or u.shape[1] != n_neighbors+n_groups+len(reducible) or not np.isfinite(u).all() or ((u < 0)|(u > 1)).any():
        raise ValueError('Invalid sparse block parameters')
    weights = u[:, :n_neighbors]
    totals = weights.sum(1, keepdims=True)
    weights = np.divide(weights, totals, out=np.full_like(weights, 1/n_neighbors), where=totals > 0)
    endpoint = weights@np.asarray(neighbors)[:, mutable]
    progress = u[:, n_neighbors:n_neighbors+n_groups][:, np.asarray(group_ids)[mutable]]
    result = np.broadcast_to(source, (len(u), len(source))).copy()
    result[:, mutable] += progress*(endpoint-np.asarray(source)[mutable])
    result[:, reducible] = np.asarray(source)[reducible]-u[:, -len(reducible):]*(np.asarray(source)[reducible]-lower)
    return result


def panel_five(model, x, indices):
    p = base_probabilities(model, x)[:, indices]
    return dict(grade=median_grades(p), nonzero=1-p[:, :, 0])


def style_margins(classifiers, x, target='Tripel'):
    margins, passed = [], []
    for model in classifiers.values():
        probs = model.predict_proba(x)
        index = list(model.classes_).index(target)
        margins.append(probs[:, index]-np.delete(probs, index, axis=1).max(1))
        passed.append(model.classes_[probs.argmax(1)] == target)
    return np.stack(margins, axis=1), np.stack(passed, axis=1).all(1)


def refit_metrics(grade_delta, nonzero_delta, margins, style_pass, requirements, tolerance=1e-10):
    """Inputs are refit x candidate x target; joint events use the SAME refit."""
    grade_ok = (grade_delta <= 0).all(2)
    supported_ok = (nonzero_delta[:, :, :4] <= tolerance).all(2)
    joint = grade_ok&supported_ok&style_pass
    q = np.quantile(nonzero_delta[:, :, :4], requirements['probability_delta_quantile'], axis=0)
    mq = np.quantile(margins, requirements.get('style_margin_quantile', .2), axis=0)
    fraction = joint.mean(0)
    maximum = requirements['maximum_four_nonzero_delta_quantile']
    minimum = requirements.get('minimum_style_margin_quantile', -np.inf)
    ok = (fraction >= requirements['minimum_joint_refit_fraction'])&(q <= maximum+tolerance).all(1)&(mq >= minimum).all(1)
    return dict(joint_fraction=fraction, nonzero_upper=q, style_lower=mq, qualified=ok,
        grade_fraction=grade_ok.mean(0), supported_fraction=supported_ok.mean(0), style_fraction=style_pass.mean(0),
        five_nonzero_fraction=((nonzero_delta <= tolerance).all(2)&grade_ok&style_pass).mean(0))


def design_order(table):
    return np.lexsort((table['objective'], table['design_slack'], table['design_violation_count'],
                       table['nominal_sensory_violations'], table['structural_violations']))


def parameter_draws(source, selected, n, seed):
    mutable = np.flatnonzero(abs(selected-source) > 1e-10)
    rng = np.random.default_rng(seed)
    progress = np.clip(1+rng.normal(0, .01, (n, len(mutable))), 0, 1)
    result = np.broadcast_to(selected, (n, len(source))).copy()
    result[:, mutable] = source[mutable]+progress*(selected[mutable]-source[mutable])
    return result
