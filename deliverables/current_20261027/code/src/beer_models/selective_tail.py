"""Selective exact-tail utility; probability learning remains unchanged."""
import numpy as np
from .raw_panel import median_grades, enforce_hierarchy, rating_loss
from .tail_risk import base_probabilities, tail_metrics, feasible


def selective_decision(probability, prior, mae_penalty, fp_penalty, parents):
    p, prior = np.asarray(probability, float), np.asarray(prior, float)
    baseline = enforce_hierarchy(median_grades(p), parents)
    loss0 = rating_loss(p, baseline)
    reward0 = np.zeros(baseline.shape)
    for grade in [5, 6]:
        if prior[grade] > 0:
            reward0 += .5*p[:, :, grade]/prior[grade]*(baseline == grade)
    choices = [baseline]
    utilities = [np.zeros(baseline.shape)]
    low = p[:, :, :5].sum(2)
    for grade in [5, 6]:
        action = np.full(baseline.shape, grade)
        reward = .5*p[:, :, grade]/prior[grade] if prior[grade] > 0 else np.full(baseline.shape, -np.inf)
        utilities.append(reward-reward0 - mae_penalty*(rating_loss(p, action)-loss0)
                         -fp_penalty*low*(1-(baseline >= 5)))
        choices.append(action)
    index = np.argmax(np.stack(utilities, axis=2), axis=2)
    grade = np.take_along_axis(np.stack(choices, axis=2), index[:, :, None], axis=2)[:, :, 0]
    return enforce_hierarchy(grade, parents)


def candidates(protocol):
    return [(lam, mu) for lam in protocol['mae_penalties'] for mu in protocol['false_positive_penalties']]


def select_policy(predictions, observed, baseline, protocol):
    base_mae = tail_metrics(observed, baseline)['raw_rating_mae']
    rows = [dict(candidate=i, mae_penalty=lam, fp_penalty=mu,
        **tail_metrics(observed, predictions[i])) for i, (lam, mu) in enumerate(candidates(protocol))]
    for row in rows:
        row['eligible'] = feasible(row, base_mae, protocol['point_limits'])
    eligible = [r for r in rows if r['eligible']]
    best = min(eligible, key=lambda r: (-(r['raw_recall_5']+r['raw_recall_6'])/2,
               r['raw_rating_mae'], r['candidate']))['candidate'] if eligible else None
    for row in rows:
        row['selected'] = row['candidate'] == best
    return best, rows


class SelectiveTailPoint:
    def __init__(self, model, prior, penalties, parents):
        self.model, self.prior, self.penalties, self.parents = model, prior, penalties, parents

    def predict(self, x):
        p = base_probabilities(self.model, x)
        median = enforce_hierarchy(median_grades(p), self.parents)
        grade = median if self.penalties is None else selective_decision(p, self.prior, *self.penalties, self.parents)
        return dict(grade=grade, raw_median_estimate=median.astype(float))
