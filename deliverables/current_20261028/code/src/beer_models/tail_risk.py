"""Finite tail decisions and nested training-only original-rating risk calibration."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from .ordered_models import cumulative_probabilities
from .raw_panel import median_grades, enforce_hierarchy, rating_loss
from .transforms import RobustAsinh


def survival(probabilities):
    return 1 - np.cumsum(probabilities, axis=2)[:, :, :6]


def binary_loss(probability, observed_fraction):
    return observed_fraction * (1-probability)**2 + (1-observed_fraction) * probability**2


def soft_logit(x, fraction, C, max_iter=1000, tol=1e-6, monotone=False):
    """Weighted binary expansion exactly equals cross-entropy to fractional outcomes."""
    y = np.asarray(fraction, float)
    if not np.isfinite(y).all() or ((y < -1e-12) | (y > 1+1e-12)).any():
        raise ValueError('Expected observed binary fractions')
    y = np.clip(y, 0, 1)
    if y.max() == 0 or y.min() == 1 or np.ptp(x, axis=0).max() < 1e-12:
        return float(y.mean())
    xx = np.concatenate([x, x], axis=0)
    labels = np.r_[np.zeros(len(y), int), np.ones(len(y), int)]
    weight = np.r_[1-y, y]
    model = LogisticRegression(C=C, solver='lbfgs', max_iter=max_iter, tol=tol).fit(xx, labels, sample_weight=weight)
    if monotone and model.coef_[0, 0] < 0:
        return float(y.mean())
    return model


def binary_predict(model, x):
    return np.full(len(x), model) if isinstance(model, float) else model.predict_proba(x)[:, 1]


class RawOrdinalModel:
    def __init__(self, protocol):
        self.protocol = protocol

    def fit(self, x, distributions):
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
        z = self.imputer.transform(x)
        self.compress = RobustAsinh().fit(z)
        self.scale = StandardScaler().fit(self.compress.transform(z))
        z = self.scale.transform(self.compress.transform(z))
        response = survival(distributions).reshape(len(x), -1)
        self.n_targets = distributions.shape[1]
        self.models = Parallel(n_jobs=self.protocol['parallel_jobs'], prefer='threads')(
            delayed(soft_logit)(z, response[:, j], self.protocol['ordinal_C'],
                               self.protocol['ordinal_max_iter'], self.protocol['ordinal_tol'])
            for j in range(response.shape[1]))
        return self

    def predict_distribution(self, x):
        z = self.scale.transform(self.compress.transform(self.imputer.transform(x)))
        scores = np.column_stack([binary_predict(model, z) for model in self.models])
        return cumulative_probabilities(np.clip(scores.reshape(len(x), self.n_targets, 6), 0, 1))


def base_probabilities(model, x):
    if isinstance(model, RawOrdinalModel):
        return model.predict_distribution(x)
    z = model.imputer.transform(x)
    arrays = model.model.predict_proba(z)
    result = np.zeros((len(x), len(arrays), 7))
    for j, (array, classes) in enumerate(zip(arrays, model.model.classes_)):
        result[:, j, np.asarray(classes, int)] = array
    return result


def cost_decision(probabilities, weights, parents):
    weighted = np.asarray(probabilities) * np.asarray(weights)
    weighted /= weighted.sum(axis=2, keepdims=True)
    raw = median_grades(weighted)
    return enforce_hierarchy(raw, parents), raw


def tail_metrics(observed, grades):
    result = {'raw_rating_mae': float(rating_loss(observed, grades).mean())}
    empirical = median_grades(observed)
    for k in [5, 6]:
        mass = observed[:, :, k]
        result[f'raw_recall_{k}'] = float((mass * (grades == k)).sum() / mass.sum()) if mass.sum() else 0.0
        mask = empirical == k
        result[f'panel_recall_{k}'] = float((grades[mask] == k).mean()) if mask.any() else 0.0
    low_mass = observed[:, :, :5].sum(axis=2)
    result['raw_high_fpr'] = float((low_mass * (grades >= 5)).sum() / low_mass.sum())
    low = empirical < 5
    result['panel_high_fpr'] = float((grades[low] >= 5).mean())
    return result


def feasible(metrics, baseline_mae, limits):
    return bool(metrics['raw_rating_mae'] <= baseline_mae + limits['maximum_raw_mae_increase'] + 1e-12
        and all(metrics[f'raw_recall_{k}'] >= limits['minimum_each_raw_grade5_grade6_recall'] for k in [5, 6])
        and all(metrics[f'panel_recall_{k}'] >= limits['minimum_each_panel_median5_median6_recall'] for k in [5, 6])
        and metrics['raw_high_fpr'] <= limits['maximum_raw_high_false_positive_rate']
        and metrics['panel_high_fpr'] <= limits['maximum_panel_high_false_positive_rate'])


class SelectedRiskModel:
    def __init__(self, models, choices, calibrators, prior, protocol):
        self.models, self.choices, self.calibrators = models, choices, calibrators
        self.prior, self.protocol = prior, protocol

    def predict_uncalibrated_distribution(self, x):
        by_source = {name: base_probabilities(model, x) for name, model in self.models.items()}
        result = np.empty((len(x), len(self.choices), 7))
        for j, source in enumerate(self.choices):
            result[:, j] = self.prior[j] if source == 'prior' else by_source[source][:, j]
        return result

    def predict_distribution(self, x):
        by_source = {name: base_probabilities(model, x) for name, model in self.models.items()}
        result = np.empty((len(x), len(self.choices), 6))
        for j, source in enumerate(self.choices):
            if source == 'prior':
                result[:, j] = survival(self.prior[None])[:, j]
                continue
            scores = survival(by_source[source])[:, j]
            epsilon = self.protocol['calibration_epsilon']
            logits = np.log(np.clip(scores, epsilon, 1-epsilon) / np.clip(1-scores, epsilon, 1-epsilon))
            for k, calibrator in enumerate(self.calibrators[j]):
                result[:, j, k] = binary_predict(calibrator, logits[:, k:k+1])
        return cumulative_probabilities(np.clip(result, 0, 1))


def fit_risk(models, inner_probabilities, observed, protocol):
    names = ['prior', *protocol['base_models']]
    choices, calibrators, rows = [], [], []
    truth = survival(observed)
    for j in range(observed.shape[1]):
        losses = [float(binary_loss(survival(inner_probabilities[name])[:, j, 0], truth[:, j, 0]).mean()) for name in names]
        source = next(name for name, loss in zip(names, losses) if loss <= min(losses) + 1e-12)
        choices.append(source)
        for name, loss in zip(names, losses):
            rows.append(dict(target_index=j, source=name, inner_nonzero_brier=loss, selected=name == source))
        per_target = []
        scores = survival(inner_probabilities[source])[:, j]
        e = protocol['calibration_epsilon']
        logits = np.log(np.clip(scores, e, 1-e) / np.clip(1-scores, e, 1-e))
        for k in range(6):
            per_target.append(soft_logit(logits[:, k:k+1], truth[:, j, k], protocol['calibration_C'], monotone=True))
        calibrators.append(per_target)
    return SelectedRiskModel(models, choices, calibrators, observed.mean(axis=0), protocol), rows


class TailPointModel:
    def __init__(self, model, weights, parents):
        self.model, self.weights, self.parents = model, weights, parents
        self.imputer = model.imputer

    def predict(self, x):
        p = base_probabilities(self.model, x)
        grade, raw = cost_decision(p, self.weights, self.parents)
        return dict(grade=grade, unconstrained_grade=raw, raw_median_estimate=median_grades(p).astype(float))


def risk_frame(frame, bundle, beers):
    import pandas as pd
    from beer_audit.checks import numeric_frame
    x = numeric_frame(frame.reindex(columns=bundle['features'])).to_numpy(float)
    p = bundle['risk_panel_model'].predict_distribution(x)
    score = survival(p)
    uncalibrated = survival(bundle['risk_panel_model'].predict_uncalibrated_distribution(x))
    model = bundle['risk_panel_model']
    return pd.DataFrame({'beer': np.repeat(beers, len(bundle['targets'])),
        'target': np.tile(bundle['targets'], len(x)),
        'probability_rating_ge1': score[:, :, 0].ravel(),
        'probability_rating_ge5': score[:, :, 4].ravel(),
        'probability_rating_6': score[:, :, 5].ravel(),
        'uncalibrated_probability_rating_ge1': uncalibrated[:, :, 0].ravel(),
        'uncalibrated_probability_rating_ge5': uncalibrated[:, :, 4].ravel(),
        'uncalibrated_probability_rating_6': uncalibrated[:, :, 5].ravel(),
        'risk_source': np.tile(model.choices, len(x)),
        'risk_validation_status': np.tile(bundle['risk_status'], len(x)),
        'calibration_method': np.tile(['training_prior' if name == 'prior' else 'inner_OOF_sigmoid_then_order_projection' for name in model.choices], len(x)),
        'calibration_quality_status': 'fitted_method_not_assumed_to_improve_heldout_calibration',
        'safety_certified': False})
