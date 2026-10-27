"""Per-target cumulative binary models for seven training-reference grades."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .ordinal import grade_decisions, seven_level_probabilities
from .transforms import RobustAsinh


def cumulative_probabilities(survival):
    """Convert P(Y>k), k=0..5, into probabilities after an L2 order projection."""
    q = np.asarray(survival, float)
    if q.ndim != 3 or q.shape[-1] != 6 or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError('Expected finite sample-by-target-by-six probabilities')
    # Equal-weight decreasing isotonic projection:
    # fitted[i] = min_{a<=i} max_{b>=i} mean(q[a:b+1]).
    # There are only six cutoffs. Vectorizing across samples/targets avoids
    # invoking sklearn's validation and solver once per row in large grids.
    prefix = np.concatenate([np.zeros(q.shape[:2] + (1,)), q.cumsum(axis=2)], axis=2)
    fixed = np.empty_like(q)
    for i in range(6):
        value = np.full(q.shape[:2], np.inf)
        for a in range(i+1):
            maximum = np.full(q.shape[:2], -np.inf)
            for b in range(i, 6):
                maximum = np.maximum(maximum, (prefix[:, :, b+1]-prefix[:, :, a])/(b-a+1))
            value = np.minimum(value, maximum)
        fixed[:, :, i] = value
    # Roundoff in prefix differences can exceed the original [0,1] support.
    fixed = np.minimum.accumulate(np.clip(fixed, 0, 1), axis=2)
    endpoints = np.concatenate([np.ones(q.shape[:2] + (1,)), fixed,
                                np.zeros(q.shape[:2] + (1,))], axis=2)
    return endpoints[:, :, :-1] - endpoints[:, :, 1:]


class CumulativeOrdinalClassifier(BaseEstimator):
    def __init__(self, C=1., max_iter=1000, tol=1e-6):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if y.ndim != 2 or len(x) != len(y) or not np.isfinite(y).all() or (y != np.round(y)).any() or ((y < 0) | (y > 6)).any():
            raise ValueError('Expected matching integer seven-grade targets')
        self.classes_ = [np.arange(7) for _ in range(y.shape[1])]
        self.models_ = []
        self.n_features_in_ = x.shape[1]
        for target in range(y.shape[1]):
            thresholds = []
            for cut in range(6):
                binary = (y[:, target] > cut).astype(int)
                if np.all(binary == binary[0]):
                    thresholds.append(float(binary[0]))
                else:
                    thresholds.append(LogisticRegression(C=self.C, solver='lbfgs',
                        max_iter=self.max_iter, tol=self.tol).fit(x, binary))
            self.models_.append(thresholds)
        return self

    def predict_proba(self, x):
        x = np.asarray(x, float)
        q = np.empty((len(x), len(self.models_), 6))
        for target, models in enumerate(self.models_):
            for cut, model in enumerate(models):
                q[:, target, cut] = model if isinstance(model, float) else model.predict_proba(x)[:, 1]
        p = cumulative_probabilities(q)
        return [p[:, target, :] for target in range(p.shape[1])]

    def predict(self, x):
        return grade_decisions(seven_level_probabilities(self, x))['conditional_median']


def make_ordered_candidate(name, protocol, seed):
    impute = ('impute', SimpleImputer(strategy='median', keep_empty_features=True))
    if name == 'current_multioutput_tree':
        return Pipeline([impute, ('model', ExtraTreesClassifier(
            n_estimators=protocol['n_estimators'], min_samples_leaf=protocol['min_samples_leaf'],
            max_features=protocol['max_features'], random_state=seed, n_jobs=1))])
    index = protocol['candidates'].index(name) - 1
    if index < 0 or index >= len(protocol['logit_C']):
        raise ValueError('Unknown frozen candidate')
    return Pipeline([impute, ('compress', RobustAsinh()), ('scale', StandardScaler()),
        ('model', CumulativeOrdinalClassifier(C=protocol['logit_C'][index],
            max_iter=protocol['logit_max_iter'], tol=protocol['logit_tol']))])


def training_choices(predictions, observed):
    """Choose per target using inner-validation predictions; first method wins ties."""
    errors = abs(np.asarray(predictions) - np.asarray(observed)[None]).mean(axis=1)
    return np.argmin(errors, axis=0), errors


class SelectedGradeModel:
    """Frozen per-target choices; models receive raw features through fitted pipelines."""
    def __init__(self, models, choices):
        self.models = models
        self.choices = np.asarray(choices, int)
        self.classes_ = [np.arange(7) for _ in self.choices]

    def predict_proba(self, x):
        probabilities = [seven_level_probabilities(m, x) for m in self.models]
        return [probabilities[choice][:, target, :]
                for target, choice in enumerate(self.choices)]

    def predict(self, x):
        return grade_decisions(seven_level_probabilities(self, x))['conditional_median']
