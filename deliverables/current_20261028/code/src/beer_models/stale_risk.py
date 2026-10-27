"""Focused genuine-label nonzero Stale risk; no intervention safety guarantee."""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .transforms import RobustAsinh
from .tail_risk import soft_logit, binary_predict


class StaleNonzeroModel:
    def __init__(self, method, aging_indices, seed):
        self.method, self.aging_indices, self.seed = method, np.asarray(aging_indices), seed

    def fit(self, x, fraction):
        self.prior = float(np.mean(fraction))
        self.indices = np.arange(x.shape[1]) if self.method == 'all_et_leaf10' else self.aging_indices
        if self.method == 'training_prior':
            return self
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x[:, self.indices])
        values = self.imputer.transform(x[:, self.indices])
        if self.method.startswith('age_logit'):
            self.robust = RobustAsinh().fit(values)
            self.scaler = StandardScaler().fit(self.robust.transform(values))
            values = self.scaler.transform(self.robust.transform(values))
            self.model = soft_logit(values, fraction, float(self.method.split('_C')[1]))
        elif self.method in ['age_et_leaf10', 'all_et_leaf10']:
            self.model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, max_features=.7,
                random_state=self.seed, n_jobs=1).fit(values, fraction)
        else:
            raise ValueError('Unknown Stale candidate')
        return self

    def predict(self, x):
        if self.method == 'training_prior': return np.full(len(x), self.prior)
        values = self.imputer.transform(x[:, self.indices])
        if self.method.startswith('age_logit'):
            values = self.scaler.transform(self.robust.transform(values))
            return binary_predict(self.model, values)
        return np.clip(self.model.predict(values), 0, 1)
