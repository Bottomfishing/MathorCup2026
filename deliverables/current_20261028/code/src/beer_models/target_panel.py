"""Target-specific original-rating distributions and inner-only median selection."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from .raw_panel import median_grades, enforce_hierarchy, rating_loss
from .tail_risk import base_probabilities


class TargetDistributionForest:
    def __init__(self, leaf, protocol, seed):
        self.leaf, self.protocol, self.seed = leaf, protocol, seed

    def fit(self, x, distribution):
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
        z = self.imputer.transform(x)
        # Every unique beer contributes once; no rating replication or class balancing.
        def fit_target(j):
            return ExtraTreesRegressor(**self.protocol['target_forest'],
                min_samples_leaf=self.leaf, random_state=self.seed+j, n_jobs=1
            ).fit(z, distribution[:, j])
        self.models = Parallel(n_jobs=self.protocol['parallel_jobs'], prefer='threads')(
            delayed(fit_target)(j) for j in range(distribution.shape[1]))
        return self

    def predict_distribution(self, x):
        z = self.imputer.transform(x)
        values = np.stack([model.predict(z) for model in self.models], axis=1)
        values = np.maximum(values, 0)
        return values / values.sum(axis=2, keepdims=True)


def distributions(model, x):
    if hasattr(model, 'predict_distribution'):
        return model.predict_distribution(x)
    return base_probabilities(model, x)


def select_targets(probabilities, observed, parents):
    """Select parents first so each child's loss includes its actual selected cap."""
    order = ['shared_et', 'training_prior', 'target_leaf3', 'target_leaf1']
    parent_of = {child: parent for parent, child in parents}
    n, t = observed.shape[:2]
    chosen = np.empty((n, t), int)
    choices, rows, remaining = {}, [], set(range(t))
    while remaining:
        ready = [j for j in sorted(remaining) if j not in parent_of or parent_of[j] in choices]
        if not ready:
            raise ValueError('Cyclic parent hierarchy')
        for j in ready:
            options = {}
            for name in order:
                grade = median_grades(probabilities[name][:, j:j+1])[:, 0]
                if j in parent_of:
                    grade = np.minimum(grade, chosen[:, parent_of[j]])
                loss = rating_loss(observed[:, j:j+1], grade[:, None]).mean()
                options[name] = (grade, float(loss))
            best = min(loss for _, loss in options.values())
            name = next(name for name in order if options[name][1] <= best+1e-12)
            choices[j] = name
            chosen[:, j] = options[name][0]
            rows.extend(dict(target_index=j, method=k, inner_mae=v[1], selected=k == name)
                        for k, v in options.items())
            remaining.remove(j)
    return [choices[j] for j in range(t)], rows


class SelectedTargetPanel:
    def __init__(self, models, prior, choices, parents):
        self.models, self.prior, self.choices, self.parents = models, prior, choices, parents

    def predict_distribution(self, x):
        by_method = {k: distributions(v, x) for k, v in self.models.items() if k in self.choices}
        result = np.empty((len(x), len(self.choices), 7))
        for j, name in enumerate(self.choices):
            result[:, j] = self.prior[j] if name == 'training_prior' else by_method[name][:, j]
        return result

    def predict(self, x):
        raw = median_grades(self.predict_distribution(x))
        return dict(grade=enforce_hierarchy(raw, self.parents), unconstrained_grade=raw,
                    raw_median_estimate=raw.astype(float))


def adoption_decision(metrics, gain, lower95, protocol):
    selected = metrics.loc[metrics.method.eq('inner_selected')]
    baseline = metrics.loc[metrics.method.eq('shared_et')].set_index('repeat')
    limits = protocol['adoption']
    each_improves = all(row.raw_rating_mae < baseline.loc[row.repeat, 'raw_rating_mae']
                        for row in selected.itertuples())
    adopted = (gain >= limits['minimum_mean_mae_gain'] and lower95 > limits['minimum_lower95_gain']
        and (each_improves or not limits['require_each_repeat_gain_positive'])
        and selected.raw_high_fpr.max() <= limits['maximum_raw_high_fpr']
        and selected.panel_high_fpr.max() <= limits['maximum_panel_high_fpr'])
    tail = protocol['tail_solution']
    tail_solved = all(selected[f'{kind}_recall_{k}'].min() >= tail[f'minimum_each_{kind}_recall5_recall6']
                      for kind in ['raw', 'panel'] for k in [5, 6])
    tail_solved &= (selected.raw_high_fpr.max() <= tail['maximum_high_fpr']
                    and selected.panel_high_fpr.max() <= tail['maximum_high_fpr'])
    return bool(adopted), bool(tail_solved)
