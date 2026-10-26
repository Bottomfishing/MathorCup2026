"""Standalone two-stage high-rating model. No official module imports this file."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from beer_models.raw_panel import enforce_hierarchy, median_grades, rating_loss
from beer_models.tail_risk import base_probabilities
from beer_models.selective_tail import selective_decision


def decide(probability, original_grade, prior, mae_penalty, fp_penalty, parents):
    """Same tail utility as the old experiment, with the original score as baseline."""
    p, base = np.asarray(probability), np.asarray(original_grade)
    loss = rating_loss(p, base)
    reward0 = sum(.5*p[:, :, k]/prior[k]*(base == k) for k in [5, 6] if prior[k] > 0)
    utility, actions = [np.zeros(base.shape)], [base]
    low = p[:, :, :5].sum(2)
    for k in [5, 6]:
        action = np.full(base.shape, k)
        reward = .5*p[:, :, k]/prior[k] if prior[k] > 0 else np.full(base.shape, -np.inf)
        utility.append(reward-reward0-mae_penalty*(rating_loss(p, action)-loss)-fp_penalty*low*(1-(base >= 5)))
        actions.append(action)
    which = np.stack(utility, 2).argmax(2)
    grade = np.take_along_axis(np.stack(actions, 2), which[:, :, None], 2)[:, :, 0]
    return enforce_hierarchy(grade, parents)


class BinaryStage:
    def __init__(self, config, seed):
        self.config, self.seed = config, seed

    def fit(self, x, negative, positive):
        negative, positive = np.asarray(negative, float), np.asarray(positive, float)
        mass0, mass1 = negative.sum(), positive.sum()
        self.n_exposed_beers = int(((negative+positive) > 0).sum())
        self.mass0, self.mass1 = float(mass0), float(mass1)
        self.prior = float(mass1/(mass0+mass1)) if mass0+mass1 > 0 else 0.
        # An unobserved conditional stage is represented by zero high6 mass,
        # never by invented negative training labels.
        self.model = None
        xx = np.concatenate([x, x])
        y = np.r_[np.zeros(len(x), int), np.ones(len(x), int)]
        raw_weights = np.r_[negative, positive]
        keep = raw_weights > 0
        if mass0 == 0 or mass1 == 0 or keep.sum() < 2*self.config['min_samples_leaf']:
            self.kind = 'observed_prior_sparse_or_one_class'
            return self
        self.w0, self.w1 = (mass0+mass1)/(2*mass0), (mass0+mass1)/(2*mass1)
        weights = raw_weights*np.where(y == 0, self.w0, self.w1)
        self.model = GradientBoostingClassifier(**self.config, random_state=self.seed).fit(xx[keep], y[keep], sample_weight=weights[keep])
        self.kind = 'balanced_boosted_classifier_odds_restored'
        return self

    def predict(self, x):
        if self.model is None:
            return np.full(len(x), self.prior)
        p = self.model.predict_proba(x)[:, 1]
        return p*self.w0/(p*self.w0+(1-p)*self.w1)


class TwoStageHighModel:
    def __init__(self, protocol, seed):
        self.protocol, self.seed = protocol, seed

    def fit(self, x, q):
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
        z = self.imputer.transform(x)
        self.low_prior = q.mean(0)[:, :5]
        def fit_target(j):
            high = q[:, j, 5:].sum(1)
            stage1 = BinaryStage(self.protocol['model'], self.seed+2*j).fit(z, q[:, j, :5].sum(1), high)
            stage2 = BinaryStage(self.protocol['model'], self.seed+2*j+1).fit(z, q[:, j, 5], q[:, j, 6])
            return stage1, stage2
        self.stages = Parallel(n_jobs=self.protocol['jobs'], prefer='threads')(
            delayed(fit_target)(j) for j in range(q.shape[1]))
        return self

    def predict_stages(self, x):
        z = self.imputer.transform(x)
        h = np.column_stack([a.predict(z) for a, _ in self.stages])
        r = np.column_stack([b.predict(z) for _, b in self.stages])
        return h, r

    def predict_distribution(self, x, baseline):
        h, conditional6 = self.predict_stages(x)
        low = baseline[:, :, :5].copy()
        mass = low.sum(2, keepdims=True)
        fallback = np.broadcast_to(self.low_prior, low.shape)
        low = np.where(mass > 0, low, fallback)
        mass = low.sum(2, keepdims=True)
        low = np.divide(low, mass, out=np.zeros_like(low), where=mass > 0)
        result = np.concatenate([low*(1-h[:, :, None]), (h*(1-conditional6))[:, :, None], (h*conditional6)[:, :, None]], axis=2)
        if not np.allclose(result.sum(2), 1, atol=1e-12):
            raise ValueError('Unsupported low-grade distribution')
        return result

    def stage_records(self, context, targets):
        return [dict(context=context, target=targets[j], stage=k+1, kind=stage.kind,
            n_exposed_beers=stage.n_exposed_beers, negative_mass=stage.mass0,
            positive_mass=stage.mass1, prior=stage.prior)
            for j, pair in enumerate(self.stages) for k, stage in enumerate(pair)]


class ExperimentalPredictor:
    def __init__(self, baseline, high_model, prior, policy, parents):
        self.baseline, self.high_model, self.prior, self.policy, self.parents = baseline, high_model, prior, policy, parents

    def predict_distribution(self, x):
        return self.high_model.predict_distribution(x, base_probabilities(self.baseline, x))

    def predict(self, x):
        base = base_probabilities(self.baseline, x)
        p = self.high_model.predict_distribution(x, base)
        original = enforce_hierarchy(median_grades(base), self.parents)
        grade = original if self.policy is None else decide(p, original, self.prior, *self.policy, self.parents)
        return dict(grade=grade, raw_median_estimate=median_grades(p).astype(float))


class JointTailForest(TwoStageHighModel):
    """One joint low/5/6 forest per target, trained on unique beer fractions."""
    def fit(self, x, q):
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
        z = self.imputer.transform(x)
        self.low_prior = q.mean(0)[:, :5]
        response = np.stack([q[:, :, :5].sum(2), q[:, :, 5], q[:, :, 6]], axis=2)
        # Scaling changes split priorities only. Leaf predictions, after undoing
        # it, remain nonnegative averages of the original observed fractions.
        self.scales = np.sqrt(np.where(response.mean(0) > 0, response.mean(0), 1.))
        self.training_mass = response.sum(0)
        self.n_training_beers = len(x)
        def fit_target(j):
            return RandomForestRegressor(**self.protocol['model'], random_state=self.seed+j, n_jobs=1).fit(z, response[:, j]/self.scales[j])
        self.models = Parallel(n_jobs=self.protocol['jobs'], prefer='threads')(
            delayed(fit_target)(j) for j in range(q.shape[1]))
        return self

    def predict_stages(self, x):
        z = self.imputer.transform(x)
        p = np.stack([m.predict(z)*self.scales[j] for j, m in enumerate(self.models)], axis=1)
        p = np.maximum(p, 0)
        p /= p.sum(2, keepdims=True)
        high = p[:, :, 1:].sum(2)
        conditional = np.divide(p[:, :, 2], high, out=np.zeros_like(high), where=high > 0)
        return high, conditional

    def stage_records(self, context, targets):
        return [dict(context=context, target=t, stage=0, kind='joint_low5_6_scaled_probability_forest',
            n_exposed_beers=self.n_training_beers, negative_mass=self.training_mass[j, 0],
            positive_mass=self.training_mass[j, 1:].sum(), prior=self.training_mass[j, 1:].sum()/self.n_training_beers)
            for j, t in enumerate(targets)]


def make_high_model(protocol, seed):
    return (JointTailForest if protocol.get('family') == 'joint_tail_forest' else TwoStageHighModel)(protocol, seed)


def apply_gates(original, proposed, gates, parents):
    grade = original.copy()
    allowed = (proposed >= 5)&gates['high'][None]
    grade[allowed] = 5
    exact6 = (proposed == 6)&gates['six'][None]&allowed
    grade[exact6] = 6
    for parent, child in parents:
        grade[:, parent] = np.maximum(grade[:, parent], grade[:, child])
    return grade


class ConfidentPredictor:
    def __init__(self, model, gates):
        self.model, self.gates, self.policy = model, gates, model.policy

    def predict_distribution(self, x):
        return self.model.predict_distribution(x)

    def predict(self, x):
        p = self.model.predict_distribution(x)
        original = self.model.baseline.predict(x)['grade']
        proposed = original if self.policy is None else decide(p, original, self.model.prior, *self.policy, [])
        return dict(grade=apply_gates(original, proposed, self.gates, self.model.parents),
                    raw_median_estimate=median_grades(p).astype(float))
