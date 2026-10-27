"""Predict original panel rating medians with beer/taster-balanced raw labels."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor


def rating_weights(raw):
    repeats = raw.groupby(['beer', 'taster']).taster.transform('size').to_numpy(float)
    tasters = raw.groupby('beer').taster.transform('nunique').to_numpy(float)
    return 1.0 / (repeats * tasters)


def rating_distribution(beer_positions, scores, weights, n_beers):
    positions, scores, weights = np.asarray(beer_positions, int), np.asarray(scores), np.asarray(weights, float)
    if scores.ndim != 2 or len(scores) != len(positions) or len(weights) != len(scores):
        raise ValueError('Rating records and beer positions must align')
    if (not np.isfinite(scores).all() or not np.equal(scores, np.floor(scores)).all()
            or ((scores < 0) | (scores > 6)).any() or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError('Expected original grades and positive taster weights')
    if (positions < 0).any() or (positions >= n_beers).any():
        raise ValueError('Rating beer index out of range')
    mass = np.bincount(positions, weights=weights, minlength=n_beers)
    if not np.allclose(mass, 1.0):
        raise ValueError('Each beer must have total rating weight one')
    result = np.zeros((n_beers, scores.shape[1], 7))
    for j in range(scores.shape[1]):
        np.add.at(result[:, j], (positions, scores[:, j].astype(int)), weights)
    return result


def median_grades(probabilities):
    p = np.asarray(probabilities, float)
    if (p.ndim != 3 or p.shape[2] != 7 or not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(axis=2), 1.0)):
        raise ValueError('Expected normalized seven-grade distributions')
    return (p.cumsum(axis=2) >= .5 - 1e-12).argmax(axis=2)


def round_grade(values):
    values = np.asarray(values, float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite raw score prediction')
    return np.floor(np.clip(values, 0, 6) + .5 - 1e-12).clip(0, 6).astype(int)


def enforce_hierarchy(grades, parents):
    result = np.asarray(grades, int).copy()
    for parent, child in parents:
        result[:, child] = np.minimum(result[:, child], result[:, parent])
    return result


def rating_loss(distributions, grades):
    return np.sum(distributions * abs(np.arange(7) - np.asarray(grades)[..., None]), axis=2)


class RawPanelModel:
    def __init__(self, method, protocol, seed, parents):
        self.method, self.protocol, self.seed, self.parents = method, protocol, seed, parents

    def fit(self, x, positions, scores, weights, distributions):
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
        z = self.imputer.transform(x)
        self.n_training_beers = len(x)
        self.reference_distribution = distributions.mean(axis=0)
        if self.method == 'training_median':
            self.model = None
            return self
        leaf = self.protocol['minimum_leaf_beer_weight'] / len(x)
        if self.method == 'extra_trees':
            self.model = ExtraTreesClassifier(**self.protocol['extra_trees'], random_state=self.seed,
                min_samples_leaf=1, min_weight_fraction_leaf=leaf, n_jobs=self.protocol['parallel_jobs'])
        elif self.method in ['gbr_depth1', 'gbr_depth2']:
            base = GradientBoostingRegressor(**self.protocol['gbr'], random_state=self.seed,
                max_depth=int(self.method[-1]), min_samples_leaf=1, min_weight_fraction_leaf=leaf)
            self.model = MultiOutputRegressor(base, n_jobs=self.protocol['parallel_jobs'])
        else:
            raise ValueError('Unknown raw panel candidate: ' + self.method)
        # Every beer has weight one. Min-weight leaf bounds cannot be bypassed by its repeats.
        self.model.fit(z[positions], scores, sample_weight=weights)
        return self

    def predict(self, x):
        z = self.imputer.transform(x)
        probabilities = None
        if self.method == 'training_median':
            probabilities = np.broadcast_to(self.reference_distribution, (len(x),) + self.reference_distribution.shape)
        elif self.method == 'extra_trees':
            arrays = self.model.predict_proba(z)
            probabilities = np.zeros((len(x), len(arrays), 7))
            for j, (array, classes) in enumerate(zip(arrays, self.model.classes_)):
                probabilities[:, j, np.asarray(classes, int)] = array
        if probabilities is not None:
            unconstrained = median_grades(probabilities)
            point = unconstrained.astype(float)
        else:
            point = np.clip(self.model.predict(z), 0, 6)
            unconstrained = round_grade(point)
        return {'grade': enforce_hierarchy(unconstrained, self.parents),
                'unconstrained_grade': unconstrained, 'raw_median_estimate': point}


def predict_raw_frame(frame, bundle):
    from beer_audit.checks import numeric_frame
    if frame.empty:
        raise ValueError('Input contains no samples')
    features, targets = bundle['features'], bundle['targets']
    beer = (frame.beer.astype('string') if 'beer' in frame else
            pd.Series([f'row_{i+1:06d}' for i in range(len(frame))], index=frame.index))
    if beer.isna().any() or beer.str.strip().eq('').any() or beer.duplicated().any():
        raise ValueError('beer must be nonempty and unique')
    values = numeric_frame(frame.reindex(columns=features)).to_numpy(float)
    observed = np.isfinite(values).sum(axis=1)
    if (observed == 0).any():
        raise ValueError('No observed model features for one or more input samples')
    predicted = bundle['raw_panel_model'].predict(values)
    scope = pd.DataFrame(bundle['scorecard_scope'])
    rows = pd.DataFrame({'beer': np.repeat(beer.to_numpy(), len(targets)),
        'target': np.tile(targets, len(frame)), 'estimated_grade_0_6': predicted['grade'].ravel(),
        'raw_median_estimate': predicted['raw_median_estimate'].ravel(),
        'score_status': bundle.get('score_status', 'raw_panel_balanced_median_prediction'), 'panel_target_available': True})
    missing = pd.DataFrame([{'beer': b, 'target': target, 'estimated_grade_0_6': pd.NA,
        'raw_median_estimate': np.nan, 'score_status': 'no_raw_panel_label', 'panel_target_available': False}
        for b in beer for target in bundle['protocol']['unlabelled_card_targets']])
    rows = pd.concat([rows, missing], ignore_index=True)
    rows['prediction_role'] = 'full_training_model_inference_not_OOF'
    rows['estimated_grade_0_6'] = rows.estimated_grade_0_6.astype('Int64')
    order_b = {b: i for i, b in enumerate(beer)}
    order_t = {t: i for i, t in enumerate(scope.target_id)}
    rows = rows.sort_values(['beer', 'target'], key=lambda c: c.map(order_b if c.name == 'beer' else order_t)).reset_index(drop=True)
    wide = rows.pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[beer, scope.target_id].reset_index()
    primary = wide[['beer', *scope.loc[scope.acceptance_scope.eq('main_content'), 'target_id']]]
    quality = pd.DataFrame({'beer': beer.to_numpy(), 'observed_feature_count': observed,
        'missing_feature_fraction': 1 - observed / len(features)})
    return rows, primary, wide, quality


def fit_shared(data, train, protocol, seed):
    lookup = np.full(len(data['x']), -1, int)
    lookup[train] = np.arange(len(train))
    rows = lookup[data['positions']] >= 0
    return RawPanelModel('extra_trees', protocol, seed, data['parents']).fit(data['x'][train],
        lookup[data['positions'][rows]], data['scores'][rows], data['weights'][rows], data['distribution'][train])


def portrait(frame,bundle):
    from .tail_risk import risk_frame
    long,primary,wide,quality=predict_raw_frame(frame,bundle)
    long=long.merge(risk_frame(frame,bundle,long.beer.drop_duplicates().to_numpy()),on=['beer','target'],how='left',validate='one_to_one')
    long['safety_certified']=False;long['actual_after_measured']=False
    return long,primary,wide,quality
