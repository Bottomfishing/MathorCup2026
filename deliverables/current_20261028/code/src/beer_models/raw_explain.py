"""Held-out explanations on the original panel scale; no causal interpretation."""
import numpy as np
import pandas as pd
import networkx as nx
from .baseline import make_pipeline
from .raw_panel import round_grade, enforce_hierarchy, rating_loss
from .explain import stratified_draw


def fit_raw_ridge(x, distribution, alpha, seed):
    model, _ = make_pipeline('ridge_asinh', {'ridge_alphas': [alpha]}, seed)
    model.set_params(model__alpha=alpha)
    return model.fit(x, distribution @ np.arange(7))


def explain_linear(model, x, background):
    z = model[:-1].transform(x)
    centre = model[:-1].transform(background).mean(axis=0)
    coefficients = np.atleast_2d(model.named_steps['model'].coef_)
    values = (z-centre)[:, None, :] * coefficients[None, :, :]
    base = model.named_steps['model'].intercept_ + coefficients @ centre
    prediction = np.asarray(model.predict(x)).reshape(len(x), -1)
    return values, np.broadcast_to(base, values.shape[:2]).copy(), prediction


def explain_quantile(model, x, background):
    import shap
    z, bg = model.imputer.transform(x), model.imputer.transform(background)
    values = np.empty((len(x), len(model.model.estimators_), x.shape[1]))
    base = np.empty(values.shape[:2])
    for j, estimator in enumerate(model.model.estimators_):
        explainer = shap.TreeExplainer(estimator, data=bg, model_output='raw',
                                      feature_perturbation='interventional')
        values[:, j] = explainer.shap_values(z, approximate=False, check_additivity=True)
        base[:, j] = float(np.asarray(explainer.expected_value).item())
    return values, base, model.model.predict(z)


def check_additivity(values, base, prediction, tolerance=1e-6):
    error = abs(values.sum(axis=2) + base - prediction)
    if not np.isfinite(error).all() or error.max() > tolerance:
        raise ValueError('Explanation does not reconstruct the unprocessed model output')
    return float(error.max())


def decision(prediction, parents):
    return enforce_hierarchy(round_grade(prediction), parents)


def nonzero_top(scores, eligible, k=3):
    eligible = np.asarray(eligible, int)
    valid = eligible[np.isfinite(scores[eligible]) & (scores[eligible] > 1e-12)]
    return valid[np.argsort(-scores[valid], kind='stable')[:k]]


def training_modules(imputed_training, threshold):
    """Learn correlated components using training predictors only, without labels."""
    rho = pd.DataFrame(imputed_training).corr(method='spearman').fillna(0).to_numpy()
    edges = np.column_stack(np.where(np.triu(abs(rho) >= threshold, k=1)))
    graph = nx.Graph()
    graph.add_nodes_from(range(imputed_training.shape[1]))
    graph.add_edges_from(edges.tolist())
    groups = [sorted(g) for g in nx.connected_components(graph)]
    groups.sort(key=lambda g: g[0])
    return groups, [(int(i), int(j), float(rho[i, j])) for i, j in edges]


def module_importance(values, groups):
    # Opposite contributions cancel within a group before taking absolute values.
    return np.column_stack([abs(values[:, :, group].sum(axis=2)).mean(axis=0) for group in groups])


def attribution_direction(x, values):
    result = np.zeros(values.shape[1:])
    for j in range(x.shape[1]):
        valid = np.isfinite(x[:, j])
        ranks = pd.Series(x[valid, j]).rank().to_numpy(float, copy=True)
        contribution = pd.DataFrame(values[valid, :, j]).rank().to_numpy(float, copy=True)
        ranks -= ranks.mean(); contribution -= contribution.mean(axis=0)
        den = np.sqrt((ranks*ranks).sum() * (contribution*contribution).sum(axis=0))
        result[:, j] = np.divide(ranks @ contribution, den, out=np.zeros(values.shape[1]), where=den>0)
    return result


def predictive_support(q, grades, baseline, styles, protocol):
    difference = rating_loss(q, baseline) - rating_loss(q, grades)
    rng = np.random.default_rng(protocol['bootstrap_seed'])
    draws = np.array([difference[stratified_draw(styles, rng)].mean(axis=0)
                      for _ in range(protocol['bootstrap_repeats'])])
    mean = difference.mean(axis=0)
    lower, upper = np.quantile(draws, [.025, .975], axis=0)
    return mean, lower, upper
