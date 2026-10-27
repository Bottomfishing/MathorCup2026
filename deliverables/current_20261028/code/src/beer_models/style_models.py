"""Fixed candidates for the concrete weak-centroid and multioutput-splitting gaps."""
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .transforms import RobustAsinh


def make_style_model(name, config, seed):
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if name == 'logistic':
        steps += [('compress', RobustAsinh()), ('scale', StandardScaler())]
        model = LogisticRegression(C=config['logistic_C'], max_iter=config['logistic_max_iter'],
                                   class_weight='balanced', random_state=seed)
    elif name == 'extra_trees':
        model = ExtraTreesClassifier(n_estimators=config['n_estimators'], min_samples_leaf=config['min_samples_leaf'],
            max_features=config['max_features'], class_weight='balanced', random_state=seed, n_jobs=1)
    else:
        raise ValueError('Unknown style model: ' + name)
    return Pipeline(steps + [('model', model)])


def make_stale_model(config, seed):
    return Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('model', ExtraTreesRegressor(n_estimators=config['n_estimators'], min_samples_leaf=config['min_samples_leaf'],
            max_features=config['max_features'], random_state=seed, n_jobs=1))])
