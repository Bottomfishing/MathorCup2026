"""Decision rules for seven-level relative-grade distributions."""
import numpy as np


def seven_level_probabilities(model, x):
    arrays = model.predict_proba(x)
    result = np.zeros((len(x), len(arrays), 7))
    for j, (probability, classes) in enumerate(zip(arrays, model.classes_)):
        result[:, j, np.asarray(classes, int)] = probability
    return result


def grade_decisions(probabilities):
    p = np.asarray(probabilities, float)
    if p.ndim != 3 or p.shape[2] != 7 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError('Expected finite nonnegative seven-grade probabilities')
    if not np.allclose(p.sum(axis=2), 1):
        raise ValueError('Grade probabilities must sum to one')
    return {'conditional_median': (np.cumsum(p, axis=2) >= .5).argmax(axis=2),
            'conditional_mode': p.argmax(axis=2)}


def ranked_probability_score(probabilities,observed):
    """Normalized ordinal score in [0,1], not a raw-rating validation metric."""
    grade_decisions(probabilities) # shared distribution validation
    p=np.asarray(probabilities,float);y=np.asarray(observed,float)
    if y.shape!=p.shape[:2] or not np.isfinite(y).all() or (y!=np.round(y)).any() or ((y<0)|(y>6)).any():
        raise ValueError('Expected integer relative grades with matching dimensions')
    truth=y[:,:,None]<=np.arange(6)
    return ((np.cumsum(p,axis=2)[:,:,:6]-truth)**2).mean(axis=2)


def central_grade_interval(probabilities,level=.9):
    """Equal-tail forecast interval; uncalibrated probabilities imply no coverage guarantee."""
    grade_decisions(probabilities)
    if not 0<level<1:raise ValueError('Interval level must lie in (0,1)')
    cdf=np.cumsum(np.asarray(probabilities,float),axis=2);tail=(1-level)/2
    # The validated distribution can sum to slightly less than one numerically.
    # Its final quantile must still exist for levels very close to one.
    cdf[:,:,-1]=1.
    return (cdf>=tail).argmax(axis=2),(cdf>=1-tail).argmax(axis=2)
