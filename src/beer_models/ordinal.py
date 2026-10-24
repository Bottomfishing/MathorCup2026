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
