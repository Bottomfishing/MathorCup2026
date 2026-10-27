"""One unknown coordinate: symbolic preservation and reachable forest states."""
import numpy as np


def preserve_unknown(source, proposed, index):
    result = np.asarray(proposed, float).copy()
    result[..., index] = np.asarray(source, float)[index]
    return result


def reachable_forest_points(forest, fixed, index, lower, upper):
    """Cover every reachable float32 split state on a finite nonnegative domain.

    sklearn tree predictors cast input to float32 before comparing to float64
    thresholds. Store the largest float32 <= threshold and its successor,
    rather than nextafter on a float64 threshold that rounds to the same input.
    """
    if not (np.isfinite(lower) and np.isfinite(upper) and lower <= upper):
        raise ValueError('A finite ordered model-support domain is required')
    lo, hi = np.float32(lower), np.float32(upper)
    x = np.asarray(fixed, np.float32)
    if not np.isfinite(np.delete(x, index)).all():
        raise ValueError('All other coordinates must be resolved by their declared preprocessing')
    points = {float(lo), float(hi)}
    for estimator in forest.estimators_:
        tree = estimator.tree_
        stack = [(0, lo, hi)]
        while stack:
            node, a, c = stack.pop()
            feature = tree.feature[node]
            if feature < 0: continue
            if feature != index:
                child = tree.children_left[node] if x[feature] <= tree.threshold[node] else tree.children_right[node]
                stack.append((child, a, c))
                continue
            left = np.float32(tree.threshold[node])
            if float(left) > tree.threshold[node]: left = np.nextafter(left, np.float32(-np.inf))
            right = np.nextafter(left, np.float32(np.inf))
            if a <= left and c >= right:
                # In the REAL input coordinate, float32 rounding switches at
                # the halfway value (the midpoint itself has an even tie rule).
                midpoint = (float(left)+float(right))/2
                if lower <= midpoint <= upper:
                    points.update([midpoint, float(np.nextafter(midpoint, -np.inf)), float(np.nextafter(midpoint, np.inf))])
            if a <= left:
                end = min(c, left); points.add(float(end))
                stack.append((tree.children_left[node], a, end))
            if c >= right:
                start = max(a, right); points.add(float(start))
                stack.append((tree.children_right[node], start, c))
    return np.array(sorted(points))


def logistic_margin_lower_bound(pipeline, fixed, index, lower, upper, target='Tripel'):
    """Conservative all-domain target probability-gap bound from logit ratios.

    Other inputs are fixed. Each class logit difference is affine in a monotone
    asinh of this coordinate, so each competitor/target ratio attains its maximum
    at an endpoint. This bounds the full softmax gap, not just sampled labels.
    """
    x = np.broadcast_to(fixed, (2, len(fixed))).copy()
    x[:, index] = [lower, upper]
    model = pipeline[-1]
    logits = model.decision_function(pipeline[:-1].transform(x))
    ti = list(model.classes_).index(target)
    competitors = np.delete(logits-logits[:, ti:ti+1], ti, axis=1)
    ratios = np.exp(np.clip(competitors.max(0), -700, 700))
    if ratios.max() > 1:
        return -1.  # Negative numerators cannot use an upper denominator bound.
    return float((1-ratios.max())/(1+ratios.sum()))


def logistic_target_crossings(pipeline, fixed, index, lower, upper, target='Tripel'):
    """All target-vs-competitor class crossings in the monotone asinh coordinate."""
    x = np.broadcast_to(fixed, (2, len(fixed))).copy(); x[:, index] = [lower, upper]
    logits = pipeline[-1].decision_function(pipeline[:-1].transform(x))
    ti = list(pipeline[-1].classes_).index(target)
    delta = logits[:, ti:ti+1]-logits
    robust = pipeline.named_steps['compress']
    phi = np.arcsinh((np.array([lower, upper])-robust.center_[index])/robust.scale_[index])
    denominator = delta[1]-delta[0]
    mask = (abs(denominator) > 1e-14)&(delta[0]*delta[1] <= 0)
    root = phi[0]-delta[0, mask]*(phi[1]-phi[0])/denominator[mask]
    values = robust.center_[index]+robust.scale_[index]*np.sinh(root)
    return values[(values >= lower)&(values <= upper)]


def distance_improvement_lower_bound(robust, source, proposed, center, index, lower, upper):
    # Both distances have the SAME unknown-coordinate squared term. Their ratio
    # has its worst value at one of the largest/smallest attainable terms.
    points = [lower, upper]
    raw_center = robust.center_[index]+robust.scale_[index]*np.sinh(center[index])
    if lower <= raw_center <= upper: points.append(raw_center)
    a = np.broadcast_to(source, (len(points), len(source))).copy()
    c = np.broadcast_to(proposed, a.shape).copy()
    a[:, index] = points; c[:, index] = points
    da = np.sqrt(np.square(robust.transform(a)-center).mean(1))
    dc = np.sqrt(np.square(robust.transform(c)-center).mean(1))
    return float((1-dc/np.maximum(da, 1e-12)).min())


def maximum_nearest_distance(robust, proposed, reference_z, index, lower, upper):
    """Exact 1D lower-envelope maximum, including pairwise parabola crossings."""
    endpoints = np.broadcast_to(proposed, (2, len(proposed))).copy()
    endpoints[:, index] = [lower, upper]
    values = robust.transform(endpoints)
    lo, hi = values[:, index]
    fixed_delta = values[0]-reference_z
    fixed_delta[:, index] = 0
    offsets = np.square(fixed_delta).sum(1)
    centers = reference_z[:, index]
    i, k = np.triu_indices(len(centers), 1)
    denom = 2*(centers[i]-centers[k])
    crossings = np.divide(offsets[i]+centers[i]**2-offsets[k]-centers[k]**2, denom,
        out=np.full(len(i), np.nan), where=abs(denom) > 1e-14)
    knots = np.unique(np.r_[lo, hi, crossings[(crossings >= lo)&(crossings <= hi)]])
    maximum = 0.
    for start in range(0, len(knots), 1024):
        distances = offsets[None]+np.square(knots[start:start+1024, None]-centers[None])
        maximum = max(maximum, float(distances.min(1).max()))
    return float(np.sqrt(maximum/len(proposed)))
