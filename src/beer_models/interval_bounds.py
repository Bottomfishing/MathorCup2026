"""Conservative fitted-model bounds over finite coordinate boxes.

AI-assisted: Codex, 2026-10-24. These are deterministic model ranges, not
statistical confidence intervals, physical bounds or intervention guarantees.
"""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from .transforms import RobustAsinh


def _tree_bounds(tree,lower,upper,target):
    todo=[0];minimum=np.inf;maximum=-np.inf
    while todo:
        node=todo.pop();feature=tree.feature[node]
        if feature<0:
            value=float(tree.value[node,target,0]);minimum=min(minimum,value);maximum=max(maximum,value)
        else:
            if lower[feature]<=tree.threshold[node]:todo.append(tree.children_left[node])
            if upper[feature]>tree.threshold[node]:todo.append(tree.children_right[node])
    if not np.isfinite([minimum,maximum]).all():raise ValueError('Box reaches no finite leaf')
    return minimum,maximum


def prediction_bounds(model,lower,upper,target=0):
    lower,upper=np.asarray(lower,float),np.asarray(upper,float)
    if lower.ndim!=1 or lower.shape!=upper.shape or not np.isfinite([lower,upper]).all() or (lower>upper).any():
        raise ValueError('Finite ordered box endpoints required')
    if isinstance(model,TransformedTargetRegressor):
        if not isinstance(model.transformer_,StandardScaler):raise ValueError('Unsupported target inverse transform')
        low,high=prediction_bounds(model.regressor_,lower,upper,target)
        dimensions=model.transformer_.n_features_in_
        zero=np.zeros((1,dimensions));unit=zero.copy();unit[0,target]=1
        offset=model.transformer_.inverse_transform(zero)[0,target]
        scale=model.transformer_.inverse_transform(unit)[0,target]-offset
        if scale<=0:raise ValueError('Target inverse must have positive scale')
        return float(offset+scale*low),float(offset+scale*high)
    if isinstance(model,Pipeline):
        endpoints=np.vstack([lower,upper])
        for _,step in model.steps[:-1]:
            if not isinstance(step,(SimpleImputer,RobustAsinh,StandardScaler)):
                raise ValueError('Only verified coordinatewise monotone transforms are supported')
            endpoints=step.transform(endpoints)
        if endpoints.shape!=(2,len(lower)):raise ValueError('Preprocessing changes coordinates')
        return prediction_bounds(model.steps[-1][1],endpoints[0],endpoints[1],target)
    if isinstance(model,Ridge):
        coef=np.asarray(model.coef_);intercept=np.asarray(model.intercept_)
        if coef.ndim==2:coef=coef[target];intercept=intercept[target]
        elif target!=0:raise ValueError('Invalid scalar target index')
        low=intercept+np.where(coef>=0,coef*lower,coef*upper).sum()
        high=intercept+np.where(coef>=0,coef*upper,coef*lower).sum()
        return float(low),float(high)
    if isinstance(model,(DecisionTreeRegressor,ExtraTreesRegressor,RandomForestRegressor)):
        # sklearn regression trees evaluate input features in float32.
        lo=lower.astype(np.float32).astype(float);hi=upper.astype(np.float32).astype(float)
        if not np.isfinite([lo,hi]).all():raise ValueError('Tree input exceeds float32 range')
        estimators=[model] if isinstance(model,DecisionTreeRegressor) else model.estimators_
        bounds=np.array([_tree_bounds(estimator.tree_,lo,hi,target) for estimator in estimators])
        # Sum of per-tree maxima is conservative; maximizers can differ by tree.
        return tuple(float(v) for v in bounds.mean(axis=0))
    raise ValueError('Unsupported fitted model type: '+type(model).__name__)
