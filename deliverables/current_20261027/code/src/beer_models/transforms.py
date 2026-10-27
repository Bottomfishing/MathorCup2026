"""Train-only robust feature compression. AI-assisted: OpenAI Codex, 2026-10-23."""
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils.validation import check_array,check_is_fitted


class RobustAsinh(TransformerMixin,BaseEstimator):
    """Monotone compression; no deletion, winsorization or physical-unit claim.

    asinh((x - training_median) / training_IQR). If the IQR is zero,
    fall back to training standard deviation, then 1 for constant columns.
    """
    def fit(self,x,y=None):
        x=check_array(x,dtype=float)
        self.n_features_in_=x.shape[1]
        self.center_=np.median(x,axis=0)
        q1,q3=np.quantile(x,[.25,.75],axis=0)
        scale=q3-q1
        std=np.std(x,axis=0)
        self.scale_=np.where(scale>1e-12,scale,np.where(std>1e-12,std,1.))
        return self

    def transform(self,x):
        check_is_fitted(self,['center_','scale_'])
        x=check_array(x,dtype=float)
        if x.shape[1]!=self.n_features_in_:raise ValueError('Feature dimension changed')
        return np.arcsinh((x-self.center_)/self.scale_)
