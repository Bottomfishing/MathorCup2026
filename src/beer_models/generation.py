"""Exploratory style-conditional generators in supplied data coordinates.

AI-assisted: OpenAI Codex, 2026-10-24. No brewing feasibility or physical-unit claim.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from .transforms import RobustAsinh


class StyleGenerator:
    def __init__(self,components=10,covariance_prior=10.):
        self.components=components;self.covariance_prior=covariance_prior

    def fit(self,x,styles):
        x=np.asarray(x,float);styles=np.asarray(styles)
        if len(x)!=len(styles) or len(x)<3 or np.isnan(x).all(axis=0).any():raise ValueError('Invalid training data')
        self.styles_=styles.copy();self.impute_=SimpleImputer(strategy='median').fit(x)
        self.x_=self.impute_.transform(x)
        self.robust_=RobustAsinh().fit(self.x_)
        z=self.robust_.transform(self.x_)
        self.pca_=PCA(n_components=min(self.components,len(x)-1,x.shape[1]),svd_solver='full').fit(z)
        scores=self.pca_.transform(z)
        self.indices_={s:np.flatnonzero(styles==s) for s in sorted(set(styles))}
        self.means_={s:scores[idx].mean(axis=0) for s,idx in self.indices_.items()}
        residual=scores.copy()
        for s,idx in self.indices_.items():residual[idx]-=self.means_[s]
        pooled=np.atleast_2d(np.cov(residual,rowvar=False))
        self.covariances_={}
        for s,idx in self.indices_.items():
            local=np.atleast_2d(np.cov(scores[idx],rowvar=False)) if len(idx)>1 else pooled
            weight=len(idx)/(len(idx)+self.covariance_prior)
            self.covariances_[s]=weight*local+(1-weight)*pooled+np.eye(scores.shape[1])*1e-8
        return self

    def sample(self,style,n,method,seed):
        if style not in self.indices_:raise ValueError('Unknown style')
        if n<1:raise ValueError('Sample count must be positive')
        rng=np.random.default_rng(seed);idx=self.indices_[style]
        parents=np.full((n,2),-1,int);weights=np.full(n,np.nan);clipped=np.zeros(n)
        if method=='resample':
            parents[:,0]=rng.choice(idx,n);generated=self.x_[parents[:,0]].copy()
        elif method=='interpolate':
            if len(idx)<2:raise ValueError('Interpolation needs at least two distinct rows')
            pairs=np.array([rng.choice(idx,2,replace=False) for _ in range(n)])
            parents=pairs;weights=rng.beta(2,2,n)
            generated=weights[:,None]*self.x_[pairs[:,0]]+(1-weights[:,None])*self.x_[pairs[:,1]]
        elif method=='latent_gaussian':
            scores=rng.multivariate_normal(self.means_[style],self.covariances_[style],size=n)
            z=self.pca_.inverse_transform(scores)
            raw=self.robust_.center_+self.robust_.scale_*np.sinh(z)
            lower=self.x_[idx].min(axis=0);upper=self.x_[idx].max(axis=0)
            generated=np.clip(raw,lower,upper)
            clipped=(generated!=raw).mean(axis=1)
        else:raise ValueError('Unknown generation method')
        if not np.isfinite(generated).all():raise ValueError('Nonfinite generated profile')
        return generated,{'parents':parents,'weights':weights,'clipped_fraction':clipped}


def conservative_screen(candidates,source,candidate_sensory,source_sensory,chemical_indices,sensory_indices,tolerance=1e-10):
    """Data-coordinate/proxy nonincrease only; NEVER a physical threshold certificate."""
    candidates=np.asarray(candidates);source=np.asarray(source)
    chemical=candidates[:,chemical_indices]-source[chemical_indices]
    sensory=candidate_sensory[:,sensory_indices]-source_sensory[sensory_indices]
    finite=np.isfinite(chemical).all(axis=1)&np.isfinite(sensory).all(axis=1)
    return finite&(chemical<=tolerance).all(axis=1)&(sensory<=tolerance).all(axis=1),chemical,sensory


def excess_severity(values,threshold):
    """Dimensionless exceedance; caller must establish common units first."""
    values=np.asarray(values,float)
    if not np.isfinite(threshold) or threshold<=0:raise ValueError('Invalid threshold')
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Concentrations must be finite and nonnegative')
    return np.maximum(values/threshold-1,0)
