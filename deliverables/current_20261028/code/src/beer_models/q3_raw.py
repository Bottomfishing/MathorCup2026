"""Finite raw-scale Q3 diagnostics and explicitly unvalidated model guards."""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm, wasserstein_distance
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from .raw_panel import median_grades, enforce_hierarchy
from .tail_risk import base_probabilities, survival


class ShrinkageCopula:
    def fit(self, x, styles, requested_styles=None):
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True).fit(x)
        self.x = self.imputer.transform(x)
        self.marginals, self.correlations, self.shrinkage = {}, {}, {}
        for style in sorted(set(styles) if requested_styles is None else set(requested_styles)):
            local = self.x[np.asarray(styles)==style]
            if len(local)<2:raise ValueError('Copula needs two training beers per style')
            self.marginals[style] = np.sort(local, axis=0)
            ranks = pd.DataFrame(local).rank().to_numpy(float)
            normal = norm.ppf((ranks-.5)/len(local))
            normal -= normal.mean(axis=0)
            scale = normal.std(axis=0)
            normal /= np.where(scale>1e-12,scale,1.)
            fit = LedoitWolf(assume_centered=True).fit(normal)
            sd = np.sqrt(np.maximum(np.diag(fit.covariance_),1e-15))
            correlation = fit.covariance_/sd[:,None]/sd[None,:]
            self.correlations[style] = correlation
            self.shrinkage[style] = float(fit.shrinkage_)
        return self

    def sample(self, style, n, seed):
        marginal = self.marginals[style]
        z = np.random.default_rng(seed).multivariate_normal(np.zeros(marginal.shape[1]),self.correlations[style],size=n)
        u = norm.cdf(z);grid=(np.arange(len(marginal))+.5)/len(marginal)
        return np.column_stack([np.interp(u[:,j],grid,marginal[:,j]) for j in range(marginal.shape[1])])


def rank_correlation(x):
    ranked=pd.DataFrame(x).rank().to_numpy(float,copy=True)
    ranked-=ranked.mean(0)
    size=np.sqrt((ranked*ranked).sum(0))
    return np.divide(ranked.T@ranked,size[:,None]*size[None,:],out=np.zeros((x.shape[1],x.shape[1])),where=(size[:,None]*size[None,:])>0)


def spectrum_diagnostics(train,heldout,generated,protocol):
    p=train.shape[1]
    loo=cdist(train,train)/np.sqrt(p);np.fill_diagonal(loo,np.inf)
    nearest=loo.min(1);radius=float(np.quantile(nearest,protocol['coverage_radius_quantile']))
    copy_radius=float(np.median(nearest)*protocol['near_copy_fraction_of_training_median_LOO_radius'])
    gt=cdist(generated,train)/np.sqrt(p);gg=cdist(generated,generated)/np.sqrt(p)
    np.fill_diagonal(gg,np.inf)
    test_gen=cdist(heldout,generated)/np.sqrt(p)
    hg=cdist(heldout,generated)/np.sqrt(p)
    hh=cdist(heldout,heldout)/np.sqrt(p);same=cdist(generated,generated)/np.sqrt(p)
    off=~np.eye(p,dtype=bool)
    result=dict(n_train=len(train),n_heldout=len(heldout),n_generated=len(generated),
        marginal_w1=float(np.mean([wasserstein_distance(heldout[:,j],generated[:,j]) for j in range(p)])),
        energy=float(2*hg.mean()-hh.mean()-same.mean()),
        variance_ratio=float(np.var(generated,axis=0).mean()/max(np.var(train,axis=0).mean(),1e-15)),
        rank_correlation_rmse=float(np.sqrt(np.mean((rank_correlation(generated)[off]-rank_correlation(train)[off])**2))),
        heldout_coverage=float(np.mean(test_gen.min(1)<=radius)),
        generated_training_radius_precision=float(np.mean(gt.min(1)<=radius)),
        exact_training_copy_rate=float(np.mean(gt.min(1)<1e-10)),
        near_training_copy_rate=float(np.mean(gt.min(1)<=copy_radius)),
        internal_near_duplicate_rate=float(np.mean(gg.min(1)<=copy_radius)),
        training_LOO_radius_p95=radius,near_copy_radius=copy_radius)
    return result


def discrete_w1(left,right):
    return np.abs(np.cumsum(left,axis=-1)[...,:-1]-np.cumsum(right,axis=-1)[...,:-1]).sum(axis=-1)


def profile_diagnostics(generator,candidates,ids,features,protocol):
    rows=[];p=len(features)
    for (method,style),group in candidates.groupby(['method','style'],sort=False):
        indices=generator.indices_[style];real=generator.robust_.transform(generator.x_[indices])
        values=group[features].to_numpy(float);z=generator.robust_.transform(values)
        loo=cdist(real,real)/np.sqrt(p);np.fill_diagonal(loo,np.inf)
        radius=float(np.median(loo.min(1))*protocol['near_copy_fraction_of_training_median_LOO_radius'])
        distances=cdist(z,real)/np.sqrt(p);internal=cdist(z,z)/np.sqrt(p);np.fill_diagonal(internal,np.inf)
        a,b=distances.argmin(1),internal.argmin(1)
        lower,upper=generator.x_[indices].min(0),generator.x_[indices].max(0)
        keys=group.generated_id.to_numpy()
        for k,gid in enumerate(keys):
            rows.append(dict(generated_id=gid,method=method,style=style,nearest_real_beer=int(ids.beer.iloc[indices[a[k]]]),
                nearest_real_distance=float(distances[k,a[k]]),exact_training_copy=bool(distances[k,a[k]]<1e-10),
                near_training_copy=bool(distances[k,a[k]]<=radius),nearest_generated_id=keys[b[k]],
                nearest_generated_distance=float(internal[k,b[k]]),internal_near_duplicate=bool(internal[k,b[k]]<=radius),
                near_copy_radius=radius,outside_style_feature_range_count=int(((values[k]<lower-1e-10)|(values[k]>upper+1e-10)).sum())))
    return pd.DataFrame(rows)


def raw_predictions(bundle,x,parents):
    risk_model=bundle['risk_panel_model']
    grade=bundle['raw_panel_model'].predict(x)['grade']
    return dict(grade=grade,calibrated=risk_model.predict_distribution(x),
        uncalibrated=risk_model.predict_uncalibrated_distribution(x))


def defect_guard(before,after,indices,tolerance):
    delta_grade=after['grade'][:,indices]-before['grade'][:,indices]
    deltas={}
    for name in ['calibrated','uncalibrated']:
        a=survival(after[name])[:,indices][:,:,[0,4,5]]
        b=survival(before[name])[:,indices][:,:,[0,4,5]]
        deltas[name]=a-b
    point=(delta_grade<=0).all(1)
    cal=(deltas['calibrated']<=tolerance).all(axis=(1,2))
    uncal=(deltas['uncalibrated']<=tolerance).all(axis=(1,2))
    return point&cal&uncal,dict(grade=delta_grade,**deltas),dict(grade=point,calibrated=cal,uncalibrated=uncal)


def resampled_records(data,draw):
    positions,scores,weights=[],[],[]
    for occurrence,beer_index in enumerate(draw):
        rows=data['positions']==beer_index
        positions.extend([occurrence]*int(rows.sum()))
        scores.append(data['scores'][rows]);weights.append(data['weights'][rows])
    return np.asarray(positions,int),np.concatenate(scores),np.concatenate(weights)


def uncalibrated_refit_predictions(model,x,parents):
    p=base_probabilities(model,x)
    return dict(grade=enforce_hierarchy(median_grades(p),parents),probability=survival(p)[:,:,[0,4,5]])
