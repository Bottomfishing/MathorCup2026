"""Q3 saved-model replay and independent constraints. AI-assisted: Codex, 2026-10-24."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
from beer_models.generation import StyleGenerator
from beer_audit.audit import sha256


def verify(out):
    manifest=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    cfg=json.loads((out/'config.json').read_text(encoding='utf-8'));source=Path(manifest['source'])
    for name,digest in manifest['source_sha256'].items():assert sha256(source/name)==digest
    assert sha256(out/'generator.joblib')==manifest['generator_sha256']
    g=joblib.load(out/'generator.joblib')
    names=pd.read_csv(out/'feature_order.csv').field.tolist();targets=pd.read_csv(out/'target_order.csv').target.tolist()
    samples=pd.read_csv(out/'samples.csv',dtype={'beer':str})
    raw=pd.read_csv(source/'1.csv',dtype={'beer':str}).set_index('beer').loc[samples.beer]
    x=raw[names].to_numpy(float);styles=samples.tasting_category_fine.to_numpy()
    np.testing.assert_allclose(g.impute_.statistics_,np.nanmedian(x,axis=0))
    table=pd.read_csv(out/'generated_profiles.csv').set_index('generated_id')
    meta=pd.read_csv(out/'generated_metadata.csv',dtype={'parent_1':str,'parent_2':str})
    assert table.index.is_unique and len(table)==900 and np.isfinite(table[names]).all().all()
    assert not meta.physical_feasibility_certified.any() and not meta.aggregate_identities_verified.any()
    for (style,method),group in meta.groupby(['style','method'],sort=False):
        generated,details=g.sample(style,len(group),method,int(group.seed.iloc[0]))
        np.testing.assert_allclose(table.loc[group.generated_id,names],generated,atol=1e-10,rtol=1e-12)
        np.testing.assert_allclose(group.clipped_feature_fraction,details['clipped_fraction'],atol=1e-12)
        reference=g.x_[g.indices_[style]]
        assert (generated>=reference.min(axis=0)-1e-10).all() and (generated<=reference.max(axis=0)+1e-10).all()
        if method=='interpolate':
            a=details['parents'][:,0];b=details['parents'][:,1];w=details['weights'][:,None]
            np.testing.assert_allclose(generated,w*g.x_[a]+(1-w)*g.x_[b],atol=1e-12)
    # Refit fold generators and recompute each distribution diagnostic.
    validation=pd.read_csv(out/'validation_by_style_fold.csv')
    for fold in sorted(samples.outer_test_fold.unique()):
        train=np.flatnonzero(samples.outer_test_fold.to_numpy()!=fold);test=np.flatnonzero(samples.outer_test_fold.to_numpy()==fold)
        gg=StyleGenerator(cfg['latent_components'],cfg['covariance_prior_sample_size']).fit(x[train],styles[train])
        np.testing.assert_allclose(gg.impute_.statistics_,np.nanmedian(x[train],axis=0))
        for si,style in enumerate(sorted(set(styles))):
            hold=gg.robust_.transform(gg.impute_.transform(x[test[styles[test]==style]]));real=gg.robust_.transform(gg.x_[gg.indices_[style]])
            for mi,method in enumerate(cfg['methods']):
                generated,detail=gg.sample(style,cfg['validation_samples_per_style'],method,cfg['seed']+int(fold)*10000+si*100+mi)
                z=gg.robust_.transform(generated);distance=cdist(z,real).min(axis=1)/np.sqrt(len(names))
                row=validation.loc[validation.fold.eq(fold)&validation['style'].eq(style)&validation.method.eq(method)].iloc[0]
                actual=[np.mean([wasserstein_distance(hold[:,j],z[:,j]) for j in range(len(names))]),
                        np.var(z,axis=0).sum()/max(np.var(real,axis=0).sum(),1e-12),np.mean(distance<1e-10),
                        distance.mean(),detail['clipped_fraction'].mean()]
                np.testing.assert_allclose(row[['heldout_marginal_wasserstein','variance_relative_to_training','training_copy_fraction','mean_training_nearest_distance','mean_clipped_feature_fraction']].to_numpy(float),actual,atol=1e-10)
    models=[]
    for record in manifest['sensory_proxy_models']:
        path=Path(record['path']);assert sha256(path)==record['sha256'];models.append(joblib.load(path))
    columns=[names.index(c) for c in manifest['proxy_features']]
    def predict(values):return np.mean([m.predict(values[:,columns]) for m in models],axis=0)
    predictions=pd.read_csv(out/'generated_sensory.csv').pivot(index='generated_id',columns='target',values='prediction')
    np.testing.assert_allclose(predictions.loc[table.index,targets],predict(table[names].to_numpy(float)),atol=1e-10)
    pool=pd.read_csv(out/'transfer_candidate_pool.csv').set_index('candidate_id')
    pool_y=pd.read_csv(out/'transfer_candidate_sensory.csv').set_index('candidate_id')
    values=pool[names].to_numpy(float);pred=predict(values)
    np.testing.assert_allclose(pred,pool_y.loc[pool.index,targets],atol=1e-10)
    results=pd.read_csv(out/'transfer_results.csv',dtype={'beer':str});gates=pd.read_csv(out/'transfer_gate_counts.csv',dtype={'beer':str})
    checks=pd.read_csv(out/'transfer_constraint_diagnostics.csv',dtype={'beer':str})
    for result in results.itertuples():
        before=raw.loc[result.beer,names].to_numpy(float);missing=np.isnan(before[[names.index(c) for c in cfg['chemical_screen_fields']]]).any()
        original=g.impute_.transform(before[None,:])[0];source_y=predict(original[None,:])[0]
        chemical=values[:,[names.index(c) for c in cfg['chemical_screen_fields']]]-original[[names.index(c) for c in cfg['chemical_screen_fields']]]
        sensory=pred[:,[targets.index(c) for c in cfg['sensory_screen_targets']]]-source_y[[targets.index(c) for c in cfg['sensory_screen_targets']]]
        valid=(chemical<=1e-10).all(axis=1)&(sensory<=1e-10).all(axis=1)
        if missing:valid[:]=False;assert result.status=='source_protected_value_missing'
        assert result.n_sampled_screen_passed==int(valid.sum())
        if not valid.any():assert pd.isna(result.selected_candidate_id)
        else:assert valid[pool.index.get_loc(result.selected_candidate_id)]
        assert not result.physical_constraints_certified
        for j,field in enumerate(cfg['chemical_screen_fields']):
            r=gates.loc[gates.beer.eq(result.beer)&gates.field.eq(field)].iloc[0]
            assert r.n_passed==int((chemical[:,j]<=1e-10).sum())
        for check in checks.loc[checks.beer.eq(result.beer)].itertuples():
            index=pool.index.get_loc(check.candidate_id)
            if check.kind=='supplied_coordinate_nonincrease':
                j=names.index(check.field);b,a=original[j],values[index,j]
            else:j=targets.index(check.field);b,a=source_y[j],pred[index,j]
            np.testing.assert_allclose([check.before,check.after,check.difference],[b,a,a-b],atol=1e-10)
            assert check.screen_pass==bool(a-b<=1e-10)
    result={'status':'passed','generated_profiles_replayed':len(table),'generated_sensory_values_checked':len(table)*len(targets),
            'fold_generators_refitted':3,'validation_style_fold_method_groups_recomputed':len(validation),
            'transfer_candidate_profiles':len(pool),'transfer_sources_checked':len(results),'physical_certification':False}
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1):verify(p.parse_args().output)
