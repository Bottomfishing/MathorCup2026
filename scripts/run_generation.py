"""Q3 exploratory generation/transfer pipeline. AI-assisted: Codex, 2026-10-24."""
import argparse
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
from beer_models.baseline import load_data
from beer_models.generation import StyleGenerator,conservative_screen
from beer_audit.audit import save_csv,write_json,sha256


def run(source,audit,baseline,permutation,config_path,out):
    source=source.resolve();out=out.resolve()
    if out==source or source in out.parents:raise ValueError('Output must not be inside original data')
    config=json.loads(config_path.read_text(encoding='utf-8'))
    ids,xf,yf,primary=load_data(source,audit);x=xf.to_numpy(float);styles=ids.tasting_category_fine.to_numpy()
    features=list(xf);targets=list(yf)
    proxy_manifest=json.loads((permutation/'run_manifest.json').read_text(encoding='utf-8'))
    for name,digest in proxy_manifest['source_sha256'].items():
        if sha256(source/name)!=digest:raise ValueError('Proxy training data differs')
    if pd.read_csv(permutation/'target_order.csv').target.tolist()!=targets:raise ValueError('Proxy target order differs')
    split=pd.read_csv(baseline/'split_manifest.csv',dtype={'beer':str}).set_index('beer').loc[ids.beer]
    folds=split.outer_test_fold.to_numpy()
    out.mkdir(parents=True,exist_ok=False);write_json(out/'config.json',config);write_json(out/'status.json',{'state':'running'})
    project=Path(__file__).resolve().parents[1]
    for p in list((project/'src').rglob('*.py'))+[Path(__file__),project/'requirements-lock.txt']:
        dest=out/'source_snapshot'/p.resolve().relative_to(project);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save_csv(out,'samples.csv',ids.assign(outer_test_fold=folds))
    save_csv(out,'feature_order.csv',[{'field':c} for c in features]);save_csv(out,'target_order.csv',[{'target':c} for c in targets])
    validation=[]
    for fold in sorted(set(folds)):
        train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
        g=StyleGenerator(config['latent_components'],config['covariance_prior_sample_size']).fit(x[train],styles[train])
        assert len(g.x_)==len(train)
        np.testing.assert_allclose(g.impute_.statistics_,np.nanmedian(x[train],axis=0))
        for si,style in enumerate(sorted(set(styles))):
            test_idx=test[styles[test]==style];train_idx=g.indices_[style]
            hold=g.robust_.transform(g.impute_.transform(x[test_idx]))
            real=g.robust_.transform(g.x_[train_idx])
            for mi,method in enumerate(config['methods']):
                generated,meta=g.sample(style,config['validation_samples_per_style'],method,config['seed']+int(fold)*10000+si*100+mi)
                z=g.robust_.transform(generated)
                nearest=cdist(z,real).min(axis=1)/np.sqrt(x.shape[1])
                validation.append({'fold':int(fold),'style':style,'method':method,'n_train_style':len(train_idx),'n_test_style':len(test_idx),
                                   'heldout_marginal_wasserstein':float(np.mean([wasserstein_distance(hold[:,j],z[:,j]) for j in range(x.shape[1])])),
                                   'variance_relative_to_training':float(np.var(z,axis=0).sum()/max(np.var(real,axis=0).sum(),1e-12)),
                                   'training_copy_fraction':float((nearest<1e-10).mean()),'mean_training_nearest_distance':float(nearest.mean()),
                                   'mean_clipped_feature_fraction':float(meta['clipped_fraction'].mean())})
        print(f'Generation validation fold {fold} complete',flush=True)
    validation=pd.DataFrame(validation);save_csv(out,'validation_by_style_fold.csv',validation)
    save_csv(out,'validation_summary.csv',validation.groupby('method',sort=False).agg(
        mean_heldout_wasserstein=('heldout_marginal_wasserstein','mean'),
        mean_variance_ratio=('variance_relative_to_training','mean'),mean_copy_fraction=('training_copy_fraction','mean'),
        mean_clipped_fraction=('mean_clipped_feature_fraction','mean')).reset_index())
    g=StyleGenerator(config['latent_components'],config['covariance_prior_sample_size']).fit(x,styles)
    joblib.dump(g,out/'generator.joblib',compress=3)
    # Use local saved 219-input forest ensemble only as a sensory proxy, never as validation of generation.
    proxy_features=pd.read_csv(permutation/'analytes_only/feature_order.csv').field.tolist()
    columns=[features.index(c) for c in proxy_features];models=[];model_records=[]
    for fit in pd.read_csv(permutation/'analytes_only/fold_models.csv').itertuples():
        path=Path(fit.model_path)
        if sha256(path)!=fit.model_sha256:raise ValueError('Sensory proxy model hash mismatch')
        models.append(joblib.load(path));model_records.append({'path':str(path),'sha256':fit.model_sha256})
    def predict(values):return np.mean([m.predict(values[:,columns]) for m in models],axis=0)
    generated_rows=[];metadata=[]
    for si,style in enumerate(config['generation_styles']):
        for mi,method in enumerate(config['methods']):
            seed=config['seed']+80000+si*100+mi
            values,meta=g.sample(style,config['generation_samples_per_style_method'],method,seed)
            for j,row in enumerate(values):
                name=f'{style}_{method}_{j:04d}'
                generated_rows.append({'generated_id':name,**dict(zip(features,row))})
                parents=meta['parents'][j]
                metadata.append({'generated_id':name,'style':style,'method':method,'seed':seed,'parent_1':ids.beer.iloc[parents[0]] if parents[0]>=0 else '',
                                 'parent_2':ids.beer.iloc[parents[1]] if parents[1]>=0 else '', 'parent_1_weight':meta['weights'][j],
                                 'clipped_feature_fraction':meta['clipped_fraction'][j],'value_scale':'supplied_1csv_coordinates',
                                 'aggregate_identities_verified':False,'physical_feasibility_certified':False})
    generated_frame=pd.DataFrame(generated_rows);save_csv(out,'generated_profiles.csv',generated_frame);save_csv(out,'generated_metadata.csv',metadata)
    generated_sensory=predict(generated_frame[features].to_numpy(float))
    save_csv(out,'generated_sensory.csv',[{'generated_id':name,'target':target,'prediction':generated_sensory[i,t],
                                         'scale':'standardized_4csv_target','scope':'saved_forest_ensemble_estimate_not_observed_rating'}
                                        for i,name in enumerate(generated_frame.generated_id) for t,target in enumerate(targets)])
    # Finite candidate search: infeasibility here means no sampled candidate passed, not a proof of global infeasibility.
    pools=[];pool_methods=[]
    for mi,method in enumerate(config['transfer_methods']):
        values,_=g.sample(config['transfer_target_style'],config['transfer_candidates_per_method'],method,config['seed']+90000+mi)
        pools.append(values);pool_methods.extend([method]*len(values))
    pool=np.concatenate(pools);proxy=predict(pool)
    pool_ids=[f'transfer_candidate_{i:04d}' for i in range(len(pool))]
    save_csv(out,'transfer_candidate_pool.csv',pd.DataFrame(pool,columns=features).assign(candidate_id=pool_ids,method=pool_methods))
    save_csv(out,'transfer_candidate_sensory.csv',pd.DataFrame(proxy,columns=targets).assign(candidate_id=pool_ids))
    chemical_indices=[features.index(c) for c in config['chemical_screen_fields']]
    sensory_indices=[targets.index(c) for c in config['sensory_screen_targets']]
    zpool=g.robust_.transform(pool);target_center=g.robust_.transform(g.x_[g.indices_[config['transfer_target_style']]]).mean(axis=0)
    target_distance=np.sqrt(np.mean((zpool-target_center)**2,axis=1))
    transfers=[];checks=[];gate_counts=[]
    for i in np.flatnonzero(styles==config['transfer_source_style']):
        source_x=g.impute_.transform(x[[i]])[0];source_y=predict(source_x[None,:])[0]
        ok,dc,ds=conservative_screen(pool,source_x,proxy,source_y,chemical_indices,sensory_indices)
        missing_protected=bool(np.isnan(x[i,chemical_indices]).any())
        if missing_protected:ok[:]=False
        gate_counts.extend({'beer':ids.beer.iloc[i],'field':field,'kind':'chemical','n_passed':int((dc[:,j]<=1e-10).sum()),'n_candidates':len(pool)}
                           for j,field in enumerate(config['chemical_screen_fields']))
        gate_counts.extend({'beer':ids.beer.iloc[i],'field':target,'kind':'sensory_proxy','n_passed':int((ds[:,j]<=1e-10).sum()),'n_candidates':len(pool)}
                           for j,target in enumerate(config['sensory_screen_targets']))
        gate_counts.append({'beer':ids.beer.iloc[i],'field':'all_chemical','kind':'joint','n_passed':int((dc<=1e-10).all(axis=1).sum()),'n_candidates':len(pool)})
        gate_counts.append({'beer':ids.beer.iloc[i],'field':'all_sensory','kind':'joint','n_passed':int((ds<=1e-10).all(axis=1).sum()),'n_candidates':len(pool)})
        change=np.sqrt(np.mean((zpool-g.robust_.transform(source_x[None,:])[0])**2,axis=1))
        weight=config['transfer_target_distance_weight'];objective=weight*target_distance+(1-weight)*change
        if ok.any():chosen=int(np.flatnonzero(ok)[np.argmin(objective[ok])]);status='screen_passed_physical_unverified'
        else:chosen=int(np.argmin(objective));status='source_protected_value_missing' if missing_protected else 'no_sampled_candidate_passed'
        transfers.append({'beer':ids.beer.iloc[i],'source_style':styles[i],'target_style':config['transfer_target_style'],
                          'selected_candidate_id':pool_ids[chosen] if ok.any() else '',
                          'diagnostic_candidate_id':pool_ids[chosen],'candidate_method':pool_methods[chosen],
                          'status':status,'n_sampled_screen_passed':int(ok.sum()),'physical_constraints_certified':False,
                          'source_missing_screen_fields':int(np.isnan(x[i,chemical_indices]).sum()),
                          'diagnostic_target_distance':target_distance[chosen],'diagnostic_change_distance':change[chosen],
                          'source_target_distance':float(np.sqrt(np.mean((g.robust_.transform(source_x[None,:])[0]-target_center)**2))),
                          'objective':objective[chosen]})
        for j,field in enumerate(config['chemical_screen_fields']):
            checks.append({'beer':ids.beer.iloc[i],'candidate_id':pool_ids[chosen],'field':field,'kind':'supplied_coordinate_nonincrease',
                           'before':source_x[chemical_indices[j]],'after':pool[chosen,chemical_indices[j]],'difference':dc[chosen,j],
                           'screen_pass':bool(dc[chosen,j]<=1e-10),'physical_threshold_status':'unverified_not_certified'})
        for j,target in enumerate(config['sensory_screen_targets']):
            checks.append({'beer':ids.beer.iloc[i],'candidate_id':pool_ids[chosen],'field':target,'kind':'model_estimated_sensory_nonincrease',
                           'before':source_y[sensory_indices[j]],'after':proxy[chosen,sensory_indices[j]],'difference':ds[chosen,j],
                           'screen_pass':bool(ds[chosen,j]<=1e-10),'physical_threshold_status':'proxy_not_physical_threshold'})
    save_csv(out,'transfer_results.csv',transfers);save_csv(out,'transfer_constraint_diagnostics.csv',checks)
    save_csv(out,'transfer_gate_counts.csv',gate_counts)
    write_json(out/'run_manifest.json',{'source':str(source),'audit':str(audit.resolve()),'baseline':str(baseline.resolve()),
        'source_sha256':{name:sha256(source/name) for name in ['1.csv','4.csv','8.xlsx']},
        'generator_sha256':sha256(out/'generator.joblib'),'sensory_proxy_models':model_records,'proxy_features':proxy_features,
        'config_sha256':sha256(config_path),'limitations':[
            'Validation uses the original three folds; exploratory configurations were not selected on an independent final holdout.',
            'All profiles use supplied coordinates, mixing log-transformed and unresolved scales. Aggregate identities and physical units are not certified.',
            'Interpolation contracts variance; latent Gaussian clipping changes its distribution and must be reported.',
            'Sensory outputs are ensemble predictions in standardized space. They are not real ratings or independent validation.',
            'Transfer tests conservative data-coordinate and predicted-sensory nonincrease; no physical threshold certification.',
            'Missing protected input values are imputed; such cases cannot establish original-sample nonincrease.',
            'Finite candidate failure is not proof of mathematical infeasibility. No constraints are relaxed.']})
    write_json(out/'status.json',{'state':'completed','generated_rows':len(generated_frame),'transfer_sources':len(transfers),
                                'physical_constraints_certified':False})
    print(pd.read_csv(out/'validation_summary.csv').to_string(index=False));print(pd.DataFrame(transfers)[['beer','status','n_sampled_screen_passed']].to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','audit','baseline','permutation','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('configs/q3_exploratory_v1.json'))
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.audit,a.baseline,a.permutation,a.config,a.output)
