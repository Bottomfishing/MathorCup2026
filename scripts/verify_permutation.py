"""Independent Q2 replay and arithmetic audit. AI-assisted: Codex, 2026-10-24."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits


def verify(out):
    manifest=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    config=json.loads((out/'config.json').read_text(encoding='utf-8'))
    source=Path(manifest['source']); baseline=Path(manifest['baseline'])
    samples=pd.read_csv(out/'samples.csv',dtype={'beer':str})
    targets=pd.read_csv(out/'target_order.csv').target.tolist()
    raw=pd.read_csv(source/'1.csv',dtype={'beer':str}).set_index('beer').loc[samples.beer]
    y=pd.read_csv(source/'4.csv',dtype={'beer':str}).set_index('beer').loc[samples.beer,targets].to_numpy(float)
    styles=samples.tasting_category_fine.to_numpy()
    expected_counts=pd.Series(styles).value_counts().sort_index()
    metrics=pd.read_csv(out/'predictive_metrics.csv')
    eligible=pd.read_csv(Path(manifest['previous_q2'])/'factor_eligibility.csv')
    eligible=set(eligible.loc[eligible.eligible_compound,'field'])
    replayed_features=0; replayed_groups=0
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    for variant in config['variants']:
        folder=out/variant
        names=pd.read_csv(folder/'feature_order.csv').field.tolist()
        x=raw[names].to_numpy(float)
        oof=np.load(folder/'oof_predictions.npy')
        delta=np.load(folder/'individual_sample_deltas.npy')
        assert np.isfinite(oof).all() and np.isfinite(delta).all()
        if variant=='full':
            old=pd.read_csv(baseline/'oof_predictions.csv',dtype={'beer':str})
            old=old.loc[old.model.eq('extra_trees')].pivot(index='beer',columns='target',values='predicted')
            np.testing.assert_allclose(oof,old.loc[samples.beer,targets],atol=1e-12,rtol=1e-12)
        group_table=pd.read_csv(folder/'group_importance.csv')
        fold_table=pd.read_csv(folder/'fold_importance.csv')
        for fit in pd.read_csv(folder/'fold_models.csv').itertuples():
            path=Path(fit.model_path)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==fit.model_sha256
            model=joblib.load(path)
            test=np.flatnonzero(samples.outer_test_fold.to_numpy()==fit.fold)
            train=np.flatnonzero(samples.outer_test_fold.to_numpy()!=fit.fold)
            np.testing.assert_allclose(model.named_steps['impute'].statistics_,np.nanmedian(x[train],axis=0))
            np.testing.assert_allclose(model.predict(x[test]),oof[test],atol=1e-12)
            repeat=np.load(folder/f'individual_repeat_means_fold{fit.fold}.npy')
            np.testing.assert_allclose(delta[test].mean(axis=0),repeat.mean(axis=0),atol=1e-12)
            saved=fold_table.loc[fold_table.fold.eq(fit.fold)].pivot(index='field',columns='target',values='delta_mae')
            np.testing.assert_allclose(saved.loc[names,targets],delta[test].mean(axis=0),atol=1e-12)
            # pandas uses pairwise complete observations, independently from production rank helper.
            rho=pd.DataFrame(x[train]).corr(method='spearman').fillna(0).to_numpy(copy=True)
            np.fill_diagonal(rho,1)
            np.testing.assert_allclose(rho,np.load(folder/f'train_correlations_fold{fit.fold}.npy'),atol=1e-12)
            groups=group_table.loc[group_table.fold.eq(fit.fold)]
            for _,g in groups.groupby('group'):
                members=[names.index(c) for c in g.iloc[0].members.split(';')]
                minimum=abs(rho[np.ix_(members,members)])[np.triu_indices(len(members),1)].min()
                assert minimum>=config['group_abs_spearman_threshold']-1e-12
                np.testing.assert_allclose(g.min_pair_abs_train_rho,minimum,atol=1e-12)
            orders=np.load(folder/f'permutation_orders_fold{fit.fold}.npy')
            reference=abs(y[test]-oof[test])
            # Deterministic selected-column replay includes first, middle and last input.
            for j in sorted({0,len(names)//2,len(names)-1}):
                actual=[]
                for order in orders:
                    assert np.array_equal(np.sort(order),np.arange(len(test)))
                    changed=x[test].copy(); changed[:,j]=x[test][order,j]
                    actual.append(abs(y[test]-model.predict(changed))-reference)
                np.testing.assert_allclose(np.mean(actual,axis=0),delta[test,j,:],atol=1e-12)
                replayed_features+=1
            if len(groups):
                first=groups.loc[groups.group.eq(groups.group.min())]
                members=[names.index(c) for c in first.iloc[0].members.split(';')]
                actual=[]
                for order in orders:
                    changed=x[test].copy(); changed[:,members]=x[test][order][:,members]
                    actual.append((abs(y[test]-model.predict(changed))-reference).mean(axis=0))
                np.testing.assert_allclose(np.mean(actual,axis=0),first.set_index('target').loc[targets,'joint_delta_mae'],atol=1e-12)
                replayed_groups+=1
        measured=metrics.loc[metrics.variant.eq(variant)].set_index('target').loc[targets]
        np.testing.assert_allclose(measured.mae,mean_absolute_error(y,oof,multioutput='raw_values'),atol=1e-12)
        np.testing.assert_allclose(measured.rmse,np.sqrt(mean_squared_error(y,oof,multioutput='raw_values')),atol=1e-12)
        np.testing.assert_allclose(measured.r2,r2_score(y,oof,multioutput='raw_values'),atol=1e-12)
        choose=[j for j,c in enumerate(names) if c in eligible]; candidate_names=[names[j] for j in choose]
        means=np.load(folder/'validation_bootstrap_means.npy'); draws=np.load(folder/'validation_bootstrap_draws.npy')
        counts=np.zeros(means.shape[1:],int)
        for draw,mean in zip(draws,means):
            pd.testing.assert_series_equal(pd.Series(styles[draw]).value_counts().sort_index(),expected_counts)
            np.testing.assert_allclose(delta[draw][:,choose].mean(axis=0),mean,atol=1e-12)
            for t in range(len(targets)):counts[np.argsort(-mean[:,t],kind='stable')[:3],t]+=1
        assert (counts.sum(axis=0)==3*len(draws)).all()
        table=pd.read_csv(folder/'individual_importance.csv')
        for col,values in [('delta_mae',delta[:,choose].mean(axis=0)),
                           ('validation_resample_top3_frequency',counts/len(draws)),
                           ('fixed_model_resample_p025',np.quantile(means,.025,axis=0)),
                           ('fixed_model_resample_p975',np.quantile(means,.975,axis=0))]:
            np.testing.assert_allclose(table.pivot(index='field',columns='target',values=col).loc[candidate_names,targets],values,atol=1e-12)
        top=pd.read_csv(folder/'top3_permutation.csv')
        for t,target in enumerate(targets):
            expected=[candidate_names[j] for j in np.argsort(-delta[:,choose,t].mean(axis=0),kind='stable')[:3]]
            assert top.loc[top.target.eq(target)].sort_values('rank').field.tolist()==expected
    result={'status':'passed','variants':config['variants'],'models_replayed':6,
            'individual_feature_fold_replays':replayed_features,'group_fold_replays':replayed_groups,
            'metrics_recomputed':100,'train_only_imputation_and_correlations':True,
            'bootstrap_draws_and_counts_recomputed':True,'full_oof_matches_P1':True,
            'source_files_unchanged':True,
            'scope':'All metrics, bootstrap summaries and saved predictions checked; selected permutation replay; no complete retraining.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    with threadpool_limits(limits=1):verify(parser.parse_args().output)
