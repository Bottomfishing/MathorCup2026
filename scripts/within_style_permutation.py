"""Exploratory within-style permutation control. AI-assisted: Codex, 2026-10-24."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_models.permutation import permutation_deltas,validation_bootstrap
from beer_audit.audit import save_csv,write_json,sha256


def run(parent,out):
    manifest=json.loads((parent/'run_manifest.json').read_text(encoding='utf-8'))
    source=Path(manifest['source']); folder=parent/'analytes_only'
    for name,digest in manifest['source_sha256'].items():
        if sha256(source/name)!=digest:raise ValueError('Source hash mismatch')
    samples=pd.read_csv(parent/'samples.csv',dtype={'beer':str})
    names=pd.read_csv(folder/'feature_order.csv').field.tolist()
    targets=pd.read_csv(parent/'target_order.csv').target.tolist()
    x=pd.read_csv(source/'1.csv',dtype={'beer':str}).set_index('beer').loc[samples.beer,names].to_numpy(float)
    y=pd.read_csv(source/'4.csv',dtype={'beer':str}).set_index('beer').loc[samples.beer,targets].to_numpy(float)
    config=json.loads((parent/'config.json').read_text(encoding='utf-8'))
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'status.json',{'state':'running'})
    delta=np.full((len(x),len(names),len(targets)),np.nan);diagnostics=[]
    for fit in pd.read_csv(folder/'fold_models.csv').itertuples():
        path=Path(fit.model_path)
        if sha256(path)!=fit.model_sha256:raise ValueError('Model hash mismatch')
        model=joblib.load(path);test=np.flatnonzero(samples.outer_test_fold.to_numpy()==fit.fold)
        styles=samples.tasting_category_fine.to_numpy()[test]
        print(f'within-style fold {fit.fold}',flush=True)
        d,r,orders=permutation_deltas(model,x[test],y[test],[[j] for j in range(len(names))],
                                    config['permutation_repeats'],config['seed']+int(fit.fold)*1000,
                                    config['prediction_batch_groups'],strata=styles)
        delta[test]=d
        for order in orders:
            assert np.array_equal(styles[order],styles)
            assert np.array_equal(np.sort(order),np.arange(len(test)))
        np.save(out/f'orders_fold{fit.fold}.npy',orders)
        np.save(out/f'repeat_means_fold{fit.fold}.npy',r)
        diagnostics.append({'fold':fit.fold,'n_test':len(test),'singleton_style_rows':int(pd.Series(styles).map(pd.Series(styles).value_counts()).eq(1).sum()),
                            'mean_unchanged_fraction':float((orders==np.arange(len(test))).mean())})
    assert np.isfinite(delta).all()
    means,counts,draws=validation_bootstrap(delta,samples.tasting_category_fine,config['validation_bootstrap_repeats'],config['seed']+9000)
    np.save(out/'individual_sample_deltas.npy',delta);np.save(out/'bootstrap_means.npy',means);np.save(out/'bootstrap_draws.npy',draws)
    average=delta.mean(axis=0);lo,hi=np.quantile(means,[.025,.975],axis=0)
    rows=[];top=[]
    for t,target in enumerate(targets):
        rankmap={j:k+1 for k,j in enumerate(np.argsort(-average[:,t],kind='stable')[:3])}
        for j,name in enumerate(names):
            row={'target':target,'field':name,'delta_mae':average[j,t],
                 'validation_resample_top3_frequency':counts[j,t]/len(draws),
                 'fixed_model_resample_p025':lo[j,t],'fixed_model_resample_p975':hi[j,t]}
            rows.append(row)
            if j in rankmap:top.append(row|{'rank':rankmap[j]})
    save_csv(out,'importance.csv',rows);save_csv(out,'top3.csv',top);save_csv(out,'permutation_diagnostics.csv',diagnostics)
    write_json(out/'run_manifest.json',{'parent':str(parent.resolve()),'source_sha256':manifest['source_sha256'],
               'config':config,'script_sha256':sha256(Path(__file__)),
               'permutation_module_sha256':sha256(Path(__file__).resolve().parents[1]/'src/beer_models/permutation.py'),
               'limitation':'Within-style unconditional permutations; small held-out style cells and frozen singleton cells reduce perturbation. Not causal adjustment.'})
    write_json(out/'status.json',{'state':'completed'})
    print(pd.DataFrame(diagnostics).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.parent,a.output)
