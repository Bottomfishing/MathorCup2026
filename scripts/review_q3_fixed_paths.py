"""Refit guards on paired bootstrap samples; test only previously frozen task3 paths."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv,write_json,sha256
from beer_models.baseline import load_data
from beer_models.final_model import estimator
from beer_models.style_models import make_stale_model
from beer_models.explain import stratified_draw
from beer_models.review_diagnostics import change_summary


def run(source,output):
    cp=Path('configs/review_followup_20261025.json');config=json.loads(cp.read_text())
    final=json.loads(Path('configs/final_protocol.json').read_text())
    style=json.loads(Path('configs/q3_style_protocol.json').read_text())
    ids,xf,yf,_=load_data(source,Path('outputs/p0_audit_20261023_final'))
    columns=[f for f in xf if f not in final['excluded_derived_features']]
    x=xf[columns].to_numpy(float);y=yf.to_numpy(float);target=yf.columns.get_loc('stale_hops')
    delivery=Path('outputs/q3_task3_20261025')
    before=pd.read_csv(delivery/'transfer_source_profiles.csv',dtype={'beer':str}).set_index('beer')
    after=pd.read_csv(delivery/'transfer_profiles.csv',dtype={'beer':str}).set_index('beer').loc[before.index]
    if list(before)!=columns or list(after)!=columns:raise ValueError('Fixed path feature order differs')
    if not np.isfinite(before.to_numpy()).all() or not np.isfinite(after.to_numpy()).all():raise ValueError('Incomplete fixed paths')
    output.mkdir(parents=True,exist_ok=False);write_json(output/'protocol.json',config)
    save_csv(output,'frozen_before.csv',before.copy().reset_index());save_csv(output,'frozen_after.csv',after.copy().reset_index())
    rng=np.random.default_rng(config['seed']);rows=[];oob=[];draws=[]
    for repeat in range(config['bootstrap_repeats']):
        draw=stratified_draw(ids.tasting_category_fine,rng);draws.append(draw)
        held=np.setdiff1d(np.arange(len(x)),draw)
        seed=config['seed']+repeat
        models={'multioutput':estimator('extra_trees',final,seed),'ridge':estimator('ridge_asinh',final,seed),
                'focused':make_stale_model(style,seed)}
        for name,model in models.items():
            model.fit(x[draw],y[draw,target] if name=='focused' else y[draw])
            np.testing.assert_allclose(model.named_steps['impute'].statistics_,np.nanmedian(x[draw],axis=0))
            b=model.predict(before.to_numpy(float));a=model.predict(after.to_numpy(float));h=model.predict(x[held])
            if name!='focused':b=b[:,target];a=a[:,target];h=h[:,target]
            for k,beer in enumerate(before.index):
                rows.append(dict(repeat=repeat,model=name,beer=beer,before=b[k],after=a[k],change=a[k]-b[k],
                    passes=bool(a[k]-b[k]<=config['nonincrease_tolerance'])))
            oob.extend(dict(repeat=repeat,model=name,beer=ids.beer.iloc[i],observed=y[i,target],prediction=h[k],
                median_baseline=float(np.median(y[draw,target]))) for k,i in enumerate(held))
        if (repeat+1)%10==0:print(f'Fixed-path Q3 refits {repeat+1}/{config["bootstrap_repeats"]}',flush=True)
    save_csv(output,'bootstrap_draws.csv',pd.DataFrame(draws));rf=pd.DataFrame(rows);save_csv(output,'refit_changes.csv',rf)
    save_csv(output,'oob_predictions.csv',oob)
    summaries=[]
    for name,g in rf.groupby('model'):
        matrix=g.pivot(index='repeat',columns='beer',values='change').reindex(columns=before.index)
        summary=change_summary(matrix.to_numpy(),config['nonincrease_tolerance'])
        summaries.extend(dict(model=name,beer=beer,**{key:value[k] for key,value in summary.items()}) for k,beer in enumerate(before.index))
    save_csv(output,'change_summary.csv',summaries)
    joint=rf.groupby(['repeat','beer']).passes.all().unstack('beer').reindex(columns=before.index)
    save_csv(output,'joint_pass_fraction.csv',[dict(beer=beer,all_three_pass_fraction=joint[beer].mean(),refits=len(joint)) for beer in before.index])
    of=pd.DataFrame(oob);metrics=[]
    for (repeat,name),g in of.groupby(['repeat','model']):
        metrics.append(dict(repeat=repeat,model=name,n=len(g),mae=abs(g.observed-g.prediction).mean(),
            median_mae=abs(g.observed-g.median_baseline).mean()))
    save_csv(output,'oob_metrics.csv',metrics)
    write_json(output/'summary.json',dict(status='completed',config_sha256=sha256(cp),protocol=config,
        source_sha256={n:sha256(source/n) for n in ['1.csv','4.csv','8.xlsx']},
        frozen_inputs_sha256={n:sha256(delivery/n) for n in ['transfer_source_profiles.csv','transfer_profiles.csv','transfer_results.csv']},
        paths=len(before),refits=config['bootstrap_repeats'],models_per_refit=3,candidate_search_performed=False,
        original_card_score_recovered=False,actual_after_stale_validated=False,
        limitation='Paired refits of existing observed data and newly seeded trees, conditional on fixed historical candidate selection. OOB errors diagnose prediction only; refit percentiles are not intervention confidence intervals. No style or physical authentication.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.output)
