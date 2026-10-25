"""Paired fixed-alpha input-scope comparison; preserve the published main ranking."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import networkx as nx
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.baseline import load_data, make_pipeline, make_splits, metric_vectors
from beer_models.explain import stratified_draw, top_indices
from beer_models.review_diagnostics import set_overlap


def run(source, output):
    cp = Path('configs/review_followup_20261025.json')
    config = json.loads(cp.read_text())
    prior = json.loads(Path('configs/q2_exploratory_v1.json').read_text())
    ids, xf, yf, primary = load_data(source, Path('outputs/p0_audit_20261023_final'))
    styles = ids.tasting_category_fine.to_numpy(); y = yf.to_numpy(float)
    strict = [c for c in xf if c not in prior['exclude_from_compound_top3']]
    scopes = {'all231': list(xf), 'strict219': strict}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', config)
    def fit(columns, draw):
        model, _ = make_pipeline('ridge_asinh', {'ridge_alphas':[config['ridge_alpha']]}, config['seed'])
        return model.set_params(model__alpha=config['ridge_alpha']).fit(xf[columns].to_numpy(float)[draw], y[draw])
    graph = nx.Graph(); graph.add_nodes_from(strict)
    edges = pd.read_csv('outputs/q2_task4_20261025/network_edges_0_6.csv')
    graph.add_edges_from((r.source, r.target) for r in edges.itertuples() if r.source in strict and r.target in strict)
    components = {field: k for k, group in enumerate(sorted(nx.connected_components(graph), key=lambda g:min(g))) for field in group}
    save_csv(output, 'network_groups.csv', [{'field':f, 'component':components[f], 'component_size':len(nx.node_connected_component(graph,f))} for f in strict])
    coefficients = {}; selections = []; main = {}; cv = []; predictions = []; partitions = []
    for name, columns in scopes.items():
        model = fit(columns, np.arange(len(xf))); coefficients[name] = model.named_steps['model'].coef_
        eligible = [columns.index(c) for c in strict]
        main[name] = {t:[columns[k] for k in top_indices(coefficients[name][j], eligible)] for j,t in enumerate(yf)}
    # Confirm the original coefficient context before reporting its comparison.
    old = pd.read_csv('outputs/q2_exploratory_20261023_v1/top3_candidates.csv')
    for row in old.itertuples():
        np.testing.assert_allclose(coefficients['all231'][yf.columns.get_loc(row.target), xf.columns.get_loc(row.field)], row.ridge_signed_coef, atol=1e-12, rtol=0)
    rng = np.random.default_rng(config['seed']); draws = []
    for repeat in range(config['bootstrap_repeats']):
        draw = stratified_draw(styles, rng); draws.append(draw)
        for name, columns in scopes.items():
            coef = fit(columns, draw).named_steps['model'].coef_
            eligible = [columns.index(c) for c in strict]
            for j,t in enumerate(yf):
                for rank,k in enumerate(top_indices(coef[j],eligible),1):
                    selections.append(dict(repeat=repeat, scope=name, target=t, rank=rank, field=columns[k], coefficient=coef[j,k], component=components[columns[k]]))
        if (repeat+1)%50==0: print(f'Paired Q2 refits {repeat+1}',flush=True)
    save_csv(output,'bootstrap_draws.csv',pd.DataFrame(draws))
    boots = pd.DataFrame(selections); save_csv(output,'bootstrap_selections.csv',boots)
    top = []; comparison = []
    for j,t in enumerate(yf):
        comparison.append(dict(target=t,primary=t in primary,top3_jaccard=set_overlap(main['all231'][t],main['strict219'][t]),
            component_jaccard=set_overlap([components[f] for f in main['all231'][t]],[components[f] for f in main['strict219'][t]]),
            all231=';'.join(main['all231'][t]),strict219=';'.join(main['strict219'][t])))
        for name, columns in scopes.items():
            selected = boots.loc[boots.scope.eq(name)&boots.target.eq(t)]
            for rank,f in enumerate(main[name][t],1):
                same = selected.loc[selected.field.eq(f)]
                component_count = selected.loc[selected.component.eq(components[f]),'repeat'].nunique()
                top.append(dict(scope=name,target=t,rank=rank,field=f,coefficient=coefficients[name][j,columns.index(f)],
                    frequency=len(same)/config['bootstrap_repeats'],component=components[f],
                    component_any_selected_frequency=component_count/config['bootstrap_repeats']))
    save_csv(output,'top3_stability.csv',top); save_csv(output,'scope_comparison.csv',comparison)
    for seed in config['split_seeds']:
        for fold,(train,test) in enumerate(make_splits(styles,config['folds'],seed)):
            partitions.extend(dict(seed=seed,fold=fold,beer=ids.beer.iloc[i],row=int(i)) for i in test)
            for name,columns in scopes.items():
                model=fit(columns,train); pred=model.predict(xf[columns].to_numpy(float)[test])
                # Every preprocessing stage was fitted within train only; no scope selection.
                np.testing.assert_allclose(model.named_steps['impute'].statistics_, np.nanmedian(xf[columns].to_numpy(float)[train],axis=0))
                metrics=metric_vectors(y[test],pred)
                cv.extend(dict(seed=seed,fold=fold,scope=name,target=t,primary=t in primary,**{k:v[j] for k,v in metrics.items()}) for j,t in enumerate(yf))
                predictions.extend(dict(seed=seed,fold=fold,scope=name,beer=ids.beer.iloc[i],target=t,observed=y[i,j],prediction=pred[k,j]) for k,i in enumerate(test) for j,t in enumerate(yf))
    save_csv(output,'splits.csv',partitions);save_csv(output,'fold_metrics.csv',cv)
    pf=pd.DataFrame(predictions);save_csv(output,'oof_predictions.csv',pf)
    pooled=[]
    for (seed,name),g in pf.groupby(['seed','scope']):
        for scope,targets in [('primary45',primary),('all50',list(yf))]:
            selected=g.loc[g.target.isin(targets)]
            pooled.append(dict(seed=seed,scope=name,targets=scope,mae=float(abs(selected.observed-selected.prediction).mean())))
    save_csv(output,'pooled_metrics.csv',pooled)
    write_json(output/'summary.json',dict(status='completed',config_sha256=sha256(cp),source_sha256={n:sha256(source/n) for n in ['1.csv','4.csv','8.xlsx']},
        targets_changed=sum(r['top3_jaccard']<1 for r in comparison),paired_refits=config['bootstrap_repeats'],
        protocol=config,main_results_replaced=False,network_used_for_prediction=False,
        limitation='Components are transitive descriptive correlation groups, not interchangeable chemicals or mechanisms. Fixed-alpha repeated CV does not retune or independently confirm a winning scope.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.output)
