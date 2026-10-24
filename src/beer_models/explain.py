"""Q2 exploratory network and bootstrap association evidence.

AI-assisted: OpenAI Codex (GPT-6), OpenAI, 2026-10-23.
Full-data explanations are descriptive; no predictive validation claims.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv,write_json,sha256
from beer_audit.checks import repair_utf8_mojibake,square_matrix_checks
from .baseline import load_data,make_pipeline


def top_indices(values,eligible,k=3):
    values=np.asarray(values,float)
    eligible=np.asarray(eligible,int)
    if len(eligible)<k or not np.isfinite(values[eligible]).all():
        raise ValueError('Insufficient eligible finite factors')
    return eligible[np.argsort(-abs(values[eligible]),kind='stable')[:k]]


def stratified_draw(styles,rng):
    styles=np.asarray(styles)
    return np.concatenate([rng.choice(np.flatnonzero(styles==s),size=int((styles==s).sum()),replace=True)
                           for s in sorted(set(styles))])


def network_summary(matrix,threshold):
    a=np.asarray(matrix,float)
    if not 0<threshold<=1:raise ValueError('Threshold out of range')
    adjacency=abs(a)>=threshold
    np.fill_diagonal(adjacency,False)
    degree=adjacency.sum(axis=1)
    unseen=set(range(len(a)));sizes=[]
    while unseen:
        todo=[min(unseen)];unseen.remove(todo[0]);size=0
        while todo:
            i=todo.pop();size+=1
            found=set(np.flatnonzero(adjacency[i]))&unseen
            unseen-=found;todo.extend(sorted(found))
        sizes.append(size)
    i,j=np.where(np.triu(adjacency,k=1))
    return {'threshold':threshold,'nodes':len(a),'edges':len(i),
            'positive_edges':int((a[i,j]>0).sum()),'negative_edges':int((a[i,j]<0).sum()),
            'isolated_nodes':int((degree==0).sum()),'components':len(sizes),'largest_component':max(sizes)},degree,i,j


def marginal_spearman(x,y):
    """Rank both sides only after dropping missing observations for each input."""
    out=np.zeros((y.shape[1],x.shape[1]))
    for j in range(x.shape[1]):
        valid=np.isfinite(x[:,j])
        if valid.sum()<3:raise ValueError('Too few observations for marginal correlation')
        a=pd.Series(x[valid,j]).rank().to_numpy()
        b=pd.DataFrame(y[valid]).rank().to_numpy()
        a=a-a.mean();b=b-b.mean(axis=0)
        den=np.sqrt((a*a).sum()*(b*b).sum(axis=0))
        out[:,j]=np.divide(a@b,den,out=np.zeros(y.shape[1]),where=den>0)
    return out


def run(source,audit,config_path,output):
    config=json.loads(config_path.read_text(encoding='utf-8'))
    source,audit,output=source.resolve(),audit.resolve(),output.resolve()
    if source==output or source in output.parents:raise ValueError('Output inside original sources')
    ids,xf,yf,primary=load_data(source,audit)
    inventory=pd.read_csv(audit/'data_inventory.csv')
    expected=inventory.loc[inventory.file.eq('2.csv'),'sha256'].iloc[0]
    if sha256(source/'2.csv')!=expected:raise ValueError('Network source changed since audit')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',config)
    write_json(output/'status.json',{'state':'running','scope':config['scope']})
    metadata=pd.read_csv(audit/'feature_dictionary.csv').set_index('field_name_R')
    aliases=metadata['chemical_name'].map(repair_utf8_mojibake)
    if aliases.duplicated().any():raise ValueError('Ambiguous compound alias')
    network=pd.read_csv(source/'2.csv',index_col=0)
    checks=square_matrix_checks(network)
    if not checks['same_ordered_axes'] or checks['nonfinite'] or checks['out_of_bounds'] or checks['symmetry_max_abs_error']>1e-10:
        raise ValueError('Invalid correlation network')
    if set(aliases)!=set(network.columns):raise ValueError('Network aliases do not fully match')
    reverse={v:k for k,v in aliases.items()}
    network.index=[reverse[c] for c in network.index];network.columns=[reverse[c] for c in network.columns]
    network=network.loc[xf.columns,xf.columns]
    graph_rows=[];node_rows=[]
    for threshold in config['network_thresholds']:
        summary,degree,i,j=network_summary(network.to_numpy(),threshold)
        graph_rows.append(summary)
        save_csv(output,f'network_edges_{threshold:.1f}.csv',
                 [{'source':xf.columns[a],'target':xf.columns[b],'spearman_rho':network.iloc[a,b]}
                  for a,b in zip(i,j)])
        node_rows.extend({'field':c,'threshold':threshold,'degree':int(degree[k]),
                          'degree_centrality':float(degree[k]/(len(degree)-1))} for k,c in enumerate(xf))
    save_csv(output,'network_sensitivity.csv',graph_rows);save_csv(output,'network_nodes.csv',node_rows)
    excluded=set(config['exclude_from_compound_top3'])
    unknown=excluded-set(xf)
    if unknown:raise ValueError('Unknown excluded field: '+str(unknown))
    eligible=np.array([i for i,c in enumerate(xf) if c not in excluded])
    save_csv(output,'factor_eligibility.csv',[{'field':c,'eligible_compound':c not in excluded,
              'reason':'individual_analyte_candidate' if c not in excluded else 'bulk_property_or_aggregate',
              'note':'Ethanol and dissolved CO2 retained as individual chemical analytes'} for c in xf])
    x,y=xf.to_numpy(float),yf.to_numpy(float)
    # Fit fixed-alpha descriptive model, with no target-based feature selection.
    settings={'ridge_alphas':[config['ridge_alpha']]}
    model,_=make_pipeline('ridge_asinh',settings,config['seed'])
    model.set_params(model__alpha=config['ridge_alpha']).fit(x,y)
    coefs=model.named_steps['model'].coef_
    rho=marginal_spearman(x,y)
    all_factors=[]
    for t,target in enumerate(yf):
        for j,field in enumerate(xf):
            all_factors.append({'target':target,'field':field,'ridge_signed_coef':coefs[t,j],
                               'spearman_rho':rho[t,j],'eligible_compound':field not in excluded})
    save_csv(output,'all_factor_scores.csv',all_factors)
    rng=np.random.default_rng(config['seed'])
    draws=[];selections=[];counts=np.zeros_like(coefs,dtype=int)
    for repeat in range(config['bootstrap_repeats']):
        draw=stratified_draw(ids.tasting_category_fine,rng);draws.append(draw)
        boot,_=make_pipeline('ridge_asinh',settings,config['seed'])
        boot.set_params(model__alpha=config['ridge_alpha']).fit(x[draw],y[draw])
        coefficients=boot.named_steps['model'].coef_
        for t,target in enumerate(yf):
            chosen=top_indices(coefficients[t],eligible)
            counts[t,chosen]+=1
            selections.extend({'repeat':repeat,'target':target,'rank':rank+1,'field':xf.columns[j],
                               'signed_coef':coefficients[t,j]} for rank,j in enumerate(chosen))
        if (repeat+1)%25==0:
            save_csv(output,'bootstrap_selections.csv',selections)
            np.save(output/'bootstrap_draws.npy',np.array(draws),allow_pickle=False)
            write_json(output/'status.json',{'state':'running','bootstrap_completed':repeat+1,'requested':config['bootstrap_repeats']})
            print(f'bootstrap {repeat+1}/{config["bootstrap_repeats"]}',flush=True)
    save_csv(output,'bootstrap_selections.csv',selections)
    np.save(output/'bootstrap_draws.npy',np.array(draws),allow_pickle=False)
    ranks=[];agreement=[];frequencies=[]
    for t,target in enumerate(yf):
        main=top_indices(coefs[t],eligible);marginal=top_indices(rho[t],eligible)
        a,b=set(main),set(marginal)
        agreement.append({'target':target,'top3_jaccard':len(a&b)/len(a|b),
                          'ridge_top3':';'.join(xf.columns[main]),'spearman_top3':';'.join(xf.columns[marginal])})
        for j in eligible:
            frequencies.append({'target':target,'field':xf.columns[j],'count':counts[t,j],
                                'bootstrap_repeats':config['bootstrap_repeats'],'frequency':counts[t,j]/config['bootstrap_repeats']})
        for rank,j in enumerate(main):
            field=xf.columns[j];meta=metadata.loc[field]
            ranks.append({'target':target,'primary':target in primary,'rank':rank+1,'field':field,
                          'chemical_name':aliases[field],'ridge_signed_coef':coefs[t,j],
                          'bootstrap_frequency':counts[t,j]/config['bootstrap_repeats'],
                          'marginal_spearman':rho[t,j],'in_marginal_top3':j in b,
                          'reference_unit':meta['unit'],'reference_origin_category':meta['origin_paper'],
                          'reference_off_flavors':meta['off_flavors'],
                          'chemical_rationale_status':'pending_primary_source_review',
                          'interpretation':'association_candidate_not_causal_driver'})
    save_csv(output,'top3_candidates.csv',ranks)
    save_csv(output,'bootstrap_frequency.csv',frequencies)
    save_csv(output,'method_agreement.csv',agreement)
    project=Path(__file__).resolve().parents[2]
    manifest={'stage':'Q2 exploratory network and association','finished_utc':datetime.now(timezone.utc).isoformat(),
              'source':str(source),'bootstrap_repeats':config['bootstrap_repeats'],'alpha':config['ridge_alpha'],
              'source_sha256':{n:sha256(source/n) for n in ['1.csv','2.csv','4.csv','8.xlsx']},
              'config_sha256':sha256(config_path),'python_versions_reference':'requirements-lock.txt',
              'code_sha256':{str(p.relative_to(project)):sha256(p) for p in sorted((project/'src').rglob('*.py'))},
              'limitations':['Full-data descriptive associations; no causal interpretation.',
                            'Ridge alpha fixed; bootstrap frequency is conditional on model and within-style resampling.',
                            'Coefficients measure one SD of transformed input, not original concentration effects.',
                            'Marginal Spearman and adjusted ridge measure different associations.',
                            'Network centrality is not per-target importance; chemical rationale remains unreviewed.']}
    write_json(output/'run_manifest.json',manifest)
    write_json(output/'status.json',{'state':'completed','bootstrap_completed':config['bootstrap_repeats'],
                                    'top3_rows':len(ranks),'network_nodes':len(xf.columns)})
    print('Q2 exploratory outputs complete',flush=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ['source','audit','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/q2_exploratory_v1.json'))
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.source,args.audit,args.config,args.output)

if __name__=='__main__':main()
