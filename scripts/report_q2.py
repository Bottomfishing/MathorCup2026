"""Reproducible Q2 diagnostic tables/figures. AI-assisted: Codex, 2026-10-24."""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from beer_audit.audit import save_csv,write_json,sha256
from beer_audit.checks import repair_utf8_mojibake


def normalize(name):
    return re.sub(r'[^\w]+','',unicodedata.normalize('NFKC',repair_utf8_mojibake(str(name))).casefold())


def jaccard(a,b):
    a,b=set(a),set(b)
    return len(a&b)/len(a|b)


def figure(fig,path):
    for ext in ['png','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def run(exploratory,permutation,within,audit,reference,out):
    out.mkdir(parents=True,exist_ok=False)
    old=pd.read_csv(exploratory/'top3_candidates.csv')
    full=pd.read_csv(permutation/'full/top3_permutation.csv')
    reduced=pd.read_csv(permutation/'analytes_only/top3_permutation.csv')
    strat=pd.read_csv(within/'top3.csv')
    targets=pd.read_csv(permutation/'target_order.csv').target.tolist()
    feature_meta=pd.read_csv(audit/'feature_dictionary.csv').set_index('field_name_R')
    evidence=pd.read_excel(reference,sheet_name='Supplementary Table S1')
    evidence.columns=evidence.columns.str.strip()
    evidence['excel_row']=np.arange(len(evidence))+2
    evidence['key']=evidence.Compound.map(normalize)
    duplicate=set(evidence.loc[evidence.key.duplicated(keep=False),'key'])
    by_name=evidence.loc[~evidence.key.isin(duplicate)].set_index('key')
    ledger=[]
    for field in sorted(set(old.field)|set(full.field)|set(reduced.field)|set(strat.field)):
        meta=feature_meta.loc[field];name=repair_utf8_mojibake(meta.chemical_name);key=normalize(name)
        hit=by_name.loc[key] if key in by_name.index else None
        row={'field':field,'chemical_name':name,'reference_unit':meta.unit,
             'unit_scope':'reference_only_not_confirmed_input_unit','name_match':'normalized_exact' if hit is not None else 'unresolved',
             'source_url':'https://zenodo.org/records/10653704','source_file_sha256':sha256(reference),
             'source_sheet':'Supplementary Table S1','source_excel_row':int(hit.excel_row) if hit is not None else None,
             'compound_class':str(hit['Compound class']).strip() if hit is not None else '',
             'origin':str(hit.Origin).strip() if hit is not None else '',
             'flavor_description_present':bool(hit is not None and pd.notna(hit.Flavor)),
             'underlying_flavor_reference':str(hit['Flavor reference']).strip() if hit is not None and pd.notna(hit['Flavor reference']) else '',
             'evidence_scope':'author_supplied_compound_metadata; target_specific_effect_not_verified',
             'causal_or_mechanistic_validation':'not_established','human_review':'pending'}
        ledger.append(row)
    ledger=pd.DataFrame(ledger);save_csv(out,'compound_evidence_ledger.csv',ledger)
    comparisons=[];candidates=[]
    metrics=pd.read_csv(permutation/'predictive_metrics.csv')
    for target in targets:
        a=old.loc[old.target.eq(target)].sort_values('rank')
        b=full.loc[full.target.eq(target)].sort_values('rank')
        c=reduced.loc[reduced.target.eq(target)].sort_values('rank')
        d=strat.loc[strat.target.eq(target)].sort_values('rank')
        comparisons.append({'target':target,'ridge_vs_forest_jaccard':jaccard(a.field,b.field),
                            'full_vs_analytes_jaccard':jaccard(b.field,c.field),
                            'global_vs_within_style_jaccard':jaccard(c.field,d.field),
                            'full_model_r2':b.model_target_r2.iloc[0],
                            'analytes_model_r2':c.model_target_r2.iloc[0]})
        for method,table in [('ridge_refit',a),('forest_full',b),('forest_analytes',c),('forest_within_style',d)]:
            for record in table.to_dict('records'):
                record['method']=method
                record['interpretation']='association_or_model_reliance_not_causal'
                record['target_specific_chemical_rationale']='pending_independent_review'
                candidates.append(record)
    comparisons=pd.DataFrame(comparisons);save_csv(out,'method_comparison.csv',comparisons)
    combined=pd.DataFrame(candidates).drop(columns=['chemical_name','reference_unit'],errors='ignore').merge(ledger,on='field',validate='many_to_one')
    save_csv(out,'candidate_evidence_long.csv',combined)
    global_imp=pd.read_csv(permutation/'analytes_only/individual_importance.csv')
    conditional=pd.read_csv(within/'importance.csv')
    paired=global_imp.merge(conditional,on=['target','field'],suffixes=('_global','_within'),validate='one_to_one')
    save_csv(out,'global_within_importance.csv',paired)
    # Descriptive full-data graph, never used to define held-out feature groups.
    edges=pd.read_csv(exploratory/'network_edges_0.6.csv')
    nodes=pd.read_csv(exploratory/'network_nodes.csv');nodes=nodes.loc[nodes.threshold.eq(.6)]
    graph=nx.Graph();graph.add_nodes_from(nodes.field)
    for e in edges.itertuples():graph.add_edge(e.source,e.target,rho=e.spearman_rho,weight=abs(e.spearman_rho))
    components=sorted(nx.connected_components(graph),key=lambda g:(-len(g),sorted(g)[0]))
    largest=graph.subgraph(components[0]);other=[c for c in components[1:] if len(c)>1]
    isolates=sorted(nx.isolates(graph))
    pos=nx.spring_layout(largest,seed=20261024,iterations=200,weight='weight')
    fig=plt.figure(figsize=(15,10));grid=fig.add_gridspec(2,3)
    ax=fig.add_subplot(grid[:,:2]);ax_other=fig.add_subplot(grid[0,2]);ax_iso=fig.add_subplot(grid[1,2])
    def draw(g,positions,axis):
        nx.draw_networkx_edges(g,positions,ax=axis,edge_color=['#357aa8' if g.edges[e]['rho']>0 else '#cf664e' for e in g.edges],alpha=.55,width=1.1)
        nx.draw_networkx_nodes(g,positions,ax=axis,node_size=[20+5*graph.degree(n) for n in g],node_color='#28655e')
        axis.axis('off')
    draw(largest,pos,ax)
    other_pos={}
    for k,component in enumerate(other):
        local=nx.circular_layout(graph.subgraph(component),scale=.32)
        for n,p in local.items():other_pos[n]=p+np.array([k%3,-(k//3)])
    draw(graph.subgraph(other_pos),other_pos,ax_other)
    nx.draw_networkx_nodes(graph,{n:(k%11,-(k//11)) for k,n in enumerate(isolates)},nodelist=isolates,ax=ax_iso,node_size=22,node_color='#9cacae')
    ax_iso.axis('off')
    top=nodes.sort_values(['degree','field'],ascending=[False,True]).head(10)
    labels={r.field:str(k+1) for k,r in enumerate(top.itertuples()) if r.field in largest}
    for k,n in enumerate(sorted(labels,key=lambda n:pos[n][1])):
        ax.annotate(labels[n],xy=pos[n],xytext=(-.65,-.1+k*.11),fontsize=9,
                    bbox={'boxstyle':'circle,pad=.2','fc':'white','ec':'#28655e'},
                    arrowprops={'arrowstyle':'-','color':'#657d78','lw':.65})
    ax.set_title(f'Largest component: {len(largest)} nodes')
    ax_other.set_title(f'Other connected components: {len(other)}')
    ax_iso.set_title(f'Isolated nodes: {len(isolates)}')
    fig.suptitle(f'Descriptive chemical network | Spearman |rho| >= 0.60\n231 nodes, {len(edges)} edges; blue = positive, orange = negative',fontsize=15)
    for k,r in enumerate(top.itertuples()):
        fig.text(.1+(k//5)*.45,.065-(k%5)*.021,f'{k+1}. {repair_utf8_mojibake(feature_meta.loc[r.field,"chemical_name_short"])} (degree {r.degree})',fontsize=9)
    fig.text(.1,-.055,'All nodes shown. Components have independent layouts; positions/distances across panels have no quantitative meaning.\nFull-data descriptive associations; not causal effects or held-out feature selection.',fontsize=9)
    figure(fig,out/'network_0_6')
    fig,axes=plt.subplots(1,2,figsize=(12,15),gridspec_kw={'width_ratios':[3,1]},sharey=True)
    values=comparisons.iloc[:,1:4].to_numpy()
    im=axes[0].imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    axes[0].set_yticks(range(len(targets)),targets,fontsize=8)
    axes[0].set_xticks(range(3),['Ridge vs forest','Full vs analytes','Global vs within style'],rotation=25,ha='right')
    for i in range(len(targets)):
        for j in range(3):axes[0].text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if values[i,j]>.65 else 'black')
    r2=comparisons.analytes_model_r2.to_numpy()
    axes[1].barh(range(len(targets)),r2,color=np.where(r2>0,'#357c75','#c66b52'))
    axes[1].axvline(0,color='gray',lw=.8);axes[1].set_xlabel('OOF R²\nAnalytes model');axes[1].grid(axis='x',alpha=.2)
    axes[0].set_title('Top-3 set agreement (Jaccard)');axes[1].set_title('Prediction quality')
    fig.colorbar(im,ax=axes[0],fraction=.025,pad=.02);figure(fig,out/'method_agreement')
    chosen=reduced.merge(paired[['target','field','delta_mae_within']],on=['target','field'],validate='one_to_one')
    fig,ax=plt.subplots(figsize=(8,6))
    ax.scatter(chosen.delta_mae,chosen.delta_mae_within,alpha=.55,s=24,color='#2c7975')
    low=min(0,chosen.delta_mae_within.min());high=max(chosen.delta_mae.max(),chosen.delta_mae_within.max())
    ax.plot([low,high],[low,high],ls='--',color='gray',lw=1);ax.axhline(0,color='gray',lw=.8)
    ax.set(xlabel='Global permutation: increase in held-out MAE',ylabel='Within-style permutation: increase in held-out MAE',title='Style sensitivity of 150 global Top-3 candidates')
    ax.grid(alpha=.15);figure(fig,out/'style_sensitivity')
    summary={'source':'Q2 exploratory, not frozen final results','candidate_rows':len(combined),'unique_candidate_compounds':len(ledger),
             'exact_metadata_matches':int(ledger.name_match.eq('normalized_exact').sum()),
             'metadata_with_flavor_description':int(ledger.flavor_description_present.sum()),
             'mean_jaccard':{c:float(comparisons[c].mean()) for c in comparisons if c.endswith('jaccard')},
             'analytes_targets_nonpositive_r2':int((comparisons.analytes_model_r2<=0).sum()),
             'global_top3_with_nonpositive_within_style_importance':int((chosen.delta_mae_within<=0).sum()),
             'global_top3_median_importance':float(chosen.delta_mae.median()),
             'same_candidates_median_within_style_importance':float(chosen.delta_mae_within.median())}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['exploratory','permutation','within','audit','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.exploratory,a.permutation,a.within,a.audit,a.reference,a.output)
