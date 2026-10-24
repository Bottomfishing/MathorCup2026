"""Q3 diagnostic plots, not final physical claims. AI-assisted: Codex, 2026-10-24."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig,path):
    for ext in ['png','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def run(folder,out):
    out.mkdir(parents=True,exist_ok=False)
    summary=pd.read_csv(folder/'validation_summary.csv')
    labels=['Resample','Interpolate','Latent Gaussian'];colors=['#8c9eaa','#2b7a78','#d38a53']
    fig,axes=plt.subplots(1,4,figsize=(14,4))
    for ax,col,title in zip(axes,['mean_heldout_wasserstein','mean_variance_ratio','mean_copy_fraction','mean_clipped_fraction'],
                            ['Held-out marginal distance\n(lower is closer)','Variance / training variance','Exact training copy fraction','Feature clipping fraction']):
        ax.bar(range(3),summary[col],color=colors)
        ax.set_xticks(range(3),labels,rotation=30,ha='right');ax.set_title(title,fontsize=10)
        ax.set_ylim(0,max(1.08,summary[col].max()*1.15));ax.grid(axis='y',alpha=.2)
        for i,v in enumerate(summary[col]):ax.text(i,v+.025,f'{v:.3f}',ha='center',fontsize=9)
    fig.suptitle('Generation diagnostics: equal-weight mean over 22 styles x 3 folds')
    fig.tight_layout();save(fig,out/'generation_tradeoffs')
    meta=pd.read_csv(folder/'generated_metadata.csv');sensory=pd.read_csv(folder/'generated_sensory.csv')
    joined=sensory.merge(meta[['generated_id','style','method']],on='generated_id',validate='many_to_one')
    means=joined.groupby(['target','style','method']).prediction.mean().unstack(['style','method'])
    target_order=pd.read_csv(folder/'target_order.csv').target.tolist()
    columns=[(s,m) for s in ['Tripel','Kriek','Lambic'] for m in ['resample','interpolate','latent_gaussian']]
    means=means.loc[target_order,columns]
    means.to_csv(out/'generated_sensory_means.csv',encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(11,15));limit=max(abs(means.to_numpy()).max(),.1)
    im=ax.imshow(means.to_numpy(),cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_yticks(range(len(target_order)),target_order,fontsize=8)
    ax.set_xticks(range(9),[f'{s}\n{m}' for s,m in columns],rotation=35,ha='right',fontsize=8)
    for x in [2.5,5.5]:ax.axvline(x,color='white',lw=2)
    ax.set_title('Generated sensory profiles: mean of 100 per style/method\nSaved forest ensemble estimates in standardized target scale',fontsize=11)
    fig.colorbar(im,ax=ax,shrink=.6,label='Model prediction (not original 0–6 rating)')
    save(fig,out/'generated_sensory_profiles')
    gates=pd.read_csv(folder/'transfer_gate_counts.csv',dtype={'beer':str});results=pd.read_csv(folder/'transfer_results.csv',dtype={'beer':str})
    fields=['DMS','X2.3.butanedione','X4VG','X.E..2.nonenal','dms','diacetyl','X4vg','t2n','stale_hops','all_chemical','all_sensory']
    matrix=gates.assign(fraction=gates.n_passed/gates.n_candidates).pivot(index='beer',columns='field',values='fraction').loc[results.beer,fields]
    fig,ax=plt.subplots(figsize=(13,6));im=ax.imshow(matrix.to_numpy(),vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(len(fields)),['DMS (input)','Diacetyl (input)','4VG (input)','Nonenal (input)','DMS (proxy)','Diacetyl (proxy)','4VG (proxy)','Nonenal (proxy)','Stale hops (proxy)','All 4 input gates','All 5 proxy gates'],rotation=35,ha='right',fontsize=8)
    ax.set_yticks(range(len(results)),[f'Beer {r.beer}'+(' *' if r.source_missing_screen_fields else '') for r in results.itertuples()])
    for i in range(len(matrix)):
        for j in range(len(fields)):ax.text(j,i,f'{matrix.iloc[i,j]:.1%}',ha='center',va='center',fontsize=7,color='white' if matrix.iloc[i,j]>.65 else 'black')
    ax.set_title('Pils/Lager → Tripel: fraction of 2,000 candidates passing each gate\nNo source passed all gates; * protected source value missing, no certification')
    fig.colorbar(im,ax=ax,label='Candidate fraction');save(fig,out/'transfer_gate_diagnostics')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.input,a.output)
