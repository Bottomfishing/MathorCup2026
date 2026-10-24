"""Plots and explicit fitted-Ridge corner witnesses. AI-assisted: Codex, 2026-10-24."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


def run(holdout,box,output):
    output.mkdir(parents=True,exist_ok=False)
    for folder in [holdout,box]:
        if json.loads((folder/'verification.json').read_text())['status']!='passed':raise ValueError('Unverified input')
    metrics=pd.read_csv(holdout/'metrics.csv');tripel=metrics.loc[metrics.scope.eq('Tripel')].set_index('method')
    scenario=pd.read_csv(box/'scenario_results.csv',dtype={'beer':str})
    values=scenario.pivot(index='beer',columns='radius',values='worst_upper_margin')
    order=['124','81','218','96','230','185','139','209','119','90','28'];values=values.loc[order]
    methods=['training_style_median','direct_focused','reference_observed','four_coordinate_max3']
    fig,(left,right)=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[1,1.5]})
    maes=tripel.loc[methods,'mae'];left.barh(range(len(methods)),maes,color=['#477d76','#8ba99d','#c98d69','#bf654d'])
    left.set_yticks(range(4),['Style median','Direct focused model','Measured reference only','Reference + four-coordinate max'])
    left.invert_yaxis();left.set_xlabel('Held-out MAE (standardized Stale score)');left.set_xlim(0,.95)
    left.set_title('28 held-out Tripels\nObservational diagnostic, not an intervention test',fontsize=10)
    for i,value in enumerate(maes):left.text(value+.012,i,f'{value:.3f}',va='center',fontsize=9)
    im=right.imshow(values,cmap='RdBu',norm=TwoSlopeNorm(vmin=-.03,vcenter=0,vmax=.2),aspect='auto')
    right.set_xticks(range(len(values.columns)),[f'{r*100:g}%' for r in values.columns])
    right.set_yticks(range(11),['Beer '+b for b in order]);right.set_xlabel('Scenario half-width / full-training feature IQR')
    right.set_title('Worst fitted-model upper margin to source\nPositive: all three anchored model bounds pass',fontsize=10)
    for i in range(len(values)):
        for j in range(len(values.columns)):
            v=values.iloc[i,j];right.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=7,color='white' if v>.12 or v<-.02 else 'black')
    fig.colorbar(im,ax=right,fraction=.045,pad=.02,extend='both',label='Standardized-score margin; color saturates above 0.2')
    fig.text(.02,.01,'Reference anchoring remains an unvalidated assumption. Model-range bounds are not confidence intervals or physical guarantees.',fontsize=9)
    fig.tight_layout(rect=[0,.035,1,1])
    for ext in ['png','svg']:fig.savefig(output/('reference_diagnostics.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    # An exact transformed-Ridge maximizer gives an actual fitted-model witness,
    # unlike a failed conservative forest bound, which alone proves no violation.
    summary=json.loads((box/'summary.json').read_text(encoding='utf-8'))
    q1=Path(summary['q1']);bundle=joblib.load(q1/'model.joblib');features=bundle['features'];stale=bundle['targets'].index('stale_hops')
    ridge=bundle['guard'];coef=ridge.named_steps['model'].coef_[stale]
    boxes=pd.read_csv(box/'coordinate_boxes.csv',dtype={'beer':str,'reference_beer':str}).set_index(['beer','radius'])
    bounds=pd.read_csv(box/'model_bounds.csv',dtype={'beer':str,'reference_beer':str})
    classifiers={n:joblib.load(Path(summary['style_models'])/(n+'.joblib')) for n in ['logistic','extra_trees']}
    witnesses=[];profiles=[]
    for row in bounds.loc[bounds.model.eq('ridge')&bounds.upper_margin_to_source.lt(-1e-10)].itertuples():
        b=boxes.loc[(row.beer,row.radius)];lo=np.array([b['lower__'+f] for f in features]);hi=np.array([b['upper__'+f] for f in features])
        corner=np.where(coef>=0,hi,lo);raw=float(ridge.predict(corner[None,:])[0,stale])
        np.testing.assert_allclose(raw,row.raw_model_upper,atol=1e-10)
        after=row.reference_observed_stale+raw-row.reference_model
        assert after>row.source_observed_stale+1e-10
        witness_id=f'{row.beer}_radius_{row.radius:g}'
        witnesses.append({'witness_id':witness_id,'beer':row.beer,'radius':row.radius,'model':'ridge',
                          'anchored_prediction':after,'source_observed_stale':row.source_observed_stale,
                          'model_guard_violation':after-row.source_observed_stale,
                          'logistic_label':classifiers['logistic'].predict(corner[None,:])[0],
                          'extra_trees_label':classifiers['extra_trees'].predict(corner[None,:])[0],
                          'scope':'coordinate-box fitted-model counterexample; not a measured physical outcome'})
        profiles.append({'witness_id':witness_id,**dict(zip(features,corner))})
    pd.DataFrame(witnesses).to_csv(output/'ridge_model_guard_witnesses.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(profiles).to_csv(output/'ridge_witness_profiles.csv',index=False,encoding='utf-8-sig')
    print(f'Plots saved; {len(witnesses)} exact fitted-Ridge witnesses verified.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['holdout','box','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.holdout,a.box,a.output)
