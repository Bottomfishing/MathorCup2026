"""Training-only empirical missingness patterns and complete seven-grade confusion."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv,write_json,sha256
from beer_models.final_model import relative_grades
from beer_models.ordinal import seven_level_probabilities,grade_decisions
from complete_modeling import load_q1_artifacts


def run(source,q1,output):
    cp=Path('configs/review_missingness_20261025.json');protocol=json.loads(cp.read_text())
    config=json.loads(Path('configs/q1_ordinal_protocol.json').read_text())
    ids,xf,yf,bundle,splits=load_q1_artifacts(source,Path('outputs/p0_audit_20261023_final'),q1)
    x=xf.to_numpy(float);y=yf.to_numpy(float);primary=np.asarray(bundle['primary'],bool)
    output.mkdir(parents=True,exist_ok=False)
    observations=np.empty_like(y,int);clean=np.empty_like(y,int);masks=[];donors=[];rates=[];predictions=[]
    for fold,(train,test) in enumerate(splits):
        truth=relative_grades(y[train],y[test]);observations[test]=truth
        model=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('model',ExtraTreesClassifier(n_estimators=config['n_estimators'],min_samples_leaf=config['min_samples_leaf'],
                max_features=config['max_features'],random_state=config['seed']+fold,n_jobs=1))])
        model.fit(x[train],relative_grades(y[train],y[train]))
        def predict(a):return grade_decisions(seven_level_probabilities(model.named_steps['model'],model.named_steps['impute'].transform(a)))['conditional_median']
        clean[test]=predict(x[test]);missing=~np.isfinite(x[train]);rate=missing.mean()
        rates.extend(dict(fold=fold,field=f,training_missing_count=int(missing[:,j].sum()),training_n=len(train)) for j,f in enumerate(xf))
        for repeat in range(protocol['repeats']):
            rng=np.random.default_rng(protocol['seed']+fold*1000+repeat)
            donor=rng.integers(len(train),size=len(test))
            donors.extend(dict(fold=fold,repeat=repeat,beer=ids.beer.iloc[i],training_donor=ids.beer.iloc[train[donor[k]]]) for k,i in enumerate(test))
            scenarios={'empirical_train_rows':missing[donor], 'uniform_train_rate':rng.random(x[test].shape)<rate}
            for name,mask in scenarios.items():
                added=mask&np.isfinite(x[test]);masked=x[test].copy();masked[added]=np.nan
                for k,j in zip(*np.where(added)):
                    masks.append(dict(fold=fold,repeat=repeat,scenario=name,beer=ids.beer.iloc[test[k]],field=xf.columns[j]))
                pred=predict(masked)
                predictions.extend(dict(fold=fold,repeat=repeat,scenario=name,beer=ids.beer.iloc[i],target=t,
                    observed=int(truth[k,j]),clean=int(clean[i,j]),masked=int(pred[k,j]),primary=bool(primary[j]))
                    for k,i in enumerate(test) for j,t in enumerate(yf))
        print(f'Empirical missingness fold {fold} complete',flush=True)
    saved=pd.read_csv('outputs/q1_task2_20261025/scores_all52_long.csv',dtype={'beer':str})
    reference=saved.loc[saved.panel_target_available].pivot(index='beer',columns='target',values='estimated_grade_0_6').loc[ids.beer.astype(str),yf.columns]
    np.testing.assert_array_equal(clean,reference.to_numpy(int))
    frame=pd.DataFrame(predictions)
    frame.to_csv(output/'scenario_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    save_csv(output,'added_missing_cells.csv',masks);save_csv(output,'training_pattern_donors.csv',donors);save_csv(output,'training_missing_rates.csv',rates)
    metrics=[]
    for (name,repeat),g in frame.loc[frame.primary].groupby(['scenario','repeat']):
        loss=abs(g.masked-g.observed);base=abs(g.clean-g.observed)
        metrics.append(dict(scenario=name,repeat=repeat,mae=loss.mean(),clean_mae=base.mean(),mae_change=(loss-base).mean(),
            changed_grade_fraction=g.masked.ne(g.clean).mean(),exact_accuracy=loss.eq(0).mean()))
    save_csv(output,'metrics.csv',metrics)
    confusion=[]
    for truth in range(7):
        for pred in range(7):
            count=int(((observations[:,primary]==truth)&(clean[:,primary]==pred)).sum())
            confusion.append(dict(observed=truth,predicted=pred,count=count,row_fraction=count/int((observations[:,primary]==truth).sum())))
    save_csv(output,'clean_confusion_primary45.csv',confusion)
    # Structural checks; a separate verifier reconstructs masks and metrics.
    assert sum(r['count'] for r in confusion)==len(x)*sum(primary)
    assert len(frame)==protocol['repeats']*2*len(x)*len(yf.columns)
    assert not frame.duplicated(['scenario','repeat','beer','target']).any()
    for r in masks:
        i=ids.index[ids.beer.astype(str).eq(str(r['beer']))][0];j=xf.columns.get_loc(r['field'])
        assert np.isfinite(x[i,j])
    write_json(output/'summary.json',dict(status='completed',protocol=protocol,config_sha256=sha256(cp),
        source_sha256={n:sha256(source/n) for n in ['1.csv','4.csv','8.xlsx']},model_features=len(xf.columns),
        model_input_missing_cells=int(np.isnan(x).sum()),model_input_cells=int(x.size),
        original_231_missing_cells=73,original_231_cells=250*231,main_results_replaced=False))
    write_json(output/'verification.json',dict(status='passed',clean_main_grades_reproduced=int(clean.size),
        scenario_prediction_rows=len(frame),confusion_count=sum(r['count'] for r in confusion),
        scope='Clean OOF equality, key uniqueness, finite-only new masks, confusion counts; not a second independent replay of every prediction.'))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source','q1','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.q1,a.output)
