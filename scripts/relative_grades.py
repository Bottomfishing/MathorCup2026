"""Provisional relative grades, NOT recovered panel ratings. AI-assisted Codex, 2026-10-24."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def grade(reference,prediction):
    reference=np.sort(np.asarray(reference,float));prediction=np.asarray(prediction,float)
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError('Grades require nonempty finite reference and finite predictions')
    # Midrank ECDF handles ties: a constant reference mapped at its value gives grade 3.
    left=np.searchsorted(reference,prediction,side='left')
    right=np.searchsorted(reference,prediction,side='right')
    percentile=(left+right)/(2*len(reference))
    return np.clip(np.floor(7*percentile),0,6).astype(int),percentile


def run(baseline,out):
    manifest=json.loads((baseline/'run_manifest.json').read_text(encoding='utf-8'))
    source=Path(manifest['source'])
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest
    splits=pd.read_csv(baseline/'split_manifest.csv',dtype={'beer':str})
    truth=pd.read_csv(source/'4.csv',dtype={'beer':str}).set_index('beer')
    targets=pd.read_csv(baseline/'target_columns.csv')
    oof=pd.read_csv(baseline/'oof_predictions.csv',dtype={'beer':str})
    oof=oof.loc[oof.model.eq('extra_trees')].copy()
    assert not oof.duplicated(['beer','target']).any()
    rows=[];calibration=[]
    for (fold,target),group in oof.groupby(['fold','target']):
        train=splits.loc[splits.outer_test_fold.ne(fold),'beer'].tolist()
        test=splits.loc[splits.outer_test_fold.eq(fold),'beer'].tolist()
        assert not set(train)&set(test) and set(group.beer)==set(test)
        reference=truth.loc[train,target].to_numpy(float)
        values,percentiles=grade(reference,group.predicted.to_numpy())
        # Independent direct count, not the searchsorted implementation.
        direct=np.array([(np.count_nonzero(reference<p)+.5*np.count_nonzero(reference==p))/len(reference) for p in group.predicted])
        np.testing.assert_allclose(percentiles,direct,rtol=0,atol=1e-12)
        np.testing.assert_array_equal(values,np.minimum(6,np.floor(7*direct).astype(int)))
        calibration.append({'fold':int(fold),'target':target,'training_beers':train,'sorted_reference':sorted(reference.tolist())})
        for record,value,percentile in zip(group.to_dict('records'),values,percentiles):
            rows.append({k:record[k] for k in ['beer','style','fold','model','target','predicted']}|
                        {'relative_grade_0_6':int(value),'training_reference_percentile':float(percentile),
                         'grade_semantics':'provisional_relative_rank_not_original_panel_rating',
                         'original_score_recovered':False})
    result=pd.DataFrame(rows);assert len(result)==250*50
    primary=targets.loc[targets.primary,'field'].tolist()
    assert len(primary)==45
    out.mkdir(parents=True,exist_ok=False)
    result.to_csv(out/'relative_grades_all50_long.csv',index=False,encoding='utf-8-sig')
    result.pivot(index='beer',columns='target',values='relative_grade_0_6').loc[splits.beer,primary].to_csv(out/'relative_grades_primary45.csv',encoding='utf-8-sig')
    result.groupby(['target','relative_grade_0_6']).size().rename('count').reset_index().to_csv(out/'grade_counts.csv',index=False,encoding='utf-8-sig')
    (out/'fold_calibration.json').write_text(json.dumps(calibration,ensure_ascii=False),encoding='utf-8')
    for reference,prediction,expected in [([2,2,2],[2],[3]),([0,1,2],[-1,3],[0,6]),([0,1,2],[0,1,2],[1,3,5])]:
        np.testing.assert_array_equal(grade(reference,prediction)[0],expected)
    metadata={'status':'passed','rows':len(result),'primary_shape':[250,45],'source_sha256':manifest['source_sha256'],
              'baseline':str(baseline.resolve()),'input_oof_sha256':hashlib.sha256((baseline/'oof_predictions.csv').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'method':'floor(7 * midrank empirical CDF of outer-training labels), clipped to 0..6',
              'independent_count_recomputation':True,'outer_test_excluded_from_reference':True,
              'original_score_recovered':False,'official_acceptance_confirmed':False,
              'limitations':['User authorized explicitly labelled assumptions; no organizer clarification available.',
                             'Grade 0 means a low relative rank, not absence of flavor; grade 6 is not maximum physical intensity.',
                             'Different targets and folds use different empirical reference distributions.',
                             'Model selected after exploratory baseline comparison; no raw-grade accuracy can be measured.',
                             'This is an OOF provisional table, not a fitted deployment scoring service.']}
    (out/'verification_and_assumptions.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(result),'primary_shape':[250,45],'verification':'passed','original_score_recovered':False}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();run(a.baseline,a.output)
