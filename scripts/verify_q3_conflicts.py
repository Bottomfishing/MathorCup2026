"""Verify finite-pool certificates using set intersection and exhaustive proper subsets."""
import argparse
import itertools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from beer_audit.audit import sha256,write_json


def verify(output):
    protocol_path=ROOT/'configs/q3_conflict_audit_20261026.json'
    protocol=json.loads(protocol_path.read_text(encoding='utf-8'));summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    path=ROOT/protocol['input'];assert summary['input_sha256']==sha256(path) and summary['protocol_sha256']==sha256(protocol_path)
    read=lambda path:pd.read_csv(path,float_precision='round_trip',dtype={'beer':str},keep_default_na=False)
    grid=read(path);subsets=read(output/'subset_feasibility.csv');cores=read(output/'minimal_blockers.csv')
    witnesses=read(output/'proper_subset_witnesses.csv');nearest=read(output/'nearest_failing_candidates.csv')
    violations=read(output/'nearest_candidate_violations.csv');events=read(output/'individual_guard_counts.csv')
    ladder=read(output/'diagnostic_constraint_ladder.csv');defects=protocol['defects'];tol=protocol['tolerance']
    assert not subsets.duplicated(['beer','groups']).any() and len(subsets)==32*grid.beer.nunique()
    assert not nearest.duplicated('beer').any()
    expected_cores=[];expected_violations=[]
    for beer,raw in grid.groupby('beer',sort=False):
        g=raw.loc[raw.range_pass&raw.chemical_pass&raw.style_pass].set_index('candidate')
        universe=set(g.index);prob=[f'{t}_{kind}_delta_{cut}' for t in defects for kind in ['calibrated','uncalibrated'] for cut in ['ge1','ge5','6']]
        grade=[t+'_grade_delta' for t in defects]
        valid={t:{k for k,row in g.iterrows() if all(row[c]<=tol for c in [t+'_grade_delta',*[c for c in prob if c.startswith(t+'_')]])} for t in defects}
        def pool(names):
            result=universe.copy()
            for name in names:result &= valid[name]
            return result
        for size in range(6):
            for names in itertools.combinations(defects,size):
                label=';'.join(names);found=pool(names)
                row=subsets.loc[subsets.beer.eq(beer)&subsets.groups.eq(label)].iloc[0]
                assert int(row.n_candidates)==len(found)
                if found:assert int(float(row.witness_candidate)) in found
                else:assert row.witness_candidate==''
                proper=[part for n in range(size) for part in itertools.combinations(names,n)]
                if not found and all(pool(part) for part in proper):expected_cores.append((beer,label,size))
        for row in witnesses.loc[witnesses.beer.eq(beer)].itertuples():
            retained=tuple(row.retained_groups.split(';')) if row.retained_groups else ()
            assert set(retained)==set(row.blocking_groups.split(';'))-{row.removed_group}
            assert int(row.candidate) in pool(retained)
        for row in events.loc[events.beer.eq(beer)].itertuples():
            values=g[row.guard].to_numpy(float)
            assert row.n_candidates==len(g) and row.n_pass==sum(values<=tol)
            np.testing.assert_allclose([row.minimum_delta,row.median_delta,row.maximum_delta],[min(values),np.median(values),max(values)],rtol=0,atol=1e-12)
        raw_pass=(g[grade]<=0).all(axis=1)
        choices=g.loc[raw_pass]
        if len(choices):
            # Plain tuple ordering, independent of the implementation's vectorized lexsort.
            score=lambda item:(max(0.,*item[1][prob].tolist()),sum(item[1][prob]>tol),.7*item[1].target_distance+.3*item[1].change_distance,item[0])
            cid,point=min(choices.iterrows(),key=score)
            row=nearest.loc[nearest.beer.eq(beer)].iloc[0]
            assert int(row.candidate)==cid and not row.promoted_to_main and not row.physical_feasibility_certified
            np.testing.assert_allclose(row.maximum_positive_probability_delta,score((cid,point))[0],atol=1e-12)
            assert row.violated_probability_guards==score((cid,point))[1]
            assert row.complete_current_constraints_pass==(cid in pool(defects))
            expected_violations.extend((beer,int(cid),c,float(point[c])) for c in prob if point[c]>tol)
        checks={'structural':universe,'structural_and_raw_grades':set(g.index[raw_pass]),
            'raw_grades_and_nonzero_events_only':set(g.index[raw_pass&(g[[c for c in prob if c.endswith('_ge1')]]<=tol).all(axis=1)]),
            'raw_grades_and_calibrated_all_events':set(g.index[raw_pass&(g[[c for c in prob if '_calibrated_' in c]]<=tol).all(axis=1)]),
            'raw_grades_and_uncalibrated_all_events':set(g.index[raw_pass&(g[[c for c in prob if '_uncalibrated_' in c]]<=tol).all(axis=1)]),
            'complete_current_scenario':pool(defects)}
        for row in ladder.loc[ladder.beer.eq(beer)].itertuples():assert row.n_candidates==len(checks[row.diagnostic_scope]) and not row.acceptance_policy_changed
    assert list(cores[['beer','blocking_groups','group_count']].itertuples(index=False,name=None))==expected_cores
    assert len(witnesses)==int(cores.group_count.sum())
    assert len(violations)==len(expected_violations)
    for actual,expected in zip(violations.itertuples(index=False,name=None),expected_violations):
        assert actual[:3]==expected[:3];np.testing.assert_allclose(actual[3],expected[3],rtol=0,atol=1e-12)
    assert summary['complete_pass_count']==int(grid.all_current_constraints_pass.sum())==0
    write_json(output/'verification.json',dict(status='passed',frozen_candidate_rows=len(grid),all_subsets_checked=len(subsets),
        minimal_blockers_checked=len(cores),proper_subset_witnesses_checked=len(witnesses),individual_guards_checked=len(events),
        nearest_candidates_checked=len(nearest),positive_violations_checked=len(violations),
        scope='Independent set intersections and exhaustive subset certificates plus independent minimax ordering; consumes frozen model predictions, does not refit them or establish physical infeasibility.'))
    print('Q3 finite-pool conflict certificates independently verified.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);verify(p.parse_args().output)
