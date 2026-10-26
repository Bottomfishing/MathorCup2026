"""Locate exact finite-pool blockers without relaxing any accepted constraint."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from beer_audit.audit import save_csv,write_json,sha256
from beer_models.constraint_conflicts import subset_counts,minimal_blockers,minimax_diagnostic


def run(output):
    cp=ROOT/'configs/q3_conflict_audit_20261026.json';protocol=json.loads(cp.read_text(encoding='utf-8'))
    path=ROOT/protocol['input'];grid=pd.read_csv(path,float_precision='round_trip',dtype={'beer':str})
    if grid.duplicated(['beer','candidate']).any():raise ValueError('Ambiguous frozen candidate IDs')
    defects=protocol['defects'];tol=protocol['tolerance']
    grade=[t+'_grade_delta' for t in defects]
    probability=[f'{t}_{kind}_delta_{event}' for t in defects for kind in ['calibrated','uncalibrated'] for event in ['ge1','ge5','6']]
    if not np.isfinite(grid[grade+probability].to_numpy()).all():raise ValueError('Nonfinite guard delta')
    for field in ['range_pass','chemical_pass','style_pass']:
        if grid[field].dtype!=bool:raise ValueError('Expected Boolean structural flags')
    output.mkdir(parents=True,exist_ok=False)
    subsets=[];cores=[];witnesses=[];events=[];nearest=[];breaches=[];ladder=[]
    for beer,all_rows in grid.groupby('beer',sort=False):
        g=all_rows.loc[all_rows[['range_pass','chemical_pass','style_pass']].all(axis=1)].reset_index(drop=True)
        groups=np.column_stack([(g[[t+'_grade_delta',*[c for c in probability if c.startswith(t+'_')]]]<=tol).all(axis=1) for t in defects])
        full=groups.all(axis=1)
        np.testing.assert_array_equal(full,g.all_current_constraints_pass.to_numpy(bool))
        counts=subset_counts(groups)
        for subset,count in counts.items():
            label=';'.join(defects[k] for k in subset)
            candidates=g.loc[groups[:,subset].all(axis=1),'candidate']
            subsets.append(dict(beer=beer,groups=label,group_count=len(subset),n_candidates=int(count),
                witness_candidate=int(candidates.iloc[0]) if len(candidates) else None))
        for core in minimal_blockers(groups):
            label=';'.join(defects[k] for k in core)
            cores.append(dict(beer=beer,blocking_groups=label,group_count=len(core),structurally_eligible=len(g),pool_only=True))
            for removed in core:
                proper=tuple(k for k in core if k!=removed)
                found=g.loc[groups[:,proper].all(axis=1),'candidate']
                witnesses.append(dict(beer=beer,blocking_groups=label,removed_group=defects[removed],
                    retained_groups=';'.join(defects[k] for k in proper),candidate=int(found.iloc[0])))
        for field in grade+probability:
            values=g[field].to_numpy(float)
            events.append(dict(beer=beer,guard=field,n_candidates=len(g),n_pass=int((values<=tol).sum()),
                minimum_delta=float(values.min()) if len(values) else None,
                median_delta=float(np.median(values)) if len(values) else None,maximum_delta=float(values.max()) if len(values) else None))
        raw=(g[grade]<=0).all(axis=1)
        cal=(g[[c for c in probability if '_calibrated_' in c]]<=tol).all(axis=1)
        uncal=(g[[c for c in probability if '_uncalibrated_' in c]]<=tol).all(axis=1)
        nonzero=(g[[c for c in probability if c.endswith('_ge1')]]<=tol).all(axis=1)
        # Alternative rows are explanatory ablations only, never new accepted results.
        for name,keep in [('structural',np.ones(len(g),bool)),('structural_and_raw_grades',raw),
                          ('raw_grades_and_nonzero_events_only',raw&nonzero),
                          ('raw_grades_and_calibrated_all_events',raw&cal),
                          ('raw_grades_and_uncalibrated_all_events',raw&uncal),('complete_current_scenario',full)]:
            ladder.append(dict(beer=beer,diagnostic_scope=name,n_candidates=int(np.sum(keep)),acceptance_policy_changed=False))
        eligible=g.loc[raw].reset_index(drop=True)
        if len(eligible):
            objective=.7*eligible.target_distance.to_numpy()+.3*eligible.change_distance.to_numpy()
            k,maximum,violations=minimax_diagnostic(eligible[probability].to_numpy(),objective,eligible.candidate.to_numpy(),tol)
            point=eligible.iloc[k]
            nearest.append(dict(beer=beer,candidate=int(point.candidate),anchor=int(point.anchor),fraction=float(point.fraction),policy=point.policy,
                maximum_positive_probability_delta=float(maximum[k]),violated_probability_guards=int(violations[k]),
                target_distance=float(point.target_distance),change_distance=float(point.change_distance),
                objective=float(objective[k]),complete_current_constraints_pass=bool(full[g.candidate.eq(point.candidate).to_numpy()][0]),
                promoted_to_main=False,physical_feasibility_certified=False))
            for field in probability:
                if point[field]>tol:breaches.append(dict(beer=beer,candidate=int(point.candidate),guard=field,positive_delta=float(point[field])))
    tables={'subset_feasibility.csv':subsets,'minimal_blockers.csv':cores,'proper_subset_witnesses.csv':witnesses,
            'individual_guard_counts.csv':events,'diagnostic_constraint_ladder.csv':ladder,
            'nearest_failing_candidates.csv':nearest,'nearest_candidate_violations.csv':breaches}
    for name,rows in tables.items():save_csv(output,name,rows)
    write_json(output/'summary.json',dict(status='completed',protocol=protocol,input_sha256=sha256(path),protocol_sha256=sha256(cp),
        sources=int(grid.beer.nunique()),frozen_candidates=len(grid),minimal_blocker_count=len(cores),
        complete_pass_count=int(grid.all_current_constraints_pass.sum()),new_candidate_search=False,main_results_replaced=False,
        scientific_scope='Exact finite-pool diagnosis of current modeled scenarios; extra probability guards are team choices, not quoted contest requirements. No new concentration threshold or evidence of actual intervention effect.'))
    print(pd.DataFrame(cores).to_string(index=False));print(pd.DataFrame(nearest)[['beer','maximum_positive_probability_delta','violated_probability_guards']].to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
