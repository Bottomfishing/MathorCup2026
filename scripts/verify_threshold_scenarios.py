"""Independently recompute scenario constraints. AI-assisted: Codex, 2026-10-24."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256


def verify(out):
    m=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'));parent=Path(m['parent'])
    assert sha256(Path(m['reference']))==m['reference_sha256']
    reference=pd.read_excel(m['reference'],sheet_name='Sensory thresholds').set_index('field_name_R')
    pm=json.loads((parent/'run_manifest.json').read_text(encoding='utf-8'));cfg=json.loads((parent/'config.json').read_text(encoding='utf-8'))
    assert sha256(parent/'generator.joblib')==pm['generator_sha256'];g=joblib.load(parent/'generator.joblib')
    for name,digest in pm['source_sha256'].items():assert sha256(Path(pm['source'])/name)==digest
    features=pd.read_csv(parent/'feature_order.csv').field.tolist();raw=pd.read_csv(Path(pm['source'])/'1.csv',dtype={'beer':str}).set_index('beer')
    pool=pd.read_csv(parent/'transfer_candidate_pool.csv').set_index('candidate_id')
    pool_y=pd.read_csv(parent/'transfer_candidate_sensory.csv').set_index('candidate_id')
    # Parent verification independently replayed these source proxy values and all pool predictions.
    assert json.loads((parent/'verification.json').read_text())['status']=='passed'
    before_proxy=pd.read_csv(parent/'transfer_constraint_diagnostics.csv',dtype={'beer':str})
    assumptions=pd.read_csv(out/'assumptions.csv');rows=pd.read_csv(out/'scenario_results.csv',dtype={'beer':str})
    checks=pd.read_csv(out/'selected_candidate_checks.csv',dtype={'beer':str});gates=pd.read_csv(out/'gate_counts.csv',dtype={'beer':str})
    z=g.robust_.transform(pool[features].to_numpy(float));center=g.robust_.transform(g.x_[g.indices_['Tripel']]).mean(axis=0)
    target=np.sqrt(np.mean((z-center)**2,axis=1))
    for row in rows.itertuples():
        raw_source=raw.loc[row.beer,features].to_numpy(float);x=g.impute_.transform(raw_source[None,:])[0]
        missing=np.isnan(raw_source[[features.index(f) for f in cfg['chemical_screen_fields']]]).any()
        passing=np.ones(len(pool),bool);calculated={}
        for a in assumptions.loc[assumptions.scenario.eq(row.scenario)].itertuples():
            assert a.threshold==reference.loc[a.field,'threshold_'+row.scenario]
            b=x[features.index(a.field)];after=pool[a.field].to_numpy()
            if a.assumed_source_transform=='log10':b=10**b;after=10**after
            sb=max(b/a.threshold-1,0);sa=np.maximum(after/a.threshold-1,0);ok=sa<=sb+1e-10
            passing&=ok;calculated[a.field]=(b,after,sb,sa,ok)
        field='X.E..2.nonenal';b=x[features.index(field)];after=pool[field].to_numpy();ok=after<=b+1e-10
        passing&=ok;calculated[field]=(b,after,np.nan,np.full(len(pool),np.nan),ok)
        field='stale_hops';b=before_proxy.loc[before_proxy.beer.eq(row.beer)&before_proxy.field.eq(field),'before'].iloc[0]
        after=pool_y.loc[pool.index,field].to_numpy();ok=after<=b+1e-10;passing&=ok
        calculated[field]=(b,after,np.nan,np.full(len(pool),np.nan),ok)
        for field,(_,_,_,_,ok) in calculated.items():
            saved=gates.loc[gates.scenario.eq(row.scenario)&gates.beer.eq(row.beer)&gates.field.eq(field)].iloc[0]
            assert saved.n_passed==int(ok.sum())
        if missing:passing[:]=False;assert row.status=='source_protected_value_missing'
        assert row.n_candidates_passed==int(passing.sum());assert not row.physical_constraints_certified
        change=np.sqrt(np.mean((z-g.robust_.transform(x[None,:])[0])**2,axis=1));w=cfg['transfer_target_distance_weight'];objective=w*target+(1-w)*change
        j=int(np.flatnonzero(passing)[np.argmin(objective[passing])]) if passing.any() else int(np.argmin(objective))
        assert row.diagnostic_candidate_id==pool.index[j]
        if passing.any():assert row.selected_candidate_id==pool.index[j]
        else:assert pd.isna(row.selected_candidate_id)
        np.testing.assert_allclose([row.target_distance,row.change_distance,row.objective],[target[j],change[j],objective[j]],atol=1e-10)
        source_distance=float(np.sqrt(np.mean((g.robust_.transform(x[None,:])[0]-center)**2)))
        np.testing.assert_allclose(row.source_target_distance,source_distance,atol=1e-10)
        assert row.target_distance_improved==bool(target[j]<source_distance-1e-10)
        for check in checks.loc[checks.scenario.eq(row.scenario)&checks.beer.eq(row.beer)].itertuples():
            b,after,sb,sa,ok=calculated[check.field]
            np.testing.assert_allclose([check.before,check.after,check.severity_before,check.severity_after],[b,after[j],sb,sa[j]],atol=1e-10,equal_nan=True)
            assert check.screen_pass==ok[j] and not check.physical_constraints_certified
    result={'status':'passed','source_scenario_pairs':len(rows),'selected_diagnostic_checks':len(checks),
            'candidate_constraints_per_scenario_source':len(pool),'physical_certification':False}
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1):verify(p.parse_args().output)
