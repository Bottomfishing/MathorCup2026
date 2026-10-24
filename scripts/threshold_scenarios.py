"""Explicit, uncertified reference-threshold sensitivity. AI-assisted: Codex, 2026-10-24."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_models.generation import excess_severity
from beer_audit.audit import save_csv,write_json,sha256


def run(parent,reference,out):
    manifest=json.loads((parent/'run_manifest.json').read_text(encoding='utf-8'))
    cfg=json.loads((parent/'config.json').read_text(encoding='utf-8'));source=Path(manifest['source'])
    for name,digest in manifest['source_sha256'].items():
        if sha256(source/name)!=digest:raise ValueError('Source hash differs')
    if sha256(parent/'generator.joblib')!=manifest['generator_sha256']:raise ValueError('Generator hash differs')
    g=joblib.load(parent/'generator.joblib')
    features=pd.read_csv(parent/'feature_order.csv').field.tolist();targets=pd.read_csv(parent/'target_order.csv').target.tolist()
    raw=pd.read_csv(source/'1.csv',dtype={'beer':str}).set_index('beer')
    thresholds=pd.read_excel(reference,sheet_name='Sensory thresholds');thresholds['excel_row']=np.arange(len(thresholds))+2
    thresholds=thresholds.set_index('field_name_R')
    transformations={'DMS':'log10','X2.3.butanedione':'identity','X4VG':'log10'}
    units={'DMS':'ug/L','X2.3.butanedione':'mg/L','X4VG':'mg/L'}
    for field in transformations:
        if thresholds.loc[field,'unit']!=units[field]:raise ValueError('Reference unit changed')
    pool=pd.read_csv(parent/'transfer_candidate_pool.csv').set_index('candidate_id');values=pool[features].to_numpy(float)
    pred=pd.read_csv(parent/'transfer_candidate_sensory.csv').set_index('candidate_id').loc[pool.index,targets].to_numpy(float)
    models=[]
    for record in manifest['sensory_proxy_models']:
        path=Path(record['path'])
        if sha256(path)!=record['sha256']:raise ValueError('Proxy hash mismatch')
        models.append(joblib.load(path))
    columns=[features.index(c) for c in manifest['proxy_features']]
    def predict(x):return np.mean([m.predict(x[:,columns]) for m in models],axis=0)
    def physical(x,field):
        j=features.index(field)
        return np.power(10.,x[...,j]) if transformations[field]=='log10' else x[...,j]
    out.mkdir(parents=True,exist_ok=False);assumptions=[];results=[];checks=[];gate_rows=[]
    z=g.robust_.transform(values)
    center=g.robust_.transform(g.x_[g.indices_[cfg['transfer_target_style']]]).mean(axis=0)
    targetdist=np.sqrt(np.mean((z-center)**2,axis=1))
    for statistic in ['median','mean']:
        for field,transform in transformations.items():
            assumptions.append({'scenario':statistic,'field':field,'assumed_source_transform':transform,'assumed_unit':units[field],
                                'threshold':thresholds.loc[field,'threshold_'+statistic],
                                'source_url':'https://zenodo.org/records/10653704','sheet':'Sensory thresholds',
                                'excel_row':int(thresholds.loc[field,'excel_row']),
                                'status':'reference_scenario_not_validated_for_matrix_or_style'})
        for beer,row in raw.loc[raw.tasting_category_fine.eq(cfg['transfer_source_style'])].iterrows():
            original=row[features].to_numpy(float)
            missing=np.isnan(original[[features.index(c) for c in cfg['chemical_screen_fields']]]).any()
            x=g.impute_.transform(original[None,:])[0];yp=predict(x[None,:])[0]
            passing=np.ones(len(values),bool);detail=[]
            for field in transformations:
                threshold=float(thresholds.loc[field,'threshold_'+statistic])
                before=float(physical(x,field));after=physical(values,field)
                s_before=float(excess_severity(before,threshold));s_after=excess_severity(after,threshold)
                ok=s_after<=s_before+1e-10;passing&=ok
                detail.append((field,'assumed_concentration_exceedance',before,after,s_before,s_after,threshold,ok))
            # Missing threshold stays missing: a separate, stronger coordinate nonincrease gate.
            field='X.E..2.nonenal';j=features.index(field);before=x[j];after=values[:,j];ok=after<=before+1e-10;passing&=ok
            detail.append((field,'nonincrease_coordinate_no_threshold',before,after,np.nan,np.full(len(values),np.nan),np.nan,ok))
            field='stale_hops';j=targets.index(field);before=yp[j];after=pred[:,j];ok=after<=before+1e-10;passing&=ok
            detail.append((field,'model_sensory_proxy_no_threshold',before,after,np.nan,np.full(len(values),np.nan),np.nan,ok))
            for field,kind,b,a,sb,sa,th,ok in detail:
                gate_rows.append({'scenario':statistic,'beer':beer,'field':field,'kind':kind,'n_passed':int(ok.sum()),'n_candidates':len(values)})
            if missing:passing[:]=False
            change=np.sqrt(np.mean((z-g.robust_.transform(x[None,:])[0])**2,axis=1))
            weight=cfg['transfer_target_distance_weight'];objective=weight*targetdist+(1-weight)*change
            chosen=int(np.flatnonzero(passing)[np.argmin(objective[passing])]) if passing.any() else int(np.argmin(objective))
            status='source_protected_value_missing' if missing else ('scenario_passed_not_certified' if passing.any() else 'no_sampled_candidate_passed')
            source_distance=float(np.sqrt(np.mean((g.robust_.transform(x[None,:])[0]-center)**2)))
            results.append({'scenario':statistic,'beer':beer,'status':status,'n_candidates_passed':int(passing.sum()),
                            'selected_candidate_id':pool.index[chosen] if passing.any() else '',
                            'diagnostic_candidate_id':pool.index[chosen],'target_distance':targetdist[chosen],
                            'change_distance':change[chosen],'source_target_distance':source_distance,
                            'target_distance_improved':bool(targetdist[chosen]<source_distance-1e-10),
                            'objective':objective[chosen],'physical_constraints_certified':False})
            for field,kind,b,a,sb,sa,th,ok in detail:
                checks.append({'scenario':statistic,'beer':beer,'candidate_id':pool.index[chosen],'field':field,'kind':kind,
                               'before':b,'after':a[chosen],'threshold':th,'severity_before':sb,'severity_after':sa[chosen],
                               'screen_pass':bool(ok[chosen]),'physical_constraints_certified':False})
    save_csv(out,'assumptions.csv',assumptions);save_csv(out,'scenario_results.csv',results)
    save_csv(out,'selected_candidate_checks.csv',checks);save_csv(out,'gate_counts.csv',gate_rows)
    write_json(out/'run_manifest.json',{'parent':str(parent.resolve()),'reference':str(reference.resolve()),'reference_sha256':sha256(reference),
         'script_sha256':sha256(Path(__file__)),'scope':'Explicit reference mean/median scenarios; no threshold invented for nonenal or stale hops.',
         'limitations':['Source transformations and threshold applicability are assumptions, not final calibrated physical constraints.',
                        'Nonenal remains a supplied-coordinate nonincrease gate; stale hops remains a model proxy.',
                        'All scenarios retain physical_constraints_certified=false; finite candidate search does not prove feasibility or infeasibility in real brewing.']})
    write_json(out/'status.json',{'state':'completed','physical_constraints_certified':False})
    print(pd.DataFrame(results).groupby(['scenario','status']).size().to_string())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.parent,a.reference,a.output)
