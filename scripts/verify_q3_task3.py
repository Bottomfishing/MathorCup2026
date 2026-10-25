"""Independently check task3 spectra, task2 classifier decisions and five-defect claims."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256, write_json
from complete_modeling import load_q1_artifacts
from complete_q3_task3 import parser


def verify(args):
    out=args.output
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    for name,digest in summary['input_sha256'].items():assert sha256(Path(name))==digest,name
    for name,digest in summary['source_sha256'].items():assert sha256(args.source/name)==digest
    def read(name, **kw):return pd.read_csv(out/name,**kw)
    ids,xf,yf,base,_=load_q1_artifacts(args.source,Path('outputs/p0_audit_20261023_final'),args.q1)
    x=xf.to_numpy(float);beers=ids.beer.astype(str).to_numpy();lookup={b:i for i,b in enumerate(beers)}
    styles=ids.tasting_category_fine.to_numpy();pils=beers[styles=='Pils/Lager'];features=list(xf)
    filled=np.where(np.isfinite(x),x,np.nanmedian(x,axis=0))
    center=np.median(filled,axis=0);q1,q3=np.quantile(filled,[.25,.75],axis=0);std=filled.std(0)
    scale=np.where(q3-q1>1e-12,q3-q1,np.where(std>1e-12,std,1.))
    def z(values):return np.arcsinh((values-center)/scale)
    tripel=filled[styles=='Tripel'];target_center=z(tripel).mean(0)
    card=joblib.load(args.scorecard/'model.joblib');targets=card['targets'];extra=['A_acetaldehyde','F_acetaldehyde']
    all_targets=[r['target_id'] for r in card['scorecard_scope']]
    primary=[r['target_id'] for r in card['scorecard_scope'] if r['acceptance_scope']=='main_content']
    generated=read('generated_profiles.csv').set_index('generated_id')
    pd.testing.assert_frame_equal(generated,pd.read_csv(args.q3/'generated_profiles.csv').set_index('generated_id'))
    metadata=read('generated_metadata.csv',dtype={'parent_1':str,'parent_2':str})
    training_config=json.loads((args.q1/'run_manifest.json').read_text(encoding='utf-8'))['config']
    seeds={s:training_config['seed']+80000+i for i,s in enumerate(training_config['generation_styles'])}
    for r in metadata.itertuples():
        a,b=lookup[r.parent_1],lookup[r.parent_2]
        assert a!=b and styles[a]==styles[b]==generated.loc[r.generated_id,'style']
        assert 0<r.weight<1 and r.seed==seeds[styles[a]] and r.method=='interpolate'
        np.testing.assert_allclose(generated.loc[r.generated_id,features].to_numpy(float),r.weight*filled[a]+(1-r.weight)*filled[b],atol=1e-9,rtol=1e-12)
        assert r.parent_1_missing_features==np.isnan(x[a]).sum() and r.parent_2_missing_features==np.isnan(x[b]).sum()
    diagnostics=read('generated_diagnostics.csv').set_index('generated_id')
    classes={n:joblib.load(args.style_models/(n+'.joblib')) for n in ['logistic','extra_trees']}
    for name,model in classes.items():
        np.testing.assert_array_equal(diagnostics.loc[generated.index,name+'_prediction'],model.predict(generated[features].to_numpy()))
    for style,group in generated.groupby('style'):
        real=filled[styles==style];values=group[features].to_numpy(float)
        assert ((values>=real.min(0)-1e-10)&(values<=real.max(0)+1e-10)).all()
        nearest=cdist(z(values),z(real)).min(1)/np.sqrt(len(features))
        np.testing.assert_allclose(diagnostics.loc[group.index,'nearest_real_style_distance'],nearest,atol=1e-12)
        assert (nearest>1e-10).all() and len(group)==100
    before=read('transfer_source_profiles.csv',dtype={'beer':str}).set_index('beer').loc[pils,features]
    after=read('transfer_profiles.csv',dtype={'beer':str}).set_index('beer').loc[pils,features]
    old=pd.read_csv(args.resolution/'transfer_profiles.csv',dtype={'beer':str}).set_index('beer').loc[pils,features]
    np.testing.assert_allclose(after,old,atol=1e-10,rtol=0)
    results=read('transfer_results.csv',dtype={'beer':str}).set_index('beer')
    original=pd.read_csv(args.resolution/'transfer_results.csv',dtype={'beer':str}).set_index('beer')
    assert set(results.index)==set(pils) and results.selected.all()
    assert not results.actual_after_stale_measured.any() and not results.physical_feasibility_certified.any()
    cal=pd.read_excel(args.source/'8.xlsx',sheet_name='Calibrated chemical dataset').set_index('beer');cal.index=cal.index.astype(str)
    anchors=pd.read_csv(args.resolution/'anchors.csv').set_index('anchor')
    protected=[features.index(f) for f in summary['config']['protected_features']];other=[j for j in range(len(features)) if j not in protected]
    defects=read('five_defect_checks.csv',dtype={'beer':str}).set_index(['beer','defect'])
    evidence=read('source_defect_evidence.csv',dtype={'beer':str}).set_index(['beer','field'])
    changes=read('transfer_coordinate_changes.csv',dtype={'beer':str}).set_index(['beer','field'])
    assert len(defects)==55 and len(changes)==11*224 and not defects.actual_after_validated.any()
    for beer in pils:
        i=lookup[beer];a=filled[i].copy();r=results.loc[beer]
        for j in protected:
            field=features[j];e=evidence.loc[(beer,field)]
            if np.isnan(x[i,j]):
                assert field=='X2.3.butanedione' and cal.loc[beer,field]==0 and e.constraint_cap_coordinate==0
                assert e.basis=='reference_zero_cap_for_nonnegative_diacetyl';a[j]=0
            else:assert e.basis=='observed_coordinate' and abs(e.constraint_cap_coordinate-x[i,j])<1e-12
        np.testing.assert_allclose(before.loc[beer],a,atol=1e-10)
        oldr=original.loc[beer];b=a+oldr.fraction*(anchors.loc[int(oldr.anchor),features].to_numpy(float)-a)
        b[protected]=a[protected] if oldr.policy=='preserve' else np.minimum(b[protected],a[protected])
        np.testing.assert_allclose(after.loc[beer],b,atol=1e-10,rtol=0)
        assert (b[protected]<=a[protected]+1e-10).all()
        assert (b[other]>=tripel.min(0)[other]-1e-10).all() and (b<=tripel.max(0)+1e-10).all()
        aa,bb=np.sqrt(((z(np.vstack([a,b]))-target_center)**2).mean(1));change=np.sqrt(((z(b)-z(a))**2).mean())
        np.testing.assert_allclose([r.source_target_distance,r.target_distance,r.change_distance,r.objective,r.distance_reduction_fraction],
            [aa,bb,change,.7*bb+.3*change,1-bb/aa],atol=1e-12)
        assert bb<aa and r.source_missing_feature_count==np.isnan(x[i]).sum()
        conditional=bool(np.isnan(x[i,protected]).any());assert r.uses_reference_zero_cap==conditional
        assert r.status==('model_candidate_with_reference_zero_assumption' if conditional else 'model_candidate_under_proxy_constraints')
        for field,defect in zip(summary['config']['protected_features'],['DMS','Diacetyl','4VG','trans-2-nonenal']):
            d=defects.loc[(beer,defect)]
            np.testing.assert_allclose([d.source_constraint_cap,d.after_coordinate,d.coordinate_delta_to_cap],
                [before.loc[beer,field],after.loc[beer,field],after.loc[beer,field]-before.loc[beer,field]],atol=1e-10)
            assert d.passed_under_assumptions and not d.threshold_used_for_pass
            assert bool(d.conditional_source_bound)==bool(np.isnan(x[i,features.index(field)]))
        c=changes.loc[beer].loc[features]
        np.testing.assert_allclose(c.source_csv_coordinate,x[i],equal_nan=True,atol=1e-12)
        np.testing.assert_allclose(c.delta_from_prepared_source,b-a,atol=1e-10)
        for model in classes.values():assert model.predict(b[None])[0]=='Tripel'
    def score_check(values, table, key):
        keys=list(values.index);table=table.set_index([key,'target'])
        assert len(table)==len(keys)*52 and not table.index.duplicated().any()
        def matrix(field,ts):return table[field].unstack().loc[keys,ts].to_numpy(float)
        standard=card['model'].predict(values.to_numpy(float))
        np.testing.assert_allclose(matrix('standardized_prediction',targets),standard,atol=1e-10)
        imputed=card['grade_model'].named_steps['impute'].transform(values.to_numpy(float))
        classifier=card['grade_model'].named_steps['model'];prob=classifier.predict_proba(imputed)
        grades=np.column_stack([labels[(np.cumsum(p,axis=1)>=.5).argmax(1)] for p,labels in zip(prob,classifier.classes_)])
        np.testing.assert_array_equal(matrix('estimated_grade_0_6',targets),grades)
        assert np.isfinite(matrix('estimated_grade_0_6',all_targets)).all()
        proxy=table.loc[(slice(None),extra),:]
        assert proxy.standardized_prediction.isna().all() and proxy.relative_grade_0_6.isna().all()
        assert proxy.relative_grade_forecast_lower90.isna().all() and not proxy.panel_target_available.any()
        # Only the declared below-threshold scenario is populated for current spectra.
        assert (values.acetaldehyde<np.log10(card['scorecard_protocol']['reference_threshold_mg_L'])).all()
        assert proxy.estimated_grade_0_6.eq(0).all() and not table.original_card_score_recovered.any()
    gen_scores=read('generated_sensory52_long.csv');score_check(generated[features],gen_scores,'generated_id')
    sensory=read('transfer_sensory52_long.csv',dtype={'beer':str})
    for suffix,values in [('before',before),('after',after)]:
        columns={c:c[:-len(suffix)-1] for c in sensory if c.endswith('_'+suffix)}
        table=sensory[['beer','target',*columns]].rename(columns=columns)
        score_check(values,table,'beer')
    for target in targets:
        rows=sensory[sensory.target.eq(target)].set_index('beer').loc[pils]
        np.testing.assert_allclose(rows.source_observed_standardized,yf.iloc[[lookup[b] for b in pils]][target],atol=1e-12)
    assert sensory.loc[sensory.target.isin(extra),'source_observed_standardized'].isna().all()
    assert not sensory.actual_after_measured.any()
    for filename,long,key,ts in [('generated_grades_all52.csv',gen_scores,'generated_id',all_targets),
        ('generated_grades_primary47.csv',gen_scores,'generated_id',primary),
        ('transfer_grades_before_all52.csv',sensory,'beer',all_targets),
        ('transfer_grades_after_all52.csv',sensory,'beer',all_targets),('transfer_grades_after_primary47.csv',sensory,'beer',primary)]:
        table=read(filename,dtype={key:str}).set_index(key);field='estimated_grade_0_6'
        if key=='beer':field+='_before' if 'before' in filename else '_after'
        expected=long.pivot(index=key,columns='target',values=field).loc[table.index,ts]
        np.testing.assert_array_equal(table[ts],expected)
    proxy=read('stale_proxy_checks.csv',dtype={'beer':str}).set_index(['beer','model'])
    models={'previous_multioutput':base['model'],'ridge':base['guard'],'focused':joblib.load(args.style_models/'focused_stale.joblib')}
    for name,model in models.items():
        a,b=model.predict(before.to_numpy()),model.predict(after.to_numpy())
        if name!='focused':a,b=a[:,targets.index('stale_hops')],b[:,targets.index('stale_hops')]
        actual=proxy.xs(name,level='model').loc[pils]
        np.testing.assert_allclose(actual[['before_standardized_proxy','after_standardized_proxy','delta']],np.column_stack([a,b,b-a]),atol=1e-10)
        assert (b<=a+1e-10).all() and actual.passed.all() and not actual.calibrated_confidence_bound.any()
    s=sensory[sensory.target.eq('stale_hops')].set_index('beer').loc[pils]
    assert (s.estimated_grade_0_6_after<=s.estimated_grade_0_6_before).all()
    for beer in pils:
        d=defects.loc[(beer,'Stale hops')]
        np.testing.assert_allclose([d.before_relative_grade,d.after_relative_grade,d.max_standardized_proxy_delta],
            [s.loc[beer,'estimated_grade_0_6_before'],s.loc[beer,'estimated_grade_0_6_after'],proxy.loc[beer].delta.max()],atol=1e-10)
        assert d.passed_under_assumptions
    for actual,parent in [('generation_model_comparison.csv',args.q3/'validation_summary.csv'),
        ('generation_validation_by_style_fold.csv',args.q3/'validation_by_style_fold.csv'),
        ('stale_loss_comparison.csv',args.resolution/'stale_loss_comparison.csv'),('reference_holdout_metrics.csv',args.holdout/'metrics.csv')]:
        pd.testing.assert_frame_equal(read(actual),pd.read_csv(parent))
    fingerprints=read('sensory_fingerprints52.csv').set_index(['style','target'])
    for (style,target),group in gen_scores.groupby(['style','target']):
        r=fingerprints.loc[(style,target)];values=group.estimated_grade_0_6
        np.testing.assert_allclose([r.mean_estimated_grade,r.p10_estimated_grade,r.p90_estimated_grade],[values.mean(),values.quantile(.1),values.quantile(.9)],atol=1e-12)
        assert r.n==100
    comparison=read('transfer_method_comparison.csv',dtype={'beer':str}).set_index('beer')
    robust=pd.read_csv(args.robust/'transfer_results.csv',dtype={'beer':str}).set_index('beer')
    np.testing.assert_array_equal(comparison.loc[pils,'reference_1pct_IQR_selected'],robust.loc[pils,'selected'])
    assert not summary['main_path_1pct_box_verified'] and not summary['actual_after_stale_validated']
    report={'status':'passed','generated_profiles':300,'generated_score_rows':15600,'transfer_sources':11,
        'transfer_before_after_score_pairs':572,'five_defect_checks':55,'stale_proxy_checks':33,'coordinate_changes':2464,
        'conditional_zero_cap_sources':['119','28'],'actual_after_stale_validated':False,
        'checks':['source and parent hashes','300 parent reconstructions, seeds and source missingness',
            'generated spectra, style diagnostics and no exact training copies','11 paths reconstructed from source and anchors',
            '44 coordinate inequalities and supported conditional zero caps','33 fitted Stale deltas and 11 task2 grade changes',
            '16744 task2 score rows independently matched to classifier probabilities and threshold rule',
            '52/47-column wide tables and 156 fingerprint summaries','retained generation/baseline/holdout evidence',
            'all 11 source statuses, measured observations and separate reference-box scope'],
        'interpretation':'Implementation and declared model scenarios checked; not a causal or physical nonworsening certificate.'}
    write_json(out/'verification.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):verify(parser().parse_args())
