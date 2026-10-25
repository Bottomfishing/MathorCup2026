"""Assemble the selected Q3 design with task2 scores and explicit five-defect evidence."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, sha256, write_json
from beer_models.scorecard import predict_scorecard_frame
from beer_models.transfer_checks import coordinate_nonincrease, transfer_status
from complete_modeling import load_q1_artifacts


def run(args):
    inputs = {}
    def read(path, **kwargs):
        inputs[str(path)] = sha256(path)
        return pd.read_csv(path, **kwargs)
    def meta(path):
        inputs[str(path)] = sha256(path)
        return json.loads(path.read_text(encoding='utf-8'))
    def model(path):
        inputs[str(path)] = sha256(path)
        return joblib.load(path)
    cfg = meta(Path('configs/q3_task3_protocol.json'))
    card_summary = meta(args.scorecard / 'summary.json')
    assert meta(args.scorecard / 'verification.json')['status'] == 'passed'
    assert meta(args.resolution / 'verification.json')['status'] == 'passed'
    parent = meta(args.resolution / 'summary.json')
    generation_config = meta(args.q1 / 'run_manifest.json')['config']
    ids, xf, yf, base, _ = load_q1_artifacts(args.source, Path('outputs/p0_audit_20261023_final'), args.q1)
    inputs[str(args.q1 / 'model.joblib')] = sha256(args.q1 / 'model.joblib')
    assert inputs[str(args.q1 / 'model.joblib')] == card_summary['parent_q1_model_sha256']
    for name, digest in parent['source_sha256'].items():
        assert sha256(args.source / name) == digest
    for name, digest in card_summary['source_sha256'].items():
        assert sha256(args.source / name) == digest
    card = model(args.scorecard / 'model.joblib')
    assert inputs[str(args.scorecard / 'model.joblib')] == card_summary['model_sha256']
    classifiers = {n: model(args.style_models / (n+'.joblib')) for n in ['logistic', 'extra_trees']}
    focused = model(args.style_models / 'focused_stale.joblib')
    for name, digest in parent['style_models_sha256'].items():
        assert inputs[str(args.style_models / (name+'.joblib'))] == digest
    generator = model(args.q3 / 'generator.joblib')
    x = xf.to_numpy(float); beers = ids.beer.astype(str).to_numpy(); lookup = {b:i for i,b in enumerate(beers)}
    np.testing.assert_allclose(generator.x_, generator.impute_.transform(x), atol=0, rtol=0)
    assert card['features'] == list(xf) and card['targets'] == list(yf)
    generated = read(args.q3 / 'generated_profiles.csv')
    metadata = read(args.q3 / 'generated_metadata.csv', dtype={'parent_1': str, 'parent_2': str})
    gen_checks = read(args.delivery / 'generated_profile_diagnostics.csv')
    gen_class = read(args.style_models / 'generated_style_diagnostics.csv')
    validation = read(args.q3 / 'validation_summary.csv')
    validation_folds = read(args.q3 / 'validation_by_style_fold.csv')
    evidence = read(args.resolution / 'source_defect_evidence.csv', dtype={'beer': str})
    previous = read(args.resolution / 'transfer_results.csv', dtype={'beer': str}).set_index('beer')
    after = read(args.resolution / 'transfer_profiles.csv', dtype={'beer': str}).set_index('beer')
    anchors = read(args.resolution / 'anchors.csv').set_index('anchor')
    defects = read(Path('outputs/task1_acceptance_20261025/defect_scope.csv')).fillna('')
    pils = beers[ids.tasting_category_fine.eq('Pils/Lager')]
    before = pd.DataFrame(generator.x_[[lookup[b] for b in pils]], index=pils, columns=xf.columns)
    before.index.name = 'beer'
    for row in evidence.itertuples():
        before.loc[row.beer, row.field] = row.constraint_cap_coordinate
    assert set(after.index) == set(pils) and set(previous.index) == set(pils)
    after = after.loc[pils, xf.columns].copy()
    # Preserve all spectra; no filtering of poorly classified generated profiles.
    generated_input = generated[xf.columns].copy(); generated_input.insert(0, 'beer', generated.generated_id)
    gen_long, gen_primary, gen_wide, _ = predict_scorecard_frame(generated_input, card)
    gen_long = gen_long.rename(columns={'beer':'generated_id'})
    gen_long.insert(1, 'style', gen_long.generated_id.map(generated.set_index('generated_id')['style']))
    gen_long['prediction_role'] = np.where(gen_long.panel_target_available,
        'synthetic_spectrum_full_fit_inference_not_OOF', 'synthetic_spectrum_threshold_scenario_no_panel_label')
    for table in [gen_primary, gen_wide]:
        table.rename(columns={'beer':'generated_id'}, inplace=True)
        table.insert(1, 'style', table.generated_id.map(generated.set_index('generated_id')['style']))
    pre_long, _, pre_wide, _ = predict_scorecard_frame(before.reset_index(), card)
    post_long, post_primary, post_wide, _ = predict_scorecard_frame(after.reset_index(), card)
    transfer_sensory = pre_long.merge(post_long, on=['beer','target'], suffixes=('_before','_after'), validate='one_to_one')
    actual = yf.copy(); actual.insert(0,'beer',beers)
    observed = actual.melt(id_vars='beer', var_name='target', value_name='source_observed_standardized')
    transfer_sensory = transfer_sensory.merge(observed, on=['beer','target'], how='left', validate='one_to_one')
    transfer_sensory['source_input_role'] = 'prepared_constraint_scenario_zero_caps_are_not_measurements'
    transfer_sensory['after_input_role'] = 'synthetic_path_candidate_not_observed'
    transfer_sensory['actual_after_measured'] = False
    zreal = generator.robust_.transform(generator.x_)
    center = zreal[ids.tasting_category_fine.eq('Tripel')].mean(axis=0)
    tripel = generator.x_[ids.tasting_category_fine.eq('Tripel')]
    protected = [list(xf).index(f) for f in cfg['protected_features']]
    other = [j for j in range(len(xf.columns)) if j not in protected]
    models = {'previous_multioutput':base['model'], 'ridge':base['guard'], 'focused':focused}
    proxy_before = {}; proxy_after = {}; stale_index = list(yf).index('stale_hops')
    for name, m in models.items():
        a, b = m.predict(before.to_numpy()), m.predict(after.to_numpy())
        proxy_before[name] = a if name == 'focused' else a[:, stale_index]
        proxy_after[name] = b if name == 'focused' else b[:, stale_index]
    classifications = {n:m.predict(after.to_numpy()) for n,m in classifiers.items()}
    source_evidence = evidence.set_index(['beer','field']); records=[]; defect_rows=[]; proxy_rows=[]; changes=[]
    for k, beer in enumerate(pils):
        i=lookup[beer]; a=before.loc[beer].to_numpy(float); b=after.loc[beer].to_numpy(float)
        r=previous.loc[beer]; za,zb=generator.robust_.transform(np.vstack([a,b]))
        da,db=np.sqrt(np.mean((np.vstack([za,zb])-center)**2,axis=1)); change=float(np.sqrt(np.mean((zb-za)**2)))
        raw=a+float(r.fraction)*(anchors.loc[int(r.anchor),xf.columns].to_numpy(float)-a)
        raw[protected]=a[protected] if r.policy=='preserve' else np.minimum(raw[protected],a[protected])
        np.testing.assert_allclose(b,raw,atol=1e-10,rtol=0)
        range_ok=bool((b[other]>=tripel.min(0)[other]-1e-10).all() and (b<=tripel.max(0)+1e-10).all())
        chemical_passes=[]; uses_zero=False
        for d in defects.loc[defects.chemical_input_1csv.ne('')].itertuples():
            field=d.chemical_input_1csv; e=source_evidence.loc[(beer,field)]
            conditional=e.basis!='observed_coordinate'; uses_zero |= conditional
            passed=coordinate_nonincrease(after.loc[beer,field], e.constraint_cap_coordinate); chemical_passes.append(passed)
            defect_rows.append({'beer':beer,'defect':d.defect,'chemical_field':field,'sensory_target':d.sensory_target_4csv,
                'check_kind':'monotone_coordinate_nonincrease','source_csv_coordinate':xf.iloc[i][field],
                'source_constraint_cap':e.constraint_cap_coordinate,'after_coordinate':after.loc[beer,field],
                'coordinate_delta_to_cap':after.loc[beer,field]-e.constraint_cap_coordinate,
                'reference_threshold':d.reference_threshold_median,'reference_unit':d.reference_unit,
                'source_evidence':e.basis,'conditional_source_bound':conditional,
                'threshold_used_for_pass':False,'passed_under_assumptions':passed,'actual_after_validated':False,
                'assumption':'same_monotone_coordinate_map_before_after'+('; nonnegative Diacetyl reference-zero cap' if conditional else '')})
        passes=[]
        for name in models:
            aa=proxy_before[name][k]; bb=proxy_after[name][k]; ok=bool(bb<=aa+1e-10); passes.append(ok)
            proxy_rows.append({'beer':beer,'model':name,'before_standardized_proxy':aa,'after_standardized_proxy':bb,
                'delta':bb-aa,'passed':ok,'calibrated_confidence_bound':False,'actual_after_measured':False})
        ga=int(pre_wide.set_index('beer').loc[beer,'stale_hops']); gb=int(post_wide.set_index('beer').loc[beer,'stale_hops'])
        grade_ok=gb<=ga
        defect_rows.append({'beer':beer,'defect':'Stale hops','chemical_field':'','sensory_target':'stale_hops',
            'check_kind':'three_standardized_point_proxies_and_task2_relative_grade',
            'before_relative_grade':ga,'after_relative_grade':gb,'max_standardized_proxy_delta':max(proxy_after[n][k]-proxy_before[n][k] for n in models),
            'source_observed_standardized':yf.iloc[i].stale_hops,'reference_unit':'standardized_panel_score_or_relative_grade',
            'conditional_source_bound':uses_zero,'threshold_used_for_pass':False,
            'passed_under_assumptions':all(passes) and grade_ok,'actual_after_validated':False,
            'assumption':'fitted_response_is_a_scenario_proxy; no reliable causal or measured guarantee'})
        style_ok=all(v[k]=='Tripel' for v in classifications.values())
        checks=[*chemical_passes,*passes,grade_ok,style_ok,range_ok,bool(db<da-1e-8)]
        status=transfer_status(checks,uses_zero)
        records.append({'beer':beer,'source_style':'Pils/Lager','target_style':'Tripel','main_method':cfg['main_transfer'],
            'selected':status.startswith('model_candidate'),'status':status,'original_path_status':r.status,
            'policy':r.policy,'anchor':int(r.anchor),'fraction':r.fraction,'source_target_distance':da,'target_distance':db,
            'change_distance':change,'objective':.7*db+.3*change,'distance_reduction_fraction':1-db/da,
            'source_missing_feature_count':int(np.isnan(x[i]).sum()),'uses_reference_zero_cap':uses_zero,
            'chemical_checks_passed':all(chemical_passes),'three_stale_proxies_passed':all(passes),
            'stale_grade_before':ga,'stale_grade_after':gb,'task2_stale_grade_passed':grade_ok,
            'logistic_style':classifications['logistic'][k],'extra_trees_style':classifications['extra_trees'][k],
            'empirical_nondefect_ranges_passed':range_ok,'actual_after_stale_measured':False,'physical_feasibility_certified':False})
        for j,field in enumerate(xf):
            changes.append({'beer':beer,'field':field,'source_csv_coordinate':x[i,j],'prepared_source_coordinate':a[j],
                'after_coordinate':b[j],'delta_from_prepared_source':b[j]-a[j],'source_csv_missing':bool(np.isnan(x[i,j])),
                'uses_reference_zero_cap':field=='X2.3.butanedione' and uses_zero,'protected':j in protected})
    generated_diagnostics=gen_checks.merge(gen_class,on=['generated_id','style'],validate='one_to_one')
    metadata['parent_1_missing_features']=[int(np.isnan(x[lookup[b]]).sum()) for b in metadata.parent_1]
    metadata['parent_2_missing_features']=[int(np.isnan(x[lookup[b]]).sum()) for b in metadata.parent_2]
    style_lookup=generated.set_index('generated_id')['style']
    seeds={s:generation_config['seed']+80000+i for i,s in enumerate(generation_config['generation_styles'])}
    metadata['seed']=metadata.generated_id.map(style_lookup).map(seeds)
    metadata['method']='interpolate'
    fingerprints=gen_long.groupby(['style','target'],sort=False).agg(n=('estimated_grade_0_6','size'),
        primary=('primary','first'),grade_kind=('grade_kind','first'),mean_estimated_grade=('estimated_grade_0_6','mean'),
        p10_estimated_grade=('estimated_grade_0_6',lambda s:s.quantile(.1)),p90_estimated_grade=('estimated_grade_0_6',lambda s:s.quantile(.9)),
        mean_standardized_prediction=('standardized_prediction','mean')).reset_index()
    nominal=read(args.reference/'transfer_results.csv',dtype={'beer':str}).set_index('beer')
    robust=read(args.robust/'transfer_results.csv',dtype={'beer':str}).set_index('beer')
    comparison=[]
    for row in records:
        beer=row['beer']
        comparison.append({'beer':beer,'main_method':cfg['main_transfer'],'main_status':row['status'],
            'path_distance_reduction':row['distance_reduction_fraction'],
            'nominal_reference_beer':nominal.loc[beer,'reference_beer'],
            'nominal_reference_distance_reduction':nominal.loc[beer,'distance_reduction_fraction'],
            'reference_1pct_IQR_selected':bool(robust.loc[beer,'selected']),'reference_1pct_IQR_status':robust.loc[beer,'status'],
            'radius_applies_to':'reference_design_only_not_a_test_of_selected_path',
            'decision':'primary_path_uses_direct_task2_predictions; reference_anchoring_lacks_holdout_advantage'})
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    tables={'generated_profiles.csv':generated,'generated_metadata.csv':metadata,'generated_diagnostics.csv':generated_diagnostics,
        'generation_model_comparison.csv':validation,'generation_validation_by_style_fold.csv':validation_folds,
        'generated_sensory52_long.csv':gen_long,'generated_grades_all52.csv':gen_wide,'generated_grades_primary47.csv':gen_primary,
        'sensory_fingerprints52.csv':fingerprints,'transfer_results.csv':records,'transfer_source_profiles.csv':before.reset_index(),
        'transfer_profiles.csv':after.reset_index(),'transfer_coordinate_changes.csv':changes,'transfer_sensory52_long.csv':transfer_sensory,
        'transfer_grades_before_all52.csv':pre_wide,'transfer_grades_after_all52.csv':post_wide,'transfer_grades_after_primary47.csv':post_primary,
        'five_defect_checks.csv':defect_rows,'stale_proxy_checks.csv':proxy_rows,'source_defect_evidence.csv':evidence,
        'transfer_method_comparison.csv':comparison,'stale_loss_comparison.csv':read(args.resolution/'stale_loss_comparison.csv'),
        'reference_holdout_metrics.csv':read(args.holdout/'metrics.csv')}
    for name,data in tables.items():save_csv(out,name,data)
    write_json(out/'summary.json',{'status':'completed','config':cfg,'input_sha256':inputs,
        'source_sha256':{n:sha256(args.source/n) for n in ['1.csv','4.csv','8.xlsx']},
        'generated_profiles':len(generated),'generated_score_rows':len(gen_long),'fingerprint_rows':len(fingerprints),
        'transfer_sources':len(pils),'transfer_score_rows':len(transfer_sensory),'five_defect_rows':len(defect_rows),
        'transfer_status_counts':pd.Series([r['status'] for r in records]).value_counts().to_dict(),
        'new_models_trained':0,'new_candidate_searches':0,'original_card_intensity_recovered':False,
        'actual_after_stale_validated':False,'main_path_1pct_box_verified':False,
        'limitations':['Grades are task2 relative or threshold-scenario estimates, not recovered raw card intensity.',
            'Stale regressors do not reliably beat the median baseline; all guards are fitted-model scenarios.',
            'Reference holdout diagnostics do not establish an advantage; reference 1% IQR checks concern separate designs.',
            'No paired intervention data, physical mass balance or brewing validation is supplied.']})
    print(f'Q3 task3: {len(gen_long)} generated scores; {len(transfer_sensory)} transfer pairs; {len(defect_rows)} defect checks.')


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    for name,default in {'q1':'outputs/final_20261024/q1','scorecard':'outputs/q1_task2_20261025',
        'q3':'outputs/final_20261024/q3','delivery':'outputs/q3_delivery_20261024','style-models':'outputs/q3_style_20261024',
        'resolution':'outputs/q3_resolution_20261024','reference':'outputs/q3_reference_20261024',
        'robust':'outputs/q3_reference_robust_20261024','holdout':'outputs/q3_reference_holdout_20261024',
        'output':'outputs/q3_task3_20261025'}.items():p.add_argument('--'+name,type=Path,default=Path(default))
    return p


if __name__=='__main__':
    with threadpool_limits(limits=1):run(parser().parse_args())
