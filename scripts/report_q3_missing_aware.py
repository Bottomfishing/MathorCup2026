"""Current119 missing-aware Q3 evidence, distinct from official Q1 inference."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv,write_json,sha256
from beer_models.tail_risk import base_probabilities,survival
from beer_models.robust_transfer import panel_five,style_margins,refit_metrics
from beer_models.raw_panel import portrait


def run(args):
    out=args.output
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert summary.get('predictive_qualified'), 'No qualified missing-aware risk model'
    bundle=joblib.load(out/'model.joblib');official=joblib.load(args.q1/'model.joblib')
    source=pd.read_csv(out/'source119_profile.csv');target=pd.read_csv(out/'transfer119_profile.csv')
    field=bundle['omitted_input'];assert source[field].isna().all() and target[field].isna().all()
    assert field not in bundle['features'] and len(bundle['features'])==223
    for name in ['DMS','X4VG','X.E..2.nonenal']:
        assert float(target[name].iloc[0]) <= float(source[name].iloc[0])+1e-10
    assert int((abs(target[bundle['features']].to_numpy()-source[bundle['features']].to_numpy())>1e-10).sum())==130
    frame=pd.concat([source,target],ignore_index=True)
    scores,_,_,_=portrait(frame,official)
    scores.to_csv(out/'official_Q1_missing_scores52.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    x=frame[bundle['features']].to_numpy(float)
    q=base_probabilities(bundle['panel'],x);risk=survival(q)
    grades=bundle['panel'].predict(x)['grade']
    targets=['dms','diacetyl','X4vg','t2n','stale_hops'];indices=[official['targets'].index(t) for t in targets]
    rows=[]
    for j,target_name in enumerate(targets):
        k=indices[j]
        row=dict(target=target_name,grade_before=int(grades[0,k]),grade_after=int(grades[1,k]),risk_source='Q3_missing_aware_ET223_uncalibrated',
            nonzero_validation_supported=j<4,tail_validation_established=False,Stale_safety_certified=False,
            grade_role='auxiliary_Q3_median_consistency_not_official_Q1_score')
        for event,label in [(0,'ge1'),(4,'ge5'),(5,'six')]:
            a,c=risk[0,k,event],risk[1,k,event]
            row[label+'_before'],row[label+'_after'],row[label+'_delta']=a,c,c-a
        rows.append(row)
    save_csv(out,'missing_aware_defect_before_after.csv',rows)
    styles=[]
    for name,model in bundle['classifiers'].items():
        predictions=model.predict(x);ti=list(model.classes_).index('Tripel');prob=model.predict_proba(x)
        styles.append(dict(model=name,source_prediction=predictions[0],after_prediction=predictions[1],Tripel_before=float(prob[0,ti]),Tripel_after=float(prob[1,ti]),calibration_established=False))
    save_csv(out,'nominal_styles.csv',styles)
    # Match the SAME masked model draws for the213-field historical control.
    old=pd.read_csv(ROOT/'outputs/q3_stage2_20261026/selected_profiles.csv').loc[lambda t:t.beer.eq(119)&t.domain.eq('expanded_sensitivity')]
    control=old[bundle['features']].to_numpy(float)
    values=np.vstack([x[1:],control])
    bp=json.loads((ROOT/'outputs/q3_pruning_20261027/base_protocol.json').read_text(encoding='utf-8'))
    comparison=[]
    for role,n,reqkey in [('development',20,'design_requirements'),('validation',12,'validation_adoption')]:
        gg,dd,mm,ss=[],[],[],[]
        for repeat in range(n):
            item=joblib.load(out/'models'/f'{role}_{repeat}.joblib')
            before=panel_five(item['panel'],x[:1],indices);after=panel_five(item['panel'],values,indices)
            margin,passed=style_margins(item['classifiers'],values)
            gg.append(after['grade']-before['grade']);dd.append(after['nonzero']-before['nonzero']);mm.append(margin);ss.append(passed)
        result=refit_metrics(np.stack(gg),np.stack(dd),np.stack(mm),np.stack(ss),bp[reqkey])
        for k,method in enumerate(['fixed_pruned130','previous_broad213']):
            comparison.append(dict(role=role,method=method,joint_fraction=float(result['joint_fraction'][k]),refit_qualified=bool(result['qualified'][k]),
                changed_fields=130 if k==0 else 213,control_only=k==1))
    save_csv(out,'matched_plan_comparison.csv',comparison)
    status=dict(beer=119,status='four_proxy_missing_input_qualified_Stale_UNRESOLVED' if summary['adopted_four_proxy_missing_aware_plan'] else 'not_qualified',
        adopted_four_proxy_conditional_plan=bool(summary['adopted_four_proxy_missing_aware_plan']),
        source_diacetyl_observed=False,source_diacetyl_numeric_value=None,exact_zero_assumption_required=False,
        omitted_input=field,available_model_inputs=223,changed_coordinates=130,
        source_coordinate_rule='preserve_actual_unknown_source_value',development_joint_fraction=summary['role_metrics']['development']['joint_fraction'],
        validation_joint_fraction=summary['role_metrics']['validation']['joint_fraction'],
        development_parameter_fraction=summary['role_metrics']['development']['parameter_fraction'],
        validation_parameter_fraction_additional_diagnostic=summary['role_metrics']['validation']['parameter_fraction'],
        extra_validation_parameter_diagnostic_passes_80_percent=summary['role_metrics']['validation']['parameter_fraction']>=.8,
        stale_evidence_sufficient=False,all_five_true_nonworsening_certified=False,physical_feasibility_certified=False,actual_after_measured=False,
        official_Q1_score_model_changed=False)
    write_json(out/'current_plan_status.json',status)
    old_status=pd.read_csv(ROOT/'outputs/q3_pruning_20261027/current_plan_status.csv')
    status_rows=[]
    for row in old_status.itertuples():
        if row.beer==119:
            status_rows.append(dict(beer=119,status=status['status'],adopted_four_proxy=True,assessed_this_round=True,model_inputs=223,
                model_role='missing_aware_ET223',joint_fraction=status['validation_joint_fraction'],Stale_status='unresolved',all_five_true_effect=False))
        else:
            status_rows.append(dict(beer=row.beer,status='historical_restricted_proposal_not_reassessed_with_new_model',adopted_four_proxy=False,assessed_this_round=False,
                model_inputs=224,model_role='previous_Q3_pruning_failure',joint_fraction=row.validation_joint_fraction,Stale_status='unresolved',all_five_true_effect=False))
    save_csv(out,'current_plan_status.csv',status_rows)
    summary.update(parent_fixed_target_sha256=sha256(ROOT/'outputs/q3_pruning_20261027/selected_profiles.csv'),
        current_source_profile_sha256=sha256(out/'source119_profile.csv'),current_transfer_profile_sha256=sha256(out/'transfer119_profile.csv'))
    write_json(out/'summary.json',summary)
    # Evidence figure: confidence interval for the failed Stale policy and
    # loss of source-value sensitivity in the deliberately masked Q3 model.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ci=pd.read_csv(ROOT/'outputs/q3_stale_risk_20261027/paired_intervals.csv').set_index('baseline').loc['training_prior']
    curve=pd.read_csv(ROOT/'outputs/q3_unknown_diacetyl_20261027/validation_interval_checks.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].errorbar([ci.mean_gain],[0],xerr=[[ci.mean_gain-ci.lower],[ci.upper-ci.mean_gain]],fmt='o',capsize=5,color='#b75d4b')
    axes[0].axvline(0,color='#596873',lw=1)
    axes[0].set(yticks=[0],yticklabels=['Selected Stale policy'],xlabel='Expected Brier gain vs prior (99% fixed-OOF interval)',title='Stale: interval crosses zero; NOT adopted')
    axes[1].plot(curve.diacetyl,curve.joint_fraction,label='Previous224-input model family',color='#788b9b')
    axes[1].axhline(status['validation_joint_fraction'],label='Missing-aware223-input family',color='#42877d',lw=2)
    axes[1].set_xscale('symlog',linthresh=.5)
    axes[1].set_xlim(0,float(curve.diacetyl.max()))
    axes[1].set_xticks([0,.5,1,10,100]);axes[1].set_xticklabels(['0','0.5','1','10','100'])
    axes[1].set(xlabel='Possible DA input coordinate (not a recovered measurement)',ylabel='Fixed-plan joint pass / matched12 draws',ylim=(.5,1.01),title='DA value is absent from the new predictor')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'evidence.png',dpi=160);fig.savefig(out/'evidence.svg');plt.close(fig)
    svg=out/'evidence.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q1',type=Path,default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/q3_missing_aware_20261027')
    with threadpool_limits(limits=1):run(parser.parse_args())
