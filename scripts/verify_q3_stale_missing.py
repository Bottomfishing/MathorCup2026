"""Verify focused Stale failure and fixed missing-aware Q3 adoption boundaries."""
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
from beer_audit.audit import sha256,write_json
from beer_models.stale_risk import StaleNonzeroModel
from beer_models.tail_risk import base_probabilities,binary_loss
from beer_models.robust_transfer import panel_five,style_margins,refit_metrics
from beer_models.generation import StyleGenerator
from beer_models.explain import stratified_draw
from scipy.spatial.distance import cdist
from q3_risk_aware_pruning import parameter_draws,FIRST
from r2_raw_q1 import load_inputs,PROTOCOL as BASE,RAW_MANIFEST,content_hash
from beer_models.raw_panel import fit_shared


def run(args):
    bp=json.loads(BASE.read_text(encoding='utf-8'));d=load_inputs(args.source,args.raw,bp)
    x,styles=d['x'],d['ids'].tasting_category_fine.to_numpy()
    stale=args.stale;out=args.output
    p=json.loads((stale/'protocol.json').read_text(encoding='utf-8'));summary=json.loads((stale/'summary.json').read_text(encoding='utf-8'))
    assert content_hash(p)==summary['protocol_sha256'] and sha256(args.raw)==summary['raw_sha256']
    for name,digest in summary['source_sha256'].items():assert sha256(args.source/name)==digest
    aging=[d['features'].index(f) for f in pd.read_csv(stale/'aging_features.csv').field_name_R]
    assert len(aging)==24
    dictionary=pd.read_csv(ROOT/'outputs/p0_audit_20261023_final/feature_dictionary.csv').set_index('field_name_R').loc[d['features']]
    np.testing.assert_array_equal(aging,np.flatnonzero(dictionary.aging.notna().to_numpy()))
    sj=d['targets'].index('stale_hops');y=1-d['distribution'][:,sj,0]
    table=pd.read_csv(stale/'oof_predictions.csv',float_precision='round_trip')
    inner=pd.read_csv(stale/'inner_selection.csv');splits=pd.read_csv(stale/'split_membership.csv')
    stale_replays=0
    for repeat in range(3):
        seen=[]
        for fold in range(3):
            cache=joblib.load(stale/'models'/f'r{repeat}_fold{fold}.joblib')
            train,test=cache['train'],cache['test'];assert not set(train)&set(test);seen.extend(test)
            inside=splits.loc[splits.repeat.eq(repeat)&splits.outer_fold.eq(fold)&splits.inner_fold.ge(0)]
            assert set(inside.beer).issubset(set(d['ids'].beer.iloc[train]))
            for _,group in inside.groupby('inner_fold'):
                assert not set(group.loc[group.role.eq('train'),'beer'])&set(group.loc[group.role.eq('validation'),'beer'])
            losses=inner.loc[inner.repeat.eq(repeat)&inner.outer_fold.eq(fold)].set_index('method').loc[p['stale_candidates']]
            expected=next(name for name in p['stale_candidates'] if losses.loc[name,'brier']<=losses.brier.min()+1e-12)
            assert expected==cache['selected']
            for name,model in cache['models'].items():
                if name=='shared_et':prob=1-base_probabilities(model,x[test])[:,sj,0]
                else:prob=model.predict(x[test])
                saved=table.loc[table.method.eq(name)&table.repeat.eq(repeat)].set_index('beer').loc[d['ids'].beer.iloc[test]]
                np.testing.assert_allclose(saved.prediction,prob,atol=1e-10,rtol=0)
                np.testing.assert_allclose(saved.expected_brier,binary_loss(prob,y[test]),atol=1e-12,rtol=0)
                if name!='training_prior':
                    cols=np.arange(x.shape[1]) if name in ['shared_et','all_et_leaf10'] else aging
                    np.testing.assert_allclose(model.imputer.statistics_,np.nanmedian(x[train][:,cols],axis=0),atol=0,rtol=0)
                else:
                    np.testing.assert_allclose(model.prior,y[train].mean(),atol=0,rtol=0)
                if repeat==0 and fold==0 and name not in ['shared_et','training_prior']:
                    fresh=StaleNonzeroModel(name,aging,p['model_seed']+99).fit(x[train],y[train])
                    np.testing.assert_allclose(fresh.predict(x[test]),prob,atol=1e-10,rtol=0)
                stale_replays+=len(test)
            chosen=table.loc[table.method.eq('inner_selected')&table.repeat.eq(repeat)].set_index('beer').loc[d['ids'].beer.iloc[test]]
            selected_model=cache['models'][cache['selected']]
            expected_prob=1-base_probabilities(selected_model,x[test])[:,sj,0] if cache['selected']=='shared_et' else selected_model.predict(x[test])
            np.testing.assert_allclose(chosen.prediction,expected_prob,atol=1e-10,rtol=0)
        np.testing.assert_array_equal(np.sort(seen),np.arange(250))
    assert not summary['adopted'] and not (stale/'model.joblib').exists()
    prior_intervals=pd.read_csv(stale/'paired_intervals.csv').set_index('baseline')
    rng=np.random.default_rng(p['bootstrap']['seed'])
    selected_loss=table.loc[table.method.eq('inner_selected')].pivot(index='repeat',columns='beer',values='expected_brier').loc[:,d['ids'].beer].to_numpy().mean(0)
    for baseline,quantiles in [('training_prior',[.005,.995]),('shared_et',[.025,.975])]:
        baseline_loss=table.loc[table.method.eq(baseline)].pivot(index='repeat',columns='beer',values='expected_brier').loc[:,d['ids'].beer].to_numpy().mean(0)
        delta=baseline_loss-selected_loss
        draws=[delta[stratified_draw(styles,rng)].mean() for _ in range(p['bootstrap']['repeats'])]
        low,high=np.quantile(draws,quantiles)
        np.testing.assert_allclose(prior_intervals.loc[baseline,['mean_gain','lower','upper']].to_numpy(float),[delta.mean(),low,high],atol=1e-12,rtol=0)
    # Fixed223-input family: reproduce ALL9 outer predictions, then one independent refit.
    qualification=json.loads((out/'predictive_qualification.json').read_text(encoding='utf-8'))
    mp=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    assert qualification['protocol_sha256']==content_hash(mp) and qualification['raw_sha256']==sha256(args.raw)
    omitted=d['features'].index(mp['omitted_input']);keep=np.array([i for i in range(224) if i!=omitted])
    small=dict(d,x=x[:,keep],features=[d['features'][i] for i in keep])
    indices=[d['targets'].index(t) for t in mp['sensory_targets']];five=indices+[sj]
    q=1-d['distribution'][:,indices,0]
    predictions=pd.read_csv(out/'oof_predictions.csv',float_precision='round_trip')
    masked_replays=0
    for repeat in range(3):
        for fold in range(3):
            cache=joblib.load(out/'models'/f'outer_{repeat}_{fold}.joblib');train,test=cache['train'],cache['test']
            assert not set(train)&set(test)
            model=cache['panel'];np.testing.assert_allclose(model.imputer.statistics_,np.nanmedian(small['x'][train],axis=0),atol=0,rtol=0)
            prob=1-base_probabilities(model,small['x'][test])[:,indices,0]
            for j,target in enumerate(mp['sensory_targets']):
                saved=predictions.loc[predictions.method.eq('masked')&predictions.repeat.eq(repeat)&predictions.target.eq(target)].set_index('beer').loc[d['ids'].beer.iloc[test]]
                np.testing.assert_allclose(saved.probability,prob[:,j],atol=1e-10,rtol=0)
                np.testing.assert_allclose(saved.Brier,binary_loss(prob[:,j],q[test,j]),atol=1e-12,rtol=0)
            if repeat==0 and fold==0:
                fresh=fit_shared(small,train,bp,model.seed)
                np.testing.assert_allclose(1-base_probabilities(fresh,small['x'][test])[:,indices,0],prob,atol=1e-10,rtol=0)
            masked_replays+=len(test)
    rng=np.random.default_rng(20261207);draws=np.stack([stratified_draw(styles,rng) for _ in range(2000)])
    intervals=pd.read_csv(out/'paired_intervals.csv').set_index('target')
    for target in mp['sensory_targets']:
        values={name:predictions.loc[predictions.target.eq(target)&predictions.method.eq(name)].pivot(index='repeat',columns='beer',values='Brier').loc[:,d['ids'].beer].to_numpy().mean(0) for name in ['prior','full','masked']}
        gain=values['prior']-values['masked'];increase=values['masked']-values['full']
        low,high=np.quantile(gain[draws].mean(1),[.005,.995]);l95,h95=np.quantile(increase[draws].mean(1),[.025,.975])
        for col,value in [('prior_gain',gain.mean()),('prior_lower99',low),('prior_upper99',high),('full_Brier_increase',increase.mean()),('full_lower95',l95),('full_upper95',h95)]:
            np.testing.assert_allclose(intervals.loc[target,col],value,atol=1e-12,rtol=0)
        assert low>0 and increase.mean()<=.0005 and h95<=.001
    # Replay fixed-plan matched cohorts, including uncertainty/nominal parameter checks.
    result=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    assert result['predictive_qualified'] and result['adopted_four_proxy_missing_aware_plan']
    official=joblib.load(args.q1/'model.joblib');assert sha256(args.q1/'model.joblib')==result['q1_model_sha256']
    source=pd.read_csv(out/'source119_profile.csv')[d['features']].to_numpy(float)[0]
    target=pd.read_csv(out/'transfer119_profile.csv')[d['features']].to_numpy(float)[0]
    assert np.isnan(source[omitted]) and np.isnan(target[omitted])
    assert np.isnan(d['x'][np.flatnonzero(d['ids'].beer.to_numpy()==119)[0],omitted])
    assert int((abs(target[keep]-source[keep])>1e-10).sum())==130
    for field in ['DMS','X4VG','X.E..2.nonenal']:
        j=d['features'].index(field);assert target[j] <= source[j]+1e-10
    source0=source.copy();target0=target.copy();source0[omitted]=0;target0[omitted]=0
    params_full=parameter_draws(source0,target0,32,20261209);params_full[:,omitted]=np.nan
    values=np.vstack([target[keep],params_full[:,keep]])
    gate=json.loads((ROOT/'outputs/q3_pruning_20261027/base_protocol.json').read_text(encoding='utf-8'))
    param_checks=pd.read_csv(out/'parameter_checks.csv')
    paired=pd.read_csv(out/'matched_draws.csv');expected=pd.read_csv(out/'transfer_checks.csv')
    for role,n,reqkey in [('development',20,'design_requirements'),('validation',12,'validation_adoption')]:
        gg,rr,mm,ss=[],[],[],[]
        for repeat in range(n):
            model=joblib.load(out/'models'/f'{role}_{repeat}.joblib')
            order=paired.loc[paired.partition.eq(role)&paired.repeat.eq(repeat)].sort_values('occurrence').beer
            train_indices=np.array([{int(beer):k for k,beer in enumerate(d['ids'].beer)}[int(beer)] for beer in order])
            np.testing.assert_allclose(model['panel'].imputer.statistics_,np.nanmedian(x[train_indices][:,keep],axis=0),atol=0,rtol=0)
            assert model['panel'].model.n_features_in_==223
            before=panel_five(model['panel'],source[None,keep],five);after=panel_five(model['panel'],values,five)
            margin,passed=style_margins(model['classifiers'],values)
            gg.append(after['grade']-before['grade']);rr.append(after['nonzero']-before['nonzero']);mm.append(margin);ss.append(passed)
            joint=bool(passed[0] and (gg[-1][0]<=0).all() and (rr[-1][0,:4]<=1e-10).all())
            row=expected.loc[expected.role.eq(role)&expected.repeat.eq(repeat)].iloc[0]
            assert joint==bool(row.joint_pass) and bool(row.source_oob)==(119 not in set(order))
        metrics=refit_metrics(np.stack(gg),np.stack(rr),np.stack(mm),np.stack(ss),gate[reqkey])
        assert metrics['qualified'][0]
        np.testing.assert_allclose(result['role_metrics'][role]['joint_fraction'],metrics['joint_fraction'][0],atol=0,rtol=0)
        np.testing.assert_allclose(result['role_metrics'][role]['risk_delta_q80'],metrics['nonzero_upper'][0],atol=1e-10,rtol=0)
        if role=='development':np.testing.assert_array_equal(param_checks.development_qualified,metrics['qualified'][1:])
    bundle=joblib.load(out/'model.joblib');assert mp['omitted_input'] not in bundle['features'] and len(bundle['features'])==223
    # Different values, including NA and the empirical maximum, disappear under the SAME named mask.
    substitutions=np.broadcast_to(target,(4,len(target))).copy();substitutions[:,omitted]=[0,.284672,121.7310182,np.nan]
    masked=substitutions[:,bundle['keep_indices']]
    np.testing.assert_array_equal(masked,np.broadcast_to(target[keep],masked.shape))
    probability=base_probabilities(bundle['panel'],masked)
    np.testing.assert_array_equal(probability,np.broadcast_to(probability[:1],probability.shape))
    a=source[None,keep];full_before=panel_five(bundle['panel'],a,five);full_after=panel_five(bundle['panel'],values,five)
    margins,_=style_margins(bundle['classifiers'],values)
    of_before=official['raw_panel_model'].predict(source[None])['grade'][:,five]
    of_after=official['raw_panel_model'].predict(np.vstack([target,params_full]))['grade'][:,five]
    nominal=(full_after['grade']<=full_before['grade']).all(1)&(full_after['nonzero'][:,:4]<=full_before['nonzero'][:,:4]+1e-10).all(1)&(margins>=.01).all(1)&(of_after<=of_before).all(1)
    generator=StyleGenerator().fit(small['x'],styles);z=generator.robust_.transform(generator.x_);dist=cdist(z,z)/np.sqrt(223);np.fill_diagonal(dist,np.inf)
    radius=np.quantile(dist.min(1),.95);center=z[generator.indices_['Tripel']].mean(0)
    za=generator.robust_.transform(a)[0];zc=generator.robust_.transform(values);initial=np.sqrt(np.square(za-center).mean())
    nominal&=(1-np.sqrt(np.square(zc-center).mean(1))/initial>=.1)&(cdist(zc,z).min(1)/np.sqrt(223)<=radius+1e-10)
    np.testing.assert_array_equal(param_checks.nominal_qualified,nominal[1:])
    np.testing.assert_array_equal(param_checks.combined_qualified,param_checks.nominal_qualified&param_checks.development_qualified)
    assert nominal[0] and param_checks.combined_qualified.mean()>=.8
    scores=pd.read_csv(out/'official_Q1_missing_scores52.csv.gz');missing=scores.target.isin(['A_acetaldehyde','F_acetaldehyde'])
    assert len(scores)==104 and missing.sum()==4 and scores.loc[missing,'estimated_grade_0_6'].isna().all()
    assert not scores.actual_after_measured.any() and not scores.safety_certified.any()
    status=json.loads((out/'current_plan_status.json').read_text(encoding='utf-8'))
    assert not status['exact_zero_assumption_required'] and status['source_diacetyl_numeric_value'] is None
    assert not status['stale_evidence_sufficient'] and not status['all_five_true_nonworsening_certified']
    write_json(out/'verification.json',dict(status='passed',stale_outer_probability_replays=stale_replays,
        independent_focused_Stale_refits=4,masked_outer_beer_replays=masked_replays,independent_masked_shared_refits=1,
        matched_transfer_models_replayed=32,parameter_draws_replayed=32,source_NA_preserved=True,
        omitted_coordinate_invariance_checked=True,formal_Q1_model_unchanged=True,Stale_unresolved=True,
        verification_is_not_new_external_validation=True))
    print('Focused Stale and missing-aware Q3 checks passed.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--raw',type=Path,default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1',type=Path,default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--stale',type=Path,default=ROOT/'outputs/q3_stale_risk_20261027')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/q3_missing_aware_20261027')
    with threadpool_limits(limits=1):run(parser.parse_args())
