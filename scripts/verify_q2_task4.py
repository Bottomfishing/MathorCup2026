"""Check task4 with one deterministic Ridge fit, saved nulls and raw graph/evidence sources."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256,write_json
from beer_audit.checks import repair_utf8_mojibake
from beer_models.baseline import load_data,make_pipeline
from beer_models.evidence import resolve_compound,annotate_relation
from complete_q2_task4 import parser


def verify(args):
    out=args.output;summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    for name,digest in summary['input_sha256'].items():assert sha256(Path(name))==digest,name
    for name,digest in summary['source_sha256'].items():assert sha256(args.source/name)==digest,name
    def read(name,**kw):return pd.read_csv(out/name,**kw)
    ids,xf,yf,primary=load_data(args.source,args.audit)
    features=list(xf);targets=list(yf)
    names=pd.read_csv(args.audit/'feature_dictionary.csv').set_index('field_name_R')
    cfg=json.loads(Path('configs/q2_exploratory_v1.json').read_text(encoding='utf-8'))
    strict_indices=np.array([i for i,f in enumerate(features) if f not in cfg['exclude_from_compound_top3']])
    family_indices=np.array([i for i,f in enumerate(features) if f not in cfg['exclude_from_compound_top3'] or f=='bitterness'])
    assert len(strict_indices)==219 and len(family_indices)==220 and len(features)==231
    fitted,_=make_pipeline('ridge_asinh',{'ridge_alphas':[1000.]},20261023)
    fitted.set_params(model__alpha=1000.).fit(xf.to_numpy(float),yf.to_numpy(float))
    coefficients=fitted.named_steps['model'].coef_
    strict=read('top3_panel_all50.csv');family=read('family_scope_top3_supplement.csv')
    original=pd.read_csv(args.delivery/'top3_all50.csv')
    pd.testing.assert_frame_equal(strict[original.columns],original,check_exact=False,atol=1e-12,rtol=0)
    scope=pd.read_csv('outputs/task1_acceptance_20261025/scorecard_scope.csv').fillna('')
    slots=read('top3_all52_slots.csv');main_slots=read('top3_primary47_slots.csv')
    assert len(slots)==156 and len(main_slots)==141 and slots.groupby('target').size().eq(3).all()
    assert set(slots.target)==set(scope.target_id) and slots.counts_as_identified_factor.sum()==150
    unknown=slots[~slots.counts_as_identified_factor]
    assert len(unknown)==6 and set(unknown.target)=={'A_acetaldehyde','F_acetaldehyde'}
    for col in ['field','ridge_signed_coef','bootstrap_frequency','within_style_selection_max_p','target_BY_q']:
        assert unknown[col].isna().all(),col
    assert unknown.slot_status.eq('unidentified_no_panel_response').all()
    known=slots[slots.counts_as_identified_factor].set_index(['target','rank']).sort_index()
    pd.testing.assert_frame_equal(known[strict.columns.difference(['target','rank'])],
        strict.set_index(['target','rank']).sort_index()[strict.columns.difference(['target','rank'])],check_dtype=False,check_exact=False,atol=1e-12,rtol=0)
    pd.testing.assert_frame_equal(read('top3_panel_primary45.csv'),strict[strict.primary].reset_index(drop=True))
    diagnostic=read('target_scope_comparison.csv').set_index('target')
    pvalues={};qvalues={};nulls={}
    def adjust_by(p):
        n=len(p);harmonic=sum(1/k for k in range(1,n+1));order=np.argsort(p,kind='stable')
        result=np.empty(n)
        for rank,index in enumerate(order):
            result[index]=min(1.,min(p[order[j]]*n*harmonic/(j+1) for j in range(rank,n)))
        return result
    for label,indices,table,file in [('strict',strict_indices,strict,'null_max_strict.csv'),('family',family_indices,family,'null_max_family.csv')]:
        null=pd.read_csv(args.reliability/file)[targets].to_numpy(float);nulls[label]=null
        assert null.shape==(1999,50)
        observed=np.abs(coefficients[:,indices]).max(1)
        p=(1+(null>=observed).sum(0))/2000;q=adjust_by(p);pvalues[label]=p;qvalues[label]=q
        for j,target in enumerate(targets):
            group=table[table.target.eq(target)].sort_values('rank');assert len(group)==3 and group['rank'].tolist()==[1,2,3]
            chosen=indices[np.argsort(-abs(coefficients[j,indices]),kind='stable')[:3]]
            assert group.field.tolist()==[features[k] for k in chosen]
            np.testing.assert_allclose(group.ridge_signed_coef,coefficients[j,chosen],atol=1e-12,rtol=0)
            np.testing.assert_allclose(group.within_style_selection_max_p,
                [(1+np.sum(null[:,j]>=abs(coefficients[j,k])))/2000 for k in chosen],atol=1e-12)
            np.testing.assert_allclose(group.target_max_statistic_p,p[j],atol=1e-12)
            np.testing.assert_allclose(group.target_BY_q,q[j],atol=1e-12)
            assert group.target_passes_BY_0_05.eq(q[j]<=.05).all()
            assert not group.causal_effect_established.any() and not group.candidate_cross_target_adjusted.any()
        pcol='within_style_max_statistic_p' if label=='strict' else 'family_within_style_max_statistic_p'
        qcol='across_50_targets_BY_q' if label=='strict' else 'family_across_50_targets_BY_q'
        np.testing.assert_allclose(diagnostic.loc[targets,pcol],p,atol=1e-12)
        np.testing.assert_allclose(diagnostic.loc[targets,qcol],q,atol=1e-12)
    assert int((qvalues['strict']<=.05).sum())==0 and int((qvalues['family']<=.05).sum())==5
    selections=pd.read_csv(args.exploratory/'bootstrap_selections.csv')
    assert len(selections)==30000 and selections.groupby(['repeat','target']).size().eq(3).all()
    assert not selections.duplicated(['repeat','target','field']).any()
    counts=selections.groupby(['target','field']).size()
    for row in strict.itertuples():
        assert abs(row.bootstrap_frequency-counts.get((row.target,row.field),0)/200)<1e-12
    saved_family=pd.read_csv(args.reliability/'family_scope_top3.csv')
    for col in ['target','rank','field','ridge_signed_coef','bootstrap_frequency','sign_agreement_all_refits','within_style_selection_max_p']:
        pd.testing.assert_series_equal(family[col],saved_family[col],check_exact=False,atol=1e-12,rtol=0)
    reference=pd.read_excel(args.reference,sheet_name='Supplementary Table S1');reference.columns=reference.columns.str.strip();reference['excel_row']=np.arange(len(reference))+2
    evidence_cfg=json.loads(Path('configs/q2_evidence.json').read_text(encoding='utf-8'))
    ledger=read('compound_name_evidence.csv').set_index('field')
    for field,row in ledger.iterrows():
        hit,method=resolve_compound(field,repair_utf8_mojibake(names.loc[field,'chemical_name']),reference,evidence_cfg['aliases'])
        assert row.reference_excel_row==hit.excel_row and row.name_resolution==method
        assert row.reference_compound==str(hit.Compound).strip() and row.reference_file_sha256==sha256(args.reference)
        assert not row.independent_mechanism_verified
    for table in [strict,family]:
        for row in table.itertuples():
            if row.field=='bitterness':
                assert row.measurement_type=='chemical_family_IBU_indicator' and row.reference_unit=='IBU'
                expected='named_family_assay_matches_bitter_target' if row.target=='bitternes' else 'target_specific_evidence_insufficient'
                assert row.chemical_evidence==expected;continue
            hit,_=resolve_compound(row.field,repair_utf8_mojibake(names.loc[row.field,'chemical_name']),reference,evidence_cfg['aliases'])
            relation,terms,rationale=annotate_relation(row.target,row.field,hit.Flavor,hit.Origin,row.ridge_signed_coef,evidence_cfg)
            assert row.chemical_evidence==relation and row.chemical_rationale==rationale
    source_threshold=pd.read_excel(args.source/'8.xlsx',sheet_name='Sensory thresholds')
    ibu=source_threshold[source_threshold.field_name_R.eq('bitterness')].iloc[0]
    assert ibu.unit=='IBU' and ibu.chemical_name=='Iso-alpha acids'
    # Reconstruct adjacency and components directly from attachment2.
    matrix=pd.read_csv(args.source/'2.csv',index_col=0)
    aliases={repair_utf8_mojibake(names.loc[f,'chemical_name']):f for f in features}
    matrix=matrix.rename(index=aliases,columns=aliases).loc[features,features].to_numpy(float)
    sensitivity=read('network_sensitivity.csv').set_index('threshold')
    nodes=read('network_nodes.csv').set_index('field').loc[features]
    for threshold in [.5,.6,.7]:
        adjacency=abs(matrix)>=threshold;np.fill_diagonal(adjacency,False)
        n,components=connected_components(adjacency,directed=False);mask=np.triu(adjacency,k=1)
        expected=[231,int(mask.sum()),int((mask&(matrix>0)).sum()),int((mask&(matrix<0)).sum()),int((adjacency.sum(1)==0).sum()),n,int(np.bincount(components).max())]
        np.testing.assert_array_equal(sensitivity.loc[threshold,['nodes','edges','positive_edges','negative_edges','isolated_nodes','components','largest_component']],expected)
        if threshold==.6:
            np.testing.assert_array_equal(nodes.degree,adjacency.sum(1))
            np.testing.assert_array_equal(nodes.component_size,np.bincount(components)[components])
    edges=read('network_edges_0_6.csv');graph=nx.read_graphml(out/'network_0_6.graphml')
    correlation_evidence=read('correlation_method_evidence.csv').set_index('method')
    saved_correlation=pd.read_csv(args.audit/'correlation_method_comparison.csv').query("file == '2.csv'").set_index('method')
    pd.testing.assert_frame_equal(correlation_evidence,saved_correlation)
    assert correlation_evidence.loc['spearman','max_abs_error']<1e-12
    assert correlation_evidence.loc['pearson','mae']>.08
    assert graph.number_of_nodes()==231 and graph.number_of_edges()==len(edges)==266 and len(edges[edges.rho.lt(0)])==3
    assert not edges.duplicated(['source','target']).any()
    position={field:i for i,field in enumerate(features)}
    for row in edges.itertuples():
        rho=matrix[position[row.source],position[row.target]]
        assert abs(rho)>=.6 and abs(row.rho-rho)<1e-12 and row.sign==('positive' if rho>0 else 'negative')
        assert abs(graph[row.source][row.target]['rho']-rho)<1e-12
    for row in strict.itertuples():
        assert row.network_degree_0_6==nodes.loc[row.field,'degree'] and row.network_component_size_0_6==nodes.loc[row.field,'component_size']
    # Missing panel targets retain named-chemical evidence but no invented rank or uncertainty.
    extra=read('extra_target_evidence.csv').set_index('target')
    forum=pd.read_csv(args.source/'7.csv',dtype={'beer':str}).set_index('beer').loc[ids.beer.astype(str)]
    for target,field in [('A_acetaldehyde','aroma_acetaldehyde'),('F_acetaldehyde','taste_acetaldehyde')]:
        r=extra.loc[target];values=forum[field]
        assert target not in yf and r.named_chemical_marker=='acetaldehyde'
        assert not r.rank_identified_from_panel_data and not r.expert_panel_labels_available and not r.forum_treated_as_expert_label
        assert pd.isna(r.marker_rank) and pd.isna(r.bootstrap_frequency) and pd.isna(r.target_BY_q)
        assert r.forum_nonmodal==values.ne(values.mode().iloc[0]).sum()
        correlation=pd.Series(values.to_numpy()).corr(pd.Series(xf.acetaldehyde.to_numpy()),method='spearman')
        np.testing.assert_allclose(correlation,r.forum_spearman_with_chemical_coordinate,atol=1e-12)
        assert r.threshold_proxy_distinct_values==1 and not r.threshold_proxy_treated_as_response
    coverage=read('target_summary52.csv').set_index('target')
    assert len(coverage)==52 and coverage.identified_statistical_factors.sum()==150 and coverage.unfilled_top3_slots.sum()==6
    for target in targets:
        group=strict[strict.target.eq(target)];r=coverage.loc[target]
        assert r.identified_statistical_factors==3 and r.top3_fields==';'.join(group.sort_values('rank').field)
        assert r.n_below_half_refit_frequency==group.bootstrap_frequency.lt(.5).sum()
        assert r.n_with_reference_support==group.chemical_evidence.ne('target_specific_evidence_insufficient').sum()
    assert int(strict.bootstrap_frequency.lt(.5).sum())==93
    assert int(strict.chemical_evidence.eq('target_specific_evidence_insufficient').sum())==114
    assert len(read('unfilled_rank_requirements.csv'))==2
    report={'status':'passed','identified_panel_associations':150,'ranked_panel_targets':50,'scope_items':52,
        'all_rank_slots':156,'unfilled_rank_slots':6,'all_template_top3_identified':False,
        'deterministic_Ridge_verification_refits':1,'recounted_bootstrap_selections':30000,
        'candidate_p_values_checked':300,'target_p_and_BY_q_pairs_checked':100,'network_nodes':231,'network_edges':266,
        'new_permutations':0,'new_bootstrap_refits':0,'causal_effects_established':False,
        'checks':['parent and source hashes','strict/family ranking and coefficients from one fixed Ridge refit',
            'saved 1999 null draws per scope: candidate maximum-statistic p and independent 50-target BY',
            '30000 strict bootstrap selections recounted; family frequencies retained from verified parent',
            'individual compound identity and descriptor annotations; separate IBU family-assay type',
            'signed attachment2 graph, GraphML, degrees/components and three thresholds',
            '52-target coverage with six genuinely empty slots; no manufactured coefficients or p-values',
            'two named-marker records and sparse forum evidence kept separate from panel responses'],
        'interpretation':'Verification establishes assembly and declared computations, not six missing factors, stability of all rankings, mechanism or causality.'}
    write_json(out/'verification.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):verify(parser().parse_args())
