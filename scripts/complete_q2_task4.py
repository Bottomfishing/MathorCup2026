"""Consolidate Q2 ranking, stability and chemical evidence without filling unsupported ranks."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import networkx as nx
import numpy as np
import pandas as pd
from beer_audit.audit import save_csv, sha256, write_json
from beer_audit.checks import repair_utf8_mojibake
from beer_models.evidence import resolve_compound, annotate_relation


def run(args):
    inputs={}
    def read(path, **kw):
        inputs[str(path)]=sha256(path);return pd.read_csv(path,**kw)
    def meta(path):
        inputs[str(path)]=sha256(path);return json.loads(path.read_text(encoding='utf-8'))
    config=meta(Path('configs/q2_task4_protocol.json'));evidence_config=meta(Path('configs/q2_evidence.json'))
    parent=meta(args.delivery/'summary.json');reliability=meta(args.reliability/'summary.json')
    assert parent['status']==reliability['status']=='passed'
    assert meta(args.reliability/'verification.json')['status']=='passed'
    scope_parent=meta(Path('outputs/task1_acceptance_20261025/verification.json'))
    for name,digest in scope_parent['source_sha256'].items():assert sha256(args.source/name)==digest,name
    for name,digest in parent['source_sha256'].items():assert sha256(args.source/name)==digest,name
    for name,digest in reliability['source_sha256'].items():assert sha256(args.source/name)==digest,name
    assert sha256(args.reference)==parent['reference_sha256'];inputs[str(args.reference)]=sha256(args.reference)
    scope=read(Path('outputs/task1_acceptance_20261025/scorecard_scope.csv')).fillna('')
    panel=read(args.delivery/'top3_all50.csv')
    enriched=read(args.reliability/'original_top3_reliability.csv')
    family=read(args.reliability/'family_scope_top3.csv')
    target_diagnostics=read(args.reliability/'target_diagnostics.csv').set_index('target')
    method_diagnostics=read(args.delivery/'target_summary.csv').set_index('target')
    names=read(args.audit/'feature_dictionary.csv').set_index('field_name_R')
    eligibility=read(args.exploratory/'factor_eligibility.csv').set_index('field')
    correlation_evidence=read(args.audit/'correlation_method_comparison.csv').query("file == '2.csv'")
    ref=pd.read_excel(args.reference,sheet_name='Supplementary Table S1')
    ref.columns=ref.columns.str.strip();ref['excel_row']=np.arange(len(ref))+2
    assert len(panel)==len(enriched)==len(family)==150
    pd.testing.assert_frame_equal(enriched[panel.columns],panel,check_exact=False,atol=1e-12,rtol=0)
    # All descriptor annotations are rechecked against the pinned local primary reference.
    resolved={};ledger=[]
    for field in sorted(set(enriched.field)|set(family.field)|{'acetaldehyde'}):
        if field=='bitterness':continue
        name=repair_utf8_mojibake(names.loc[field,'chemical_name'])
        hit,method=resolve_compound(field,name,ref,evidence_config['aliases']);resolved[field]=hit
        ledger.append({'field':field,'chemical_name':name,'name_resolution':method,
            'reference_compound':str(hit.Compound).strip(),'reference_sheet':'Supplementary Table S1',
            'reference_excel_row':int(hit.excel_row),'reference_source_url':'https://zenodo.org/records/10653704',
            'reference_file_sha256':sha256(args.reference),'compound_class':str(hit['Compound class']).strip(),
            'origin':str(hit.Origin).strip(),'reference_descriptor':str(hit.Flavor).strip() if pd.notna(hit.Flavor) else '',
            'underlying_flavor_reference':str(hit['Flavor reference']).strip() if pd.notna(hit['Flavor reference']) else '',
            'reference_unit':names.loc[field,'unit'],'unit_scope':'reference_only_not_certified_input_unit',
            'independent_mechanism_verified':False})
    network=read(args.source/'2.csv',index_col=0)
    aliases={repair_utf8_mojibake(row.chemical_name):field for field,row in names.iterrows()}
    network=network.rename(index=aliases,columns=aliases).loc[names.index,names.index]
    a=network.to_numpy(float);np.testing.assert_allclose(a,a.T,atol=1e-12)
    graph=nx.Graph(correlation_type='supplied_matrix_matches_pairwise_Spearman',absolute_threshold=.6,
        scope='full_dataset_descriptive_not_causal')
    graph.add_nodes_from(names.index)
    ii,jj=np.where(np.triu(abs(a)>=.6,k=1));edges=[]
    for i,j in zip(ii,jj):
        field,other=names.index[i],names.index[j];rho=float(a[i,j]);sign='positive' if rho>0 else 'negative'
        graph.add_edge(field,other,rho=rho,sign=sign);edges.append({'source':field,'target':other,'rho':rho,'sign':sign})
    components={field:len(c) for c in nx.connected_components(graph) for field in c}
    node_rows=[]
    for field in names.index:
        name=repair_utf8_mojibake(names.loc[field,'chemical_name'])
        graph.nodes[field].update(chemical_name=name,degree=graph.degree[field],component_size=components[field])
        node_rows.append({'field':field,'chemical_name':name,'degree':graph.degree[field],'component_size':components[field],
            'eligible_strict219':bool(eligibility.loc[field,'eligible_compound']),
            'eligible_family220':bool(eligibility.loc[field,'eligible_compound']) or field=='bitterness'})
    sensitivity=[]
    for threshold in [.5,.6,.7]:
        mask=np.triu(abs(a)>=threshold,k=1);local=nx.from_numpy_array((abs(a)>=threshold)&~np.eye(len(a),dtype=bool))
        sensitivity.append({'threshold':threshold,'nodes':len(a),'edges':int(mask.sum()),
            'positive_edges':int((mask&(a>0)).sum()),'negative_edges':int((mask&(a<0)).sum()),
            'isolated_nodes':nx.number_of_isolates(local),'components':nx.number_connected_components(local),
            'largest_component':max(map(len,nx.connected_components(local)))})
    def enrich_context(table,is_family):
        table=table.copy()
        table['ranking_scope']=config['supplementary_scope'] if is_family else config['primary_scope']
        table['label_source']='4.csv_standardized_expert_panel'
        table['interpretation']='full_data_Ridge_association_not_task2_classifier_or_causal_effect'
        table['bootstrap_repeats']=200;table['bootstrap_scheme']='within_style_refit_imputation_transform_scaling_and_Ridge'
        for col in ['ridge_vs_forest_jaccard','full_vs_analytes_jaccard','global_vs_within_style_jaccard']:
            table[col]=table.target.map(method_diagnostics[col])
        table['target_max_statistic_p']=table.target.map(target_diagnostics['family_within_style_max_statistic_p' if is_family else 'within_style_max_statistic_p'])
        table['target_BY_q']=table.target.map(target_diagnostics['family_across_50_targets_BY_q' if is_family else 'across_50_targets_BY_q'])
        table['target_passes_BY_0_05']=table.target_BY_q.le(.05)
        table['candidate_p_scope']='within_target_max_over_220' if is_family else 'within_target_max_over_219'
        table['candidate_max_p_le_0_05']=table.within_style_selection_max_p.le(.05)
        table['candidate_cross_target_adjusted']=False
        table['network_degree_0_6']=table.field.map(dict(graph.degree))
        table['network_component_size_0_6']=table.field.map(components)
        table['measurement_type']=np.where(table.field.eq('bitterness'),'chemical_family_IBU_indicator','individual_analyte_candidate')
        table['reference_unit']=table.field.map(names.unit)
        table['unit_scope']='reference_only_not_certified_input_unit'
        table['causal_effect_established']=False
        for idx,row in table.iterrows():
            if row.field=='bitterness':
                # Attachment8 identifies a family assay, not a molecule in Table S1.
                relation='named_family_assay_matches_bitter_target' if row.target=='bitternes' else 'target_specific_evidence_insufficient'
                rationale='附件8将该指标命名为Iso-alpha acids，单位IBU；属于家族测定指标，不是单一分子。'
                rationale+=' 与苦味目标命名对应，仍不证明因果。' if row.target=='bitternes' else ' 对该目标未提供特异直接依据，不能将系数解释为增浓效应。'
                source_url='attachment8:Sensory thresholds';excel_row=6
            else:
                hit=resolved[row.field]
                relation,terms,rationale=annotate_relation(row.target,row.field,hit.Flavor,hit.Origin,row.ridge_signed_coef,evidence_config)
                if not is_family:
                    old_terms='' if pd.isna(row.matched_reference_terms) else row.matched_reference_terms
                    assert relation==row.chemical_evidence and terms==old_terms
                table.loc[idx,'matched_reference_terms']=terms
                source_url='https://zenodo.org/records/10653704';excel_row=int(hit.excel_row)
            table.loc[idx,'chemical_evidence']=relation;table.loc[idx,'chemical_rationale']=rationale
            table.loc[idx,'reference_source_url']=source_url;table.loc[idx,'reference_excel_row']=excel_row
        return table
    strict=enrich_context(enriched,False);family=enrich_context(family,True)
    score_scope=scope.set_index('target_id')
    for table in [strict,family]:
        table['template_section']=table.target.map(score_scope.template_section)
        table['template_score_cell']=table.target.map(score_scope.template_score_formula_cell)
    slots=scope[['target_id','acceptance_scope','panel_target_available']].rename(columns={'target_id':'target'})
    slots=slots.merge(pd.DataFrame({'rank':[1,2,3]}),how='cross')
    slots=slots.merge(strict,on=['target','rank'],how='left',validate='one_to_one')
    slots['slot_status']=np.where(slots.field.notna(),'ranked_panel_association','unidentified_no_panel_response')
    slots['counts_as_identified_factor']=slots.field.notna()
    auxiliary=read(args.scorecard/'auxiliary_evidence.csv').set_index('target')
    task2_summary=meta(args.scorecard/'summary.json')
    assert task2_summary['source_sha256']['7.csv']==sha256(args.source/'7.csv')
    card=read(args.scorecard/'estimated_grades_all52.csv')
    extra=[];missing=[];target_rows=[]
    for row in scope.itertuples():
        t=row.target_id;selected=strict.loc[strict.target.eq(t)].sort_values('rank')
        available=bool(row.panel_target_available)
        common={'target':t,'acceptance_scope':row.acceptance_scope,'template_section':row.template_section,
            'template_score_cell':row.template_score_formula_cell,'panel_target_available':available,
            'expected_top3_slots':3,'identified_statistical_factors':len(selected),'unfilled_top3_slots':3-len(selected),
            'status':'ranked_panel_associations' if available else 'unidentified_no_panel_response',
            'top3_fields':';'.join(selected.field),'all_three_direct_mechanisms_verified':False}
        if available:
            common.update(minimum_refit_frequency=float(selected.bootstrap_frequency.min()),
                median_refit_frequency=float(selected.bootstrap_frequency.median()),
                n_with_reference_support=int(selected.chemical_evidence.ne('target_specific_evidence_insufficient').sum()),
                n_below_half_refit_frequency=int(selected.bootstrap_frequency.lt(.5).sum()),
                target_max_statistic_p=float(target_diagnostics.loc[t,'within_style_max_statistic_p']),
                target_BY_q=float(target_diagnostics.loc[t,'across_50_targets_BY_q']))
        else:
            assert t in ['A_acetaldehyde','F_acetaldehyde'] and card[t].eq(0).all()
            hit=resolved['acetaldehyde'];aux=auxiliary.loc[t]
            extra.append({'target':t,'named_chemical_marker':'acetaldehyde','chemical_name':'Acetaldehyde',
                'marker_rank':None,'rank_identified_from_panel_data':False,'expert_panel_labels_available':False,
                'reference_sheet':'Supplementary Table S1','reference_excel_row':int(hit.excel_row),
                'reference_source_url':'https://zenodo.org/records/10653704','reference_file_sha256':sha256(args.reference),
                'reference_descriptor':str(hit.Flavor).strip(),'template_descriptor':row.template_label,
                'threshold_reference_mg_L':task2_summary['reference_threshold_mg_L'],
                'threshold_proxy_distinct_values':int(card[t].nunique()),'threshold_proxy_treated_as_response':False,
                'forum_field':aux.forum_field,'forum_nonmissing':int(aux.nonmissing),'forum_nonmodal':int(aux.nonmodal_value_count),
                'forum_spearman_with_chemical_coordinate':aux.spearman_with_acetaldehyde_coordinate,
                'forum_treated_as_expert_label':False,'bootstrap_frequency':None,'target_BY_q':None,
                'causal_effect_established':False,
                'reason':'Named marker and reference descriptors support identity, not three ranked panel effects. Constant constructed scores have no usable response variation; sparse forum descriptors are a different outcome.'})
            missing.append({'target':t,'unfilled_slots':3,'required_evidence':'nonconstant expert-panel response for this modality, or independent intervention evidence sufficient for a separately justified mechanism ranking',
                'current_delivery':'named acetaldehyde marker and reference/auxiliary evidence only; no fabricated Top-3'})
        # The target table spans all 52 items, including explicitly unfilled ranks.
        target_rows.append(common)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    tables={'top3_panel_all50.csv':strict,'top3_panel_primary45.csv':strict.loc[strict.primary],
        'top3_all52_slots.csv':slots,'top3_primary47_slots.csv':slots.loc[slots.acceptance_scope.eq('main_content')],
        'target_summary52.csv':target_rows,'family_scope_top3_supplement.csv':family,
        'target_scope_comparison.csv':target_diagnostics.reset_index(),'extra_target_evidence.csv':extra,'unfilled_rank_requirements.csv':missing,
        'compound_name_evidence.csv':ledger,'network_nodes.csv':node_rows,'network_edges_0_6.csv':edges,
        'network_sensitivity.csv':sensitivity,'correlation_method_evidence.csv':correlation_evidence,
        'named_marker_diagnostics.csv':read(args.reliability/'named_marker_diagnostics.csv')}
    for name,data in tables.items():save_csv(out,name,data)
    nx.write_graphml(graph,out/'network_0_6.graphml')
    write_json(out/'summary.json',{'status':'completed_with_explicit_identification_gaps','config':config,'input_sha256':inputs,
        'source_sha256':{n:sha256(args.source/n) for n in ['1.csv','2.csv','3.xlsx','4.csv','7.csv','8.xlsx']},
        'scope_items':52,'main_content_items':47,'panel_ranked_targets':50,'identified_main_rows':135,'identified_panel_rows':150,
        'all_scope_slots':156,'unfilled_slots':6,'unidentified_targets':['A_acetaldehyde','F_acetaldehyde'],
        'network_nodes':len(graph),'network_edges':graph.number_of_edges(),'reference_resolved_individual_names':len(ledger),
        'below_half_refit_frequency':int(strict.bootstrap_frequency.lt(.5).sum()),
        'chemical_evidence_counts':strict.chemical_evidence.value_counts().to_dict(),
        'strict_target_BY_passes':int(target_diagnostics.passes_BY_0_05.sum()),'family_target_BY_passes':int(target_diagnostics.family_passes_BY_0_05.sum()),
        'new_permutations':0,'new_bootstrap_refits':0,'new_models_selected':0,'causal_effects_established':False,
        'interpretation':'150 identified descriptive factors plus six explicitly unfilled rank slots. Two named-marker evidence rows are not six discovered factors; BY families and chemical support remain separate.'})
    print('Q2 task4: 150 ranked associations, 156 scope slots with 6 unfilled; 231 nodes and 266 edges.')


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    for name,default in {'audit':'outputs/p0_audit_20261023_final','delivery':'outputs/q2_delivery_20261024',
        'reliability':'outputs/q2_reliability_20261024','exploratory':'outputs/q2_exploratory_20261023_v1',
        'reference':'data/reference/zenodo10653704/Supplemental Files and Figure source files.xlsx',
        'scorecard':'outputs/q1_task2_20261025','output':'outputs/q2_task4_20261025'}.items():
        p.add_argument('--'+name,type=Path,default=Path(default))
    return p


if __name__=='__main__':run(parser().parse_args())
