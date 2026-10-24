"""Read-only source audit for the contest acceptance scope; never trains models."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from beer_audit.audit import save_csv, sha256, write_json


def run(source, statement, output):
    project = Path(__file__).resolve().parents[1]
    audit = project / 'outputs/p0_audit_20261023_final'
    inventory = pd.read_csv(audit / 'data_inventory.csv').set_index('file')
    names = ['1.csv', '2.csv', '3.xlsx', '4.csv', '7.csv', '8.xlsx']
    hashes = {name: sha256(source / name) for name in names}
    for name in names:
        assert hashes[name] == inventory.loc[name, 'sha256'], 'Source changed: ' + name
    x = pd.read_csv(source / '1.csv'); y = pd.read_csv(source / '4.csv'); reviews = pd.read_csv(source / '7.csv')
    assert set(x.beer) == set(y.beer) == set(reviews.beer) and not x.beer.duplicated().any()
    for frame in [y, reviews]:
        aligned = frame.set_index('beer').loc[x.beer]
        np.testing.assert_array_equal(aligned.tasting_category_fine, x.tasting_category_fine)
    mapping = pd.read_csv(audit / 'sensory_mapping.csv').set_index('R_name')
    targets = set(y.columns) - {'beer', 'beer_id', 'tasting_category_fine'}
    assert targets == set(mapping.index)
    workbook = load_workbook(source / '3.xlsx', read_only=True, data_only=False)
    cards = [workbook[f'beer_{i}'] for i in range(1, 13)]
    summary = workbook['summary']
    summary_headers = {str(c.value): c.coordinate for c in summary[1] if c.value is not None}
    aliases = {'A_esters_fruity': 'A_ester_fruity', 'F_esters_fruity': 'F_ester_fruity', 'X4vg': '4vg'}
    records = []

    def add(target, label_cell, score_cell, section, parent='', auxiliary=''):
        label = cards[0][label_cell].value
        assert isinstance(label, str) and label
        in_panel = target in targets
        primary = not in_panel or mapping.loc[target, 'scope'] == 'aroma_flavour_primary_candidate'
        header = aliases.get(target, target)
        if in_panel:
            assert header in summary_headers
        else:
            assert header not in summary_headers
        for card in cards:
            assert card[label_cell].value == label
            formula = card[score_cell].value
            assert isinstance(formula, str) and formula.startswith('=IF(ISBLANK(')
            if parent:
                assert parent in formula
                input_cells = [('C' if score_cell.startswith('Q') else 'S') + str(card[score_cell].row)]
            else:
                columns = ['C', 'E', 'G', 'I', 'K', 'M', 'O'] if score_cell.startswith('Q') else ['S', 'U', 'W', 'Y', 'AA', 'AC', 'AE']
                input_cells = [col + str(card[score_cell].row) for col in columns]
            assert all(card[c].value is None for c in input_cells), 'Unexpected filled scorecard inputs'
        records.append({'target_id': target, 'template_sheet': 'beer_1', 'template_label_cell': label_cell,
            'template_label': label, 'template_score_formula_cell': score_cell, 'template_section': section,
            'template_rule': 'checkbox_zero_or_parent' if parent else 'independent_0_6_rating',
            'parent_formula_cell': parent, 'acceptance_scope': 'main_content' if primary else 'supplementary',
            'existing_primary45': bool(in_panel and primary), 'panel_target_4csv': target if in_panel else '',
            'panel_target_available': in_panel, 'summary_header': header if in_panel else '',
            'summary_header_cell': summary_headers.get(header, ''),
            'current_model_covered': in_panel, 'auxiliary_forum_7csv': auxiliary,
            'auxiliary_nonmissing': int(reviews[auxiliary].notna().sum()) if auxiliary else 0,
            'label_status': 'standardized_panel_not_raw_0_6' if in_panel else 'template_only_no_panel_target',
            'note': 'Forum descriptor is a different label source; not a panel-score replacement' if auxiliary else ''})

    groups = [('malt', 6, 7, [10, 12, 14, 16], ['grain', 'bread', 'cara', 'burn']),
              ('hops', 19, 20, [23, 25, 27, 29], ['citrus', 'tropical', 'noble', 'woody']),
              ('esters', 32, 33, [36, 38, 40, 42], ['ethac', 'isoaa', 'flower', 'fruity'])]
    for prefix, label_col, descriptor_col, formula_col, section in [('A', 'B', 'E', 'Q', 'Aroma'), ('F', 'R', 'U', 'AG', 'Flavour')]:
        for group, label_row, score_row, rows, suffixes in groups:
            add(f'{prefix}_{group}_all', f'{label_col}{label_row}', f'{formula_col}{score_row}', section)
            for row, suffix in zip(rows, suffixes):
                add(f'{prefix}_{group}_{suffix}', f'{descriptor_col}{row}', f'{formula_col}{row}', section, f'{formula_col}{score_row}')
        add(f'{prefix}_acetaldehyde', f'{label_col}44', f'{formula_col}45', section,
            auxiliary='aroma_acetaldehyde' if prefix == 'A' else 'taste_acetaldehyde')
    for target, cell, formula, section in [
        ('body', 'B49', 'Q50', 'Basic sensory'), ('alcohol', 'R49', 'AG50', 'Basic sensory'),
        ('bitternes', 'B53', 'Q54', 'Basic sensory'), ('sweetness', 'R53', 'AG54', 'Basic sensory'),
        ('co2', 'B57', 'Q58', 'Basic sensory'), ('acidity', 'R57', 'AG58', 'Basic sensory'),
        ('aftertaste', 'B61', 'Q62', 'Basic sensory'), ('dms', 'B67', 'Q68', 'Faults'),
        ('t2n', 'R67', 'AG68', 'Faults'), ('metallic', 'B71', 'Q72', 'Faults'),
        ('diacetyl', 'R71', 'AG72', 'Faults'), ('X4vg', 'B75', 'Q76', 'Faults'),
        ('stale_hops', 'R75', 'AG76', 'Faults'), ('orange', 'B81', 'Q82', 'Spiced beers'),
        ('lactic', 'R81', 'AG82', 'Lambic/Geuze'), ('coriander', 'B85', 'Q86', 'Spiced beers'),
        ('acetic', 'R85', 'AG86', 'Lambic/Geuze'), ('clove', 'B89', 'Q90', 'Spiced beers'),
        ('barnyard', 'R89', 'AG90', 'Lambic/Geuze'), ('overall', 'B93', 'Q94', 'Overall')]:
        add(target, cell, formula, section)
    scope = pd.DataFrame(records)
    assert len(scope) == 52 and not scope.target_id.duplicated().any()
    assert set(scope.loc[scope.panel_target_available, 'target_id']) == targets
    assert len(scope.loc[scope.acceptance_scope.eq('main_content')]) == 47
    assert set(scope.loc[~scope.panel_target_available, 'target_id']) == {'A_acetaldehyde', 'F_acetaldehyde'}
    assert set(scope.loc[scope.existing_primary45, 'target_id']) == set(mapping.loc[mapping.scope.eq('aroma_flavour_primary_candidate')].index)
    assert set(summary_headers) - {'Beer', 'comments'} == {aliases.get(t, t) for t in targets}
    formulas = {c.coordinate for row in cards[0] for c in row if isinstance(c.value, str)
                and c.value.startswith('=IF(ISBLANK(') and c.column in [17, 33]}
    named_formulas = set(scope.template_score_formula_cell)
    assert named_formulas <= formulas and formulas - named_formulas == {'AG62', 'AG94'}
    model_manifest = json.loads((project / 'outputs/final_20261024/q1/run_manifest.json').read_text(encoding='utf-8'))
    assert set(model_manifest['targets']) == targets
    thresholds = pd.read_excel(source / '8.xlsx', sheet_name='Sensory thresholds')
    thresholds['field_name_R'] = thresholds.field_name_R.str.strip()
    thresholds = thresholds.set_index('field_name_R')
    defect_records = []
    for name, feature, target, transform, policy in [
        ('DMS', 'DMS', 'dms', 'positive_pairs_log10; zero_reference_exception', 'monotone_coordinate_nonincrease_or_valid_threshold_excess'),
        ('Diacetyl', 'X2.3.butanedione', 'diacetyl', 'mostly_identity_with_three_conflicts', 'nonincrease; reference_zero_cap_for_beer119_28_is_conditional'),
        ('4VG', 'X4VG', 'X4vg', 'positive_pairs_log10; 168_zero_references', 'monotone_coordinate_nonincrease_or_valid_threshold_excess'),
        ('trans-2-nonenal', 'X.E..2.nonenal', 't2n', 'mostly_identity_with_two_conflicts', 'nonincrease_without_inventing_threshold'),
        ('Stale hops', '', 'stale_hops', 'standardized_sensory_target_not_single_compound', 'explicit_sensory_proxy_nonincrease_with_validation_and_limitations')]:
        row = thresholds.loc[feature] if feature else None
        defect_records.append({'defect': name, 'chemical_input_1csv': feature, 'sensory_target_4csv': target,
            'reference_unit': row['unit'] if feature else 'standardized_panel_score',
            'reference_threshold_mean': row.threshold_mean if feature else np.nan,
            'reference_threshold_median': row.threshold_median if feature else np.nan,
            'threshold_status': 'reference_only_matrix_and_scale_not_automatically_certified' if feature and pd.notna(row.threshold_mean) else 'not_available_do_not_invent',
            'coordinate_evidence': transform, 'acceptance_policy': policy,
            'template_score_cell': scope.set_index('target_id').loc[target, 'template_score_formula_cell']})
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output, 'scorecard_scope.csv', scope)
    save_csv(output, 'defect_scope.csv', defect_records)
    result = {'status': 'passed', 'meaning': 'Scope audit passed; not model or three-question completion',
        'source_sha256': hashes, 'statement_sha256': sha256(statement), 'statement_requirement_page': 2,
        'script_sha256': sha256(Path(__file__)), 'samples': len(x), 'chemical_coordinates': len(x.columns)-3,
        'styles': int(x.tasting_category_fine.nunique()), 'scorecards_checked': len(cards), 'template_items': 52,
        'template_main_content_items': 47, 'supplementary_items': 5, 'existing_panel_targets': 50,
        'existing_primary_targets': 45, 'template_only_missing_panel_targets': ['A_acetaldehyde', 'F_acetaldehyde'],
        'score_formula_cells_in_example_card': len(formulas), 'excluded_unnamed_or_comment_formulas': sorted(formulas - named_formulas),
        'checkbox_parent_relations': int(scope.parent_formula_cell.ne('').sum()), 'original_inputs_filled': 0,
        'existing_models_retrained': False, 'checks': ['six attachment hashes match original audit',
            'sample identifiers and styles align', 'all 52 label/formula locations agree across 12 cards',
            'all 50 panel targets and three summary aliases mapped', '47 content items and five supplements distinguished',
            '54 score formulas account for 52 named items and two unused/comment cells', 'current model target metadata matches the 50 panel fields',
            'two template-only items lack panel labels', 'all referenced scorecard inputs remain blank',
            'five named defects mapped without invented physical thresholds']}
    write_json(output / 'verification.json', result)
    workbook.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--statement', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/task1_acceptance_20261025'))
    args = parser.parse_args()
    statement = args.statement or args.source.parent / '2026年MathorCup大数据挑战赛-赛道A初赛.pdf'
    run(args.source, statement, args.output)
