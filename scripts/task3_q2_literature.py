"""Assemble curated literature annotations without changing frozen Q2 statistics."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/task3_q2_literature_20261025.json'
BASE = ROOT / 'outputs/q2_task4_20261025'
INPUTS = ['top3_panel_all50.csv', 'top3_all52_slots.csv',
          'named_marker_diagnostics.csv', 'family_scope_top3_supplement.csv']


def read(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def digest(value):
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def table_digest(table):
    # Hash parsed cells, preserving row/column order but allowing Git newline conversion.
    return digest({'columns': list(table), 'records': table.to_dict('records')})


def annotations(config, frame):
    lookup = {(r['target'], str(r['rank']), r['field']): r
              for r in config['pair_assessments']}
    sources = {r['source_id']: r for r in config['sources']}
    result = []
    for row in frame.itertuples():
        item = lookup.get((row.target, row.rank, row.field))
        ids = item['source_ids'] if item else []
        if not row.field:
            status, typ, note = 'unidentified_no_panel_label', 'not_applicable', '无面板响应，排名继续留空'
        elif item:
            status, typ, note = 'reviewed', item['evidence_type'], item['assessment']
        else:
            status, typ, note = 'not_reviewed_this_task', 'not_reviewed_this_task', '不在本任务16个重点目标内'
        result.append({
            'lit_review_status': status, 'lit_evidence_type': typ,
            'lit_source_ids': ';'.join(ids),
            'lit_independent_source_ids': ';'.join(i for i in ids if sources[i]['independent_of_dataset']),
            'lit_assessment': note,
            'lit_coefficient_interpretation': ('负条件系数不证明抑制作用' if row.ridge_signed_coef
                and float(row.ridge_signed_coef) < 0 else '条件关联不证明增加浓度的感官效应'),
            'lit_dataset_causal_effect_established': 'False',
        })
    return pd.concat([frame.reset_index(drop=True), pd.DataFrame(result)], axis=1)


def assemble(config, inputs):
    sources = {s['source_id']: s for s in config['sources']}
    assert len(sources) == len(config['sources']) == 16
    assert sources['DATA01']['independent_of_dataset'] is False
    assert all(s['independent_of_dataset'] for i, s in sources.items() if i != 'DATA01')
    focus_targets = config['focus_targets']
    assert len(set(focus_targets)) == len(focus_targets) == 16
    assert config['causal_effect_established'] is False
    pairs = config['pair_assessments']
    assert len(pairs) == 48
    actual = {(r.target, r.rank, r.field) for r in inputs[INPUTS[0]].itertuples()
              if r.target in focus_targets}
    expected = {(r['target'], str(r['rank']), r['field']) for r in pairs}
    assert len(expected) == 48 and actual == expected
    for r in pairs + config['marker_controls'] + config['target_contexts']:
        assert len(r['source_ids']) == len(set(r['source_ids']))
        assert set(r['source_ids']) <= sources.keys()
        assert r['assessment'].strip()
    full = annotations(config, inputs[INPUTS[0]])
    slots = annotations(config, inputs[INPUTS[1]])
    focused = full.loc[full.lit_review_status.eq('reviewed')].reset_index(drop=True)
    assert len(full) == 150 and len(focused) == 48
    assert len(slots) == 156 and slots.field.eq('').sum() == 6
    assert focused.groupby('target').size().eq(3).all()
    # Preserve every original field, including existing descriptor annotations and P/BY values.
    pd.testing.assert_frame_equal(full[list(inputs[INPUTS[0]])], inputs[INPUTS[0]])
    pd.testing.assert_frame_equal(slots[list(inputs[INPUTS[1]])], inputs[INPUTS[1]])
    summary = []
    for target, group in focused.groupby('target', sort=False):
        summary.append(dict(target=target, reviewed_pairs=len(group),
            evidence_types=';'.join(group.lit_evidence_type),
            target_BY_q=group.target_BY_q.iloc[0],
            target_passes_BY_0_05=group.target_passes_BY_0_05.iloc[0],
            dataset_causal_effect_established='False'))
    controls = []
    for item in config['marker_controls']:
        item = item.copy()
        target, field = item['target'], item['field']
        selected = full.loc[full.target.eq(target) & full.field.eq(field)]
        marker = inputs[INPUTS[2]]
        marker = marker.loc[marker.target.eq(target) & marker.field.eq(field)]
        family = inputs[INPUTS[3]]
        family = family.loc[family.target.eq(target) & family.field.eq(field)]
        item['strict_top3_selected'] = str(not selected.empty)
        item['strict_rank'] = (selected['rank'].iloc[0] if not selected.empty else
            marker.absolute_coefficient_rank_in_strict_scope.iloc[0] if not marker.empty else '')
        item['family_top3_rank'] = family['rank'].iloc[0] if not family.empty else ''
        item['rank_provenance'] = ('family_supplement_only' if item['ranking_scope'] == 'family_220'
            else 'strict_top3' if not selected.empty else 'existing_named_marker_diagnostic'
            if not marker.empty else 'not_in_strict_top3_full_rank_not_recorded')
        item['source_ids'] = ';'.join(item['source_ids'])
        controls.append(item)
    contexts = [{**r, 'source_ids': ';'.join(r['source_ids'])} for r in config['target_contexts']]
    return {'top3_all50_literature.csv': full, 'top3_all52_slots_literature.csv': slots,
            'focus48_literature.csv': focused, 'focus_target_summary.csv': pd.DataFrame(summary),
            'source_catalog.csv': pd.DataFrame(config['sources']),
            'named_marker_controls.csv': pd.DataFrame(controls),
            'target_contexts.csv': pd.DataFrame(contexts)}


def run(output, verify_only):
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    inputs = {name: read(BASE / name) for name in INPUTS}
    fingerprints = {name: table_digest(frame) for name, frame in inputs.items()}
    assert fingerprints == config['input_semantic_sha256'], 'Frozen Q2 input cells changed'
    tables = assemble(config, inputs)
    if not verify_only:
        output.mkdir(parents=True, exist_ok=False)
        for name, frame in tables.items():
            frame.to_csv(output / name, index=False, encoding='utf-8-sig')
    for name, expected in tables.items():
        pd.testing.assert_frame_equal(read(output / name), expected.astype(str), obj=name)
    focused = tables['focus48_literature.csv']
    verification = dict(status='passed', baseline_commit=config['baseline_commit'],
        config_content_sha256=digest(config), input_semantic_sha256=fingerprints,
        unchanged_original_columns=46, unchanged_panel_rows=150, unchanged_all52_slots=156,
        preserved_unidentified_slots=6, focus_targets=16, reviewed_pairs=48,
        not_reviewed_panel_pairs=102, sources=16, independent_sources=15,
        marker_controls=9, target_contexts=1,
        evidence_type_counts={k: int(v) for k, v in focused.lit_evidence_type.value_counts().items()},
        dataset_causal_effect_established=False,
        checks=['all_original_cells_preserved', 'exact_focus_pair_coverage',
                'all_source_references_resolve', 'same_dataset_not_counted_independent',
                'markers_separate_from_ranking', 'saved_tables_match_curated_ledger'])
    path = output / 'verification.json'
    if verify_only:
        assert json.loads(path.read_text(encoding='utf-8')) == verification
    else:
        path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/task3_q2_literature_20261025')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    run(args.output, args.verify_only)
