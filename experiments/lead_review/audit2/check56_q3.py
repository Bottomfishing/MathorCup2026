"""Independent audit items 5+6+8: Q3 migration closure and gate-figure provenance.

  * structural completeness of the migration closure tables (66 spectra, 33x227,
    3432 sensory rows, 33 evidence cases)
  * claim_level / style-diagnostic / defect-status honesty
  * independent sensory replay: rebuild the 66 spectra in the official 224-feature
    coordinates, run q1.joblib through inference.run and compare the 52-target
    portrait with the delivered s2c_sensory_before_after_52.csv
  * gate figure method provenance (v3 hybrid_support_projected, not v1 copula_quantile)

Run from repo root:
    .venv\\Scripts\\python.exe experiments/lead_review/audit2/check56_q3.py
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / 'experiments' / 'lead_review' / 'audit2'
MIG = ROOT / 'outputs' / 'q3_migration_20261028'
Q3V3 = ROOT / 'outputs' / 'q3_copula_v3_20261028'
Q3V1 = ROOT / 'outputs' / 'q3_copula_20261028'
SOURCE = Path(r'C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题'
              r'\2026年MathorCup大数据挑战赛-赛道A初赛\附件')
MODEL = ROOT / 'outputs' / 'current_models' / 'q1.joblib'
FIGURES = ROOT / 'deliverables' / 'current_20261028' / 'figures'


def gate_methods(folder):
    path = folder / 'generation_validation.csv'
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return sorted(str(name) for name in frame['method'].dropna().unique())


def main():
    report = {'item': 'q3_migration_closure_and_gate_figure'}

    spectra = pd.read_csv(MIG / 's2c_migration_spectra_224_long.csv')
    after_input = pd.read_csv(MIG / 's2c_migration_after_input_33.csv')
    sensory = pd.read_csv(MIG / 's2c_sensory_before_after_52.csv')
    evidence = pd.read_csv(MIG / 's2c_evidence_by_case.csv')
    migration_verification = json.loads((MIG / 'verification.json').read_text(encoding='utf-8'))

    feature_cols = [c for c in after_input.columns if c not in ('generated_id', 'style', 'method')]
    report['structure'] = {
        'spectra_rows': int(len(spectra)),
        'spectra_cases': int(spectra['case_id'].nunique()),
        'spectra_roles': {str(k): int(v) for k, v in spectra['role'].value_counts().items()},
        'spectra_feature_columns': len(feature_cols),
        'after_input_rows': int(len(after_input)),
        'after_input_feature_columns': len(feature_cols),
        'sensory_rows': int(len(sensory)),
        'sensory_cases': int(sensory['case_id'].nunique()),
        'sensory_targets': int(sensory['target_52'].nunique()),
        'sensory_roles': {str(k): int(v) for k, v in sensory['role'].value_counts().items()},
        'expected_sensory_rows': int(sensory['case_id'].nunique() * sensory['role'].nunique()
                                     * sensory['target_52'].nunique()),
        'sensory_rows_complete': bool(len(sensory) == sensory['case_id'].nunique()
                                      * sensory['role'].nunique() * sensory['target_52'].nunique()),
        'evidence_rows': int(len(evidence)),
        'evidence_cases': int(evidence['case_id'].nunique()),
        'evidence_columns': len(evidence.columns),
        'verification_status': migration_verification.get('status'),
        'verification_status_passed': migration_verification.get('status_passed'),
        'verification_checks_all_pass': all(bool(v) for v in migration_verification.get('checks', {}).values()) if isinstance(migration_verification.get('checks'), dict) else None,
    }

    # spectra long vs after_input values (the 'after' rows must be the after-input spectra).
    wide_after = after_input.set_index('generated_id')[feature_cols]
    spectra_after = spectra[spectra['role'] == 'after'].set_index('case_id')[feature_cols]
    common = [c for c in spectra_after.index if c in wide_after.index]
    diff_after = float(np.nanmax(np.abs(spectra_after.loc[common].to_numpy(float)
                                       - wide_after.loc[common].to_numpy(float))))
    report['after_consistency'] = {'cases_compared': len(common), 'max_abs_diff': diff_after}

    # 'before' rows must be the source beers' attachment chemistry.
    attachment = pd.read_csv(SOURCE / '1.csv')
    attachment['beer'] = attachment['beer'].astype(str)
    attachment = attachment.set_index('beer')
    before = spectra[spectra['role'] == 'before'].drop_duplicates('source_beer').set_index('source_beer')
    before.index = before.index.astype(str)
    before_cases = [b for b in before.index if b in attachment.index]
    before_diff = float(np.nanmax(np.abs(before.loc[before_cases, feature_cols].to_numpy(float)
                                         - attachment.loc[before_cases, feature_cols].to_numpy(float))))
    report['before_consistency'] = {'cases_compared': len(before_cases), 'max_abs_diff': before_diff}

    # claim-level / style / defect honesty.
    report['claim_level_counts'] = {str(k): int(v) for k, v in evidence['claim_level'].value_counts().items()}
    evidence['style_diagnostic_cases_int'] = pd.to_numeric(evidence['style_diagnostic_cases'], errors='coerce')
    evidence['style_diagnostic_total_int'] = pd.to_numeric(evidence['style_diagnostic_total'], errors='coerce')
    # the diagnostic is a property of the target style and is repeated on every case row:
    # "k of the 11 source beers migrate to this style's centroid".
    per_style = evidence.sort_values('case_id').drop_duplicates('target_style')
    report['style_diagnostics'] = {
        str(style): {'cases': int(per_style.loc[per_style['target_style'] == style,
                                                'style_diagnostic_cases_int'].iloc[0]),
                     'total': int(per_style.loc[per_style['target_style'] == style,
                                                'style_diagnostic_total_int'].iloc[0])}
        for style in sorted(per_style['target_style'].unique())}
    report['style_diagnostics_match_claim'] = (
        report['style_diagnostics'].get('Tripel') == {'cases': 0, 'total': 11}
        and report['style_diagnostics'].get('Kriek') == {'cases': 11, 'total': 11}
        and report['style_diagnostics'].get('Lambic') == {'cases': 2, 'total': 11})
    defect_columns = [c for c in evidence.columns if c.startswith('defect_status_')]
    report['defect_status_counts'] = {c: {str(k): int(v) for k, v in evidence[c].value_counts().items()}
                                      for c in defect_columns}
    report['defects_worse_total'] = int(pd.to_numeric(evidence['defects_worse_count'], errors='coerce').fillna(0).sum())
    report['t2n_stale_unknown'] = bool(
        evidence['defect_status_t2n'].eq('unknown').all()
        and evidence['defect_status_Stale_hops'].eq('unknown').all())
    claimed_claim_levels = migration_verification.get('claim_level_counts', {})
    report['verification_claims_match_table'] = bool(
        {str(k): int(v) for k, v in claimed_claim_levels.items()}
        == {str(k): int(v) for k, v in report['claim_level_counts'].items()})

    # Independent sensory replay through the official inference entry point.
    from beer_models.inference import run as inference_run

    profiles = spectra[['case_id', 'role', *feature_cols]].copy()
    profiles['beer'] = profiles['case_id'].astype(str) + '::' + profiles['role'].astype(str)
    profiles = profiles.drop(columns=['case_id', 'role'])[['beer', *feature_cols]]
    replay_dir = AUDIT / 'q3_sensory_replay'
    if replay_dir.exists():
        import shutil
        shutil.rmtree(replay_dir)
    replay_input = AUDIT / 'q3_sensory_replay_input.csv'
    profiles.to_csv(replay_input, index=False, encoding='utf-8-sig')
    inference_run(MODEL, replay_input, replay_dir)
    replay = pd.read_csv(replay_dir / 'scores_all52_long.csv', dtype={'beer': str})
    replay['case_id'] = replay['beer'].str.split('::').str[0]
    replay['role'] = replay['beer'].str.split('::').str[1]
    delivered = sensory.rename(columns={'grade_raw': 'grade_delivered'})
    merged = delivered.merge(
        replay[['case_id', 'role', 'target', 'estimated_grade_0_6']].rename(
            columns={'target': 'target_52', 'estimated_grade_0_6': 'grade_replayed'}),
        on=['case_id', 'role', 'target_52'], how='outer', validate='one_to_one')
    matched = merged['grade_delivered'].notna() & merged['grade_replayed'].notna()
    mismatched = int((merged.loc[matched, 'grade_delivered'] != merged.loc[matched, 'grade_replayed']).sum())
    report['sensory_replay'] = {
        'rows': int(len(merged)),
        'matched_rows': int(matched.sum()),
        'mismatched_cells': mismatched,
        'max_abs_diff': float((merged.loc[matched, 'grade_delivered'] - merged.loc[matched, 'grade_replayed']).abs().max()),
        'replay_dir': str(replay_dir.relative_to(ROOT)),
    }

    # Gate-figure provenance.
    v3_methods = gate_methods(Q3V3)
    v1_methods = gate_methods(Q3V1)
    figure_svg = FIGURES / 'Q3_生成门槛.svg'
    svg_text = figure_svg.read_text(encoding='utf-8', errors='replace') if figure_svg.exists() else ''
    report['gate_figure'] = {
        'v3_generation_validation_methods': v3_methods,
        'v1_generation_validation_methods': v1_methods,
        'figure_exists': figure_svg.exists(),
        'figure_mentions_v3_hybrid': 'hybrid_support_projected' in svg_text,
        'figure_mentions_v1_copula_quantile': 'copula_quantile' in svg_text,
        'figure_method_tokens': sorted(set(re.findall(r'hybrid_support_projected|donor_residual_projected|interpolate|copula_quantile', svg_text))),
    }
    report['verdict'] = 'fixed' if (
        report['structure']['sensory_rows_complete']
        and report['structure']['evidence_rows'] == 33
        and report['after_consistency']['max_abs_diff'] < 1e-9
        and report['before_consistency']['max_abs_diff'] < 1e-9
        and report['style_diagnostics_match_claim']
        and report['defects_worse_total'] == 0
        and report['t2n_stale_unknown']
        and report['verification_claims_match_table']
        and mismatched == 0
        and report['gate_figure']['figure_mentions_v3_hybrid']
        and not report['gate_figure']['figure_mentions_v1_copula_quantile']
    ) else 'partial'
    (AUDIT / 'check56_q3.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
