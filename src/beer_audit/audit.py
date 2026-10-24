"""Reproducible read-only audit. AI-assisted: OpenAI Codex (GPT-6), 2026-10-23."""
import argparse
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from .checks import (IDS, align_samples, compare_scale, numeric_frame, square_matrix_checks,
                     pairwise_spearman, repair_utf8_mojibake)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_csv(folder, name, obj):
    pd.DataFrame(obj).to_csv(folder / name, index=False, encoding='utf-8-sig')


def read_csv(path):
    # Keep barcodes exact: numeric conversion is applied to measurements only.
    return pd.read_csv(path, dtype=str, encoding='utf-8-sig')


def card_audit(path):
    formulas = openpyxl.load_workbook(path, read_only=False, data_only=False)
    cached = openpyxl.load_workbook(path, read_only=False, data_only=True)
    records = []
    for sheet in formulas:
        if not sheet.title.startswith('beer_'):
            continue
        input_refs = set()
        for row in sheet:
            for cell in row:
                if cell.data_type == 'f':
                    input_refs.update(re.findall(r'ISBLANK\(\$?([A-Z]+)\$?(\d+)\)', cell.value))
        coords = sorted({a + b for a, b in input_refs})
        filled = [p for p in coords if sheet[p].value is not None]
        records.append({'sheet': sheet.title, 'referenced_input_cells': len(coords),
                        'filled_input_cells': len(filled), 'filled_coordinates': ';'.join(filled)})
    summary = formulas['summary']
    score_cells = [c for row in summary for c in row if c.data_type == 'f']
    zeros = sum(cached['summary'][c.coordinate].value == 0 for c in score_cells)
    names = [c.value for c in summary[1] if c.value is not None][1:]
    result = {'cards': records, 'summary_formula_cells': len(score_cells),
              'summary_cached_zero_cells': zeros, 'summary_names': names,
              'interpretation': 'template_like_inputs_not_observed_scores',
              'original_scale_identified': False}
    formulas.close()
    cached.close()
    return result


def profile_numeric(file, frame):
    rows = []
    for col in frame:
        s = frame[col]
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        rows.append({'file': file, 'field': col, 'n_observed': int(s.count()),
                     'n_missing': int(s.isna().sum()), 'n_unique': int(s.nunique()),
                     'all_missing': bool(s.isna().all()), 'constant_observed': bool(s.nunique() == 1),
                     'minimum': s.min(), 'maximum': s.max(), 'mean': s.mean(), 'std_ddof1': s.std(),
                     'iqr_outlier_count': int(((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).sum()),
                     'outlier_action': 'flag_only_do_not_delete'})
    return rows


def run(source, output):
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output == source or source in output.parents:
        raise ValueError('Output must not be inside original source folder')
    output.mkdir(parents=True, exist_ok=False)
    inventory = [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                 for p in sorted(source.iterdir()) if p.is_file()]
    save_csv(output, 'data_inventory.csv', inventory)
    raw = {name: read_csv(source / name) for name in ['1.csv', '4.csv', '7.csv']}
    for name in ['4.csv', '7.csv']:
        raw[name] = align_samples(raw['1.csv'], raw[name])
    # The reference itself also must pass uniqueness validation.
    align_samples(raw['1.csv'], raw['1.csv'])
    nums = {name: numeric_frame(frame.drop(columns=IDS)) for name, frame in raw.items()}
    save_csv(output, 'numeric_profile.csv', [r for k,v in nums.items() for r in profile_numeric(k,v)])
    sample_map = raw['1.csv'][IDS].copy()
    for name in ['4.csv','7.csv']:
        sample_map['beer_id_' + name] = raw[name]['beer_id']
        sample_map['barcode_exact_match_' + name] = raw['1.csv']['beer_id'].eq(raw[name]['beer_id'])
    sample_map['join_basis'] = 'unique_beer_key_and_matching_style'
    save_csv(output, 'sample_mapping.csv', sample_map)

    workbook = pd.ExcelFile(source / '8.xlsx', engine='openpyxl')
    thresholds = pd.read_excel(workbook, sheet_name='Sensory thresholds')
    sensory = pd.read_excel(workbook, sheet_name='Sensory names')
    calibrated = pd.read_excel(workbook, sheet_name='Calibrated chemical dataset', dtype=str)
    workbook.close()
    calibrated = align_samples(raw['1.csv'], calibrated)
    cal = numeric_frame(calibrated.drop(columns=IDS))
    if thresholds['field_name_R'].dropna().duplicated().any():
        raise ValueError('Duplicate chemical aliases in reference')
    features = pd.DataFrame({'field_name_R': nums['1.csv'].columns})
    features = features.merge(thresholds, on='field_name_R', how='left', validate='one_to_one', indicator=True)
    features['alias_status'] = features.pop('_merge').astype(str)
    features['transform_status'] = 'not_checked_no_calibrated_pair'
    scales, conflicts = [], []
    for col in cal:
        if col not in nums['1.csv']:
            continue
        a, b = nums['1.csv'][col], cal[col]
        result = compare_scale(a, b)
        scales.append({'field': col, **result})
        features.loc[features.field_name_R.eq(col), 'transform_status'] = result['status']
        features.loc[features.field_name_R.eq(col), 'candidate_transform'] = result['candidate']
        kind = result['candidate']
        if kind == 'undetermined':
            continue
        expected = b.copy()
        if kind in ['log10', 'ln']:
            expected[:] = np.nan
            positive = b > 0
            expected.loc[positive] = (np.log10 if kind == 'log10' else np.log)(b.loc[positive])
        valid = a.notna() & b.notna()
        matches = np.isclose(a, expected, atol=2e-6, rtol=1e-6, equal_nan=False)
        for i in a.index[valid & ~matches]:
            conflicts.append({'beer': raw['1.csv'].loc[i, 'beer'], 'field': col,
                              'source_value': a[i], 'reference_value': b[i], 'candidate': kind,
                              'reason': 'zero_reference_not_invertible' if b[i] == 0 and kind != 'identity'
                              else 'pair_disagrees', 'action': 'retain_both_no_automatic_replacement'})
    # Conservative role tags: physical parameters and computed totals are not individual chemicals.
    physical = {'color.EBC.','pH','bitterness','ethanol..v.v.','CO2.PSI.','protein.g.L.','betaglucan.mg.L.'}
    aggregate = {'kcalperc','acids_sum','esters_sum','aroma_hops_sum','sulfur_sum','GFS','unfermentables'}
    features['role'] = ['physical_or_bulk_parameter' if c in physical else
                        'aggregate_candidate_verify_definition' if c in aggregate else
                        'chemical_or_other_measurement_verify_definition' for c in features.field_name_R]
    features['unit_status'] = np.where(features.unit.notna(), 'reference_unit_not_automatically_source_unit', 'unknown')
    save_csv(output, 'feature_dictionary.csv', features)
    save_csv(output, 'transform_checks.csv', scales)
    save_csv(output, 'calibrated_conflicts.csv', conflicts)

    card = card_audit(source / '3.xlsx')
    write_json(output / 'scorecard_audit.json', card)
    panel = sensory.loc[sensory.dataset.eq('Trained panel')].copy()
    if panel.R_name.duplicated().any():
        raise ValueError('Ambiguous sensory alias')
    mapping = pd.DataFrame({'R_name': nums['4.csv'].columns}).merge(panel,on='R_name',how='left',validate='one_to_one')
    mapping['scope'] = np.where(mapping.Category_Paper.isin(['Aroma','Taste']),
                                'aroma_flavour_primary_candidate', 'supplementary_palate_or_overall')
    mapping['original_scale_identified'] = False
    save_csv(output, 'sensory_mapping.csv', mapping)
    save_csv(output, 'scorecard_name_differences.csv',
             [{'field': c, 'location': 'scorecard_only'} for c in sorted(set(card['summary_names'])-set(nums['4.csv']))] +
             [{'field': c, 'location': '4csv_only'} for c in sorted(set(nums['4.csv'])-set(card['summary_names']))])

    matrices = {}
    correlation_comparison = []
    chemical_lookup = features.set_index('field_name_R')['chemical_name'].map(repair_utf8_mojibake).to_dict()
    # PanelFeedback distinguishes orange from F_hops_citrus; PaperLong contains
    # the same label for these two different variables in the supplied workbook.
    sensory_lookup = mapping.set_index('R_name')['CleanName_PanelFeedback'].to_dict()
    for filename, data, lookup in [('2.csv', nums['1.csv'], chemical_lookup), ('5.csv', nums['4.csv'], sensory_lookup)]:
        given = pd.read_csv(source / filename, index_col=0)
        matrices[filename] = square_matrix_checks(given)
        renamed = data.rename(columns=lookup)
        duplicates = renamed.columns[renamed.columns.duplicated(keep=False)].tolist()
        matrices[filename]['ambiguous_aliases_excluded'] = duplicates
        renamed = renamed.loc[:, ~renamed.columns.duplicated(keep=False)]
        common = [c for c in given.columns if c in renamed.columns]
        matrices[filename]['mapped_columns'] = len(common)
        matrices[filename]['unmapped_columns'] = [c for c in given.columns if c not in common]
        actual = given.loc[common,common].to_numpy(float)
        triangle = np.triu(np.ones(actual.shape,dtype=bool),k=1)
        for method in ['pearson','spearman']:
            # Rank + Pearson avoids adding scipy only for this audit.
            values = renamed[common]
            estimated = (pairwise_spearman(values) if method=='spearman' else values.corr()).to_numpy()
            valid = triangle & np.isfinite(actual) & np.isfinite(estimated)
            errors = abs(actual[valid]-estimated[valid])
            correlation_comparison.append({'file':filename,'method':method,'matched_columns':len(common),
                                           'pairs':int(valid.sum()),'mae':float(errors.mean()) if errors.size else None,
                                           'max_abs_error':float(errors.max()) if errors.size else None,
                                           'note':'Spearman re-ranks complete pairs; source preprocessing/rounding may cause differences'})
    cross = pd.read_csv(source/'6.csv',index_col=0)
    cross_values = numeric_frame(cross).to_numpy(float)
    matrices['6.csv'] = {'rows':len(cross),'columns':len(cross.columns),
                        'nonfinite':int((~np.isfinite(cross_values)).sum()),
                        'out_of_bounds':int((abs(cross_values)>1+1e-8).sum())}
    write_json(output/'matrix_checks.json',matrices)
    save_csv(output,'correlation_method_comparison.csv',correlation_comparison)

    styles = raw['1.csv'].tasting_category_fine.value_counts().rename_axis('style').reset_index(name='n')
    save_csv(output,'style_counts.csv',styles)
    metadata = {'original_scale_identified':False,
                'standardized_space_baseline_allowed':True,
                'original_0_6_inverse_transform_allowed':False,
                'reason':'No per-taster raw scores/normalization parameters established; card inputs are audited separately.',
                'primary_target_candidates':mapping.loc[mapping.scope.eq('aroma_flavour_primary_candidate'),'R_name'].tolist()}
    write_json(output/'model_readiness.json', metadata)
    summary = {'sample_count':len(raw['1.csv']),'chemical_feature_count':len(nums['1.csv'].columns),
               'sensory_target_count':len(nums['4.csv'].columns),'primary_target_candidates':len(metadata['primary_target_candidates']),
               'style_count':len(styles),'minimum_style_count':int(styles.n.min()),
               'missing_chemical_cells':int(nums['1.csv'].isna().sum().sum()),
               'missing_chemical_fraction':float(nums['1.csv'].isna().sum().sum()/nums['1.csv'].size),
               'chemical_constant_columns':nums['1.csv'].columns[nums['1.csv'].nunique().eq(1)].tolist(),
               'all_missing_forum_columns':nums['7.csv'].columns[nums['7.csv'].isna().all()].tolist(),
               'card_filled_inputs':sum(r['filled_input_cells'] for r in card['cards']),
               'reference_compared_columns':len(scales),'conflict_cells':len(conflicts),
               'zero_reference_conflict_cells':sum(r['reason']=='zero_reference_not_invertible' for r in conflicts),
               'nonzero_pair_conflict_cells':sum(r['reason']=='pair_disagrees' for r in conflicts),
               'positive_log10_match_columns':[r['field'] for r in scales if r['candidate']=='log10' and r['status']=='positive_pairs_match'],
               'sources_unchanged':all(sha256(source/i['file'])==i['sha256'] for i in inventory)}
    write_json(output/'summary.json',summary)
    project = Path(__file__).resolve().parents[2]
    manifest = {'started_at_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),'output':str(output),
                'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
                'openpyxl':openpyxl.__version__,
                'code_sha256':{str(p.relative_to(project)):sha256(p) for p in sorted((project/'src').rglob('*.py'))},
                'stage':'P0 audit only; no models trained','ai_assistance':'OpenAI Codex (GPT-6), 2026-10-23'}
    write_json(output/'run_manifest.json',manifest)
    report = f'''# A题数据审计结果

本报告由程序生成；未训练预测或生成模型。原始文件未修改：{summary['sources_unchanged']}。

## 已核验

- {summary['sample_count']} 个样本，{summary['chemical_feature_count']} 个化学/理化/综合特征，50 个感官目标。
- 按 8.xlsx 的 Aroma/Taste 分类得到 {summary['primary_target_candidates']} 个第一问候选主目标，其余维度保留作补充，最终题意仍须结合评分卡解释。
- 1、4、7 表经唯一 beer 键与风格一致性检查对齐；条码精度差异保留在 sample_mapping.csv。
- 化学特征共 {summary['missing_chemical_cells']} 个缺失值，占 {summary['missing_chemical_fraction']:.4%}；异常值仅标记、不自动删除。
- 共 {summary['style_count']} 个风格，最少 {summary['minimum_style_count']} 样本，不宜机械执行严格风格分层五折。
- 评分卡共 {len(card['cards'])} 张，ISBLANK 公式引用的输入格填入数为 {summary['card_filled_inputs']}；summary 的 {card['summary_formula_cells']} 个公式中 {card['summary_cached_zero_cells']} 个缓存为零。禁止把模板默认值作标签。
- 对照校准表的 70 列中，{len(summary['positive_log10_match_columns'])} 列在正值配对上与 log10 一致（容差见代码）。这包括 pH、色度、苦度、酒精度；与题面概述有冲突。
- 校准对照发现 {summary['conflict_cells']} 个不一致/零参考不可逆单元格，其中 {summary['zero_reference_conflict_cells']} 个为参考零值、{summary['nonzero_pair_conflict_cells']} 个为其他配对差异，详见 calibrated_conflicts.csv。正值匹配不意味着所有样本均可无损逆变换。
- 7.csv 的全缺失字段：{', '.join(summary['all_missing_forum_columns'])}。数据说明将其定义为论坛评论描述符，不直接视为专家品评分。

## 主要限制与处理

1. **0～6 分原始尺度未识别**：原研究说明按品评员标准化，当前未取得原始评分或相应参数。不能把全样本 min-max 拉伸或模板零值解释为真实评分恢复。标准化空间基线可以继续；最终分级需论证校准方法或获取补充信息。
2. **数据说明冲突**：附件数据说明将相关矩阵称为 Spearman，题面将表2称为 Pearson。correlation_method_comparison.csv 提供可匹配字段的重算差异；匹配不足、缺失处理及变换会影响结果，不以误差较小单独证明生成方法。
3. **校准值冲突**：零参考、少量非零差异不得自动覆盖原表；化合物阈值计算须核实源尺度和单位。
4. **汇总与列名差异**：scorecard_name_differences.csv 列出评分卡与标签表的名称差异。已有 GBM/SHAP 只作参考，不作训练真值或本队实验结果。
5. **有限抽取范围**：已检查公式及其引用输入格；未进行 Excel 原生引擎重算或逐页视觉检查，未把缓存值当重算证据。
6. **别名元数据问题**：Sensory names 的 CleanName_PaperLong 将 orange 与 F_hops_citrus 写成同一名称；相关矩阵核验改用可区分的 CleanName_PanelFeedback。化合物名称中的可逆 UTF-8/Latin1 乱码修复仅作用于比较映射，不改原文件。未匹配名称详见 matrix_checks.json。

## 下一步

在保留上述限制的前提下建立训练折内预处理和标准化目标的基线验证。评分尺度核验、缺陷阈值和列名映射继续独立推进；正式 0～6 分交付尚未通过验收。

来源：本轮 data_inventory.csv 中的原始附件；[原研究 Methods / Sensory dataset](https://pmc.ncbi.nlm.nih.gov/articles/PMC10966102/)，访问日期 2026-10-23。数值发现来自本地附件重算。
'''
    (output/'data_quality_report.md').write_text(report,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description='Audit original A attachments without modifying them')
    parser.add_argument('--source',type=Path,required=True,help='Folder containing 1.csv through 8.xlsx')
    parser.add_argument('--output',type=Path,required=True,help='New output directory; existing paths are rejected')
    args=parser.parse_args()
    try:
        run(args.source,args.output)
    except Exception as exc:
        # Never allow a partially generated folder to be mistaken for success.
        print(f'AUDIT FAILED: {type(exc).__name__}: {exc}')
        raise
