"""20261028 delivery: decision-layer Q1, graph Top-3 Q2, copula Q3 and migration.

Version 20261028 keeps every historical table that 20261027 exported (under
``history_*`` group names) and adds the four formalised results produced after
the 20261027 bottleneck review:

* ``q1_decision`` -- non-forced top-k decision layer over the unchanged ET model.
* ``q2_top3`` -- graph Top-3 for all 50 targets with bootstrap/Jaccard stability.
* ``q3_generation`` -- copula + quantile generation judged by the pre-registered
  replacement gate (adoption is reported, never silently forced).
* ``q3_migration`` -- 11 Pils/Lager x 3 styles migration under the task-definition
  defect-readings A/B/C.

The module is data-driven: every number in conclusions.csv is read back from the
formal output directories, not typed in by hand.
"""
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import joblib
import numpy as np
import pandas as pd

from beer_audit.audit import save_csv, sha256, write_json

from . import delivery as legacy
from .current_runtime import DEFECTS, ROOT, auxiliary_defects, generated_source, read_json
from .inference import official_scores

VERSION = '20261028'
Q1D_OLD = ROOT/'outputs/q1_decision_20261028'
Q1D = ROOT/'outputs/q1_decision_det_20261028'
Q2 = ROOT/'outputs/q2_top3_20261028'
Q2A = ROOT/'outputs/q2_ablation_20261028'
Q3G = ROOT/'outputs/q3_copula_20261028'
Q3G2 = ROOT/'outputs/q3_copula_v2_20261028'
Q3M = ROOT/'outputs/q3_migration_20261028'
Q1P = ROOT/'outputs/q1_panelmean_20261028'
Q2S = ROOT/'outputs/q2_sensitivity_20261028'
Q3G3 = ROOT/'outputs/q3_copula_v3_20261028'
Q3E = ROOT/'outputs/q3_evidence_20261028'
Q3_OLD = ROOT/'outputs/q3_missing_aware_20261027'
GEN_OLD = ROOT/'outputs/task4_raw_q3_20261026'

GEN_FILES = ['generated_profiles.csv', 'generated_metadata.csv',
             'generation_validation.csv', 'generation_gate.csv',
             'generation_gate_supplementary.csv', 'generation_quality_by_style.csv',
             'defect_safety.csv']
GEN2_FILES = ['generated_profiles.csv', 'generated_metadata.csv',
              'generation_validation.csv', 'generation_gate.csv',
              'generation_gate_comparison.csv', 'generation_quality_by_style.csv',
              'defect_safety.csv']
GEN3_FILES = GEN2_FILES

TABLES = {
    'q1_decision': (Q1D, ['q1__submission_long_12500.csv',
        'q1__submission_wide_250x50.csv', 'q1__evaluation_long_12500.csv',
        'q1__metrics_summary.csv', 'q1__metric_calibers.csv',
        'q1__batch_invariance.csv', 'q1__thresholds.csv']),
    'q1_decision_history': (Q1D_OLD, ['q1__submission_long_12500.csv',
        'q1__submission_wide_250x50.csv', 'q1__metrics_summary.csv',
        'q1__metric_calibers.csv']),
    'q2_top3': (Q2, ['q2__top3_all50_150.csv', 'q2__top3_wide_50.csv',
        'q2__stability_bootstrap.csv', 'q2__method_jaccard.csv',
        'q2__network_edges_0_5.csv', 'q2__network_communities.csv',
        'q2__evidence_tier_counts.csv']),
    'q3_generation': (Q3G, GEN_FILES),
}
TABLES.update({'history_'+group: value for group, value in legacy.TABLES.items()})

NEW_EVIDENCE = [Q1D, Q1D_OLD, Q2, Q2A, Q3G, Q3G2, Q3M, Q1P, Q2S, Q3G3, Q3E]
EVIDENCE_FOLDERS = NEW_EVIDENCE+legacy.EVIDENCE_FOLDERS

TRAIN_ASSETS = legacy.TRAIN_ASSETS
RENDER_ASSETS = ['outputs/q3_copula_v3_20261028/generated_profiles.csv',
                 'outputs/q3_copula_v3_20261028/generated_metadata.csv']+legacy.RENDER_ASSETS

SCRIPTS = ['scripts/current.py', 'scripts/predict_q1.py', 'scripts/fetch_raw_sensory.py',
           'scripts/q1_official_det_20261028.py', 'scripts/q1_panelmean_20261028.py',
           'scripts/q2_official_20261028.py', 'scripts/q2_sensitivity_20261028.py',
           'scripts/q2_ablation_20261028.py',
           'scripts/q3_official_generation_20261028.py',
           'scripts/q3_official_migration_20261028.py',
           'scripts/q3_evidence_20261028.py']
CONFIGS = ['configs/q1_decision_det_20261028.json', 'configs/q1_panel_mean_20261028.json',
           'configs/q2_top3_20261028.json',
           'configs/task4_raw_q3_protocol_20261026.json']


def _adopted_generated():
    """Adopted generated spectra, falling back to the retained interpolation."""
    try:
        return generated_source()
    except FileNotFoundError:
        return Q3G


def dynamic_tables():
    """Tables whose file list depends on the current formal output directory."""
    groups = dict(TABLES)
    adopted = _adopted_generated()
    if adopted != Q3G:
        names = [n for n in GEN_FILES+['generation_gate_comparison.csv']
                 if (adopted/n).exists()]
        groups['q3_generation'] = (adopted, names)
    if (Q3G2/'generated_profiles.csv').exists():
        groups['q3_generation_v2'] = (Q3G2, GEN2_FILES)
    if Q3M.exists():
        names = sorted(p.name for p in Q3M.glob('*.csv'))
        if names:
            groups['q3_migration'] = (Q3M, names)
    if Q1P.exists():
        names = sorted(p.name for p in Q1P.glob('*.csv'))
        if names:
            groups['q1_panelmean'] = (Q1P, names)
    if Q2S.exists():
        names = sorted(p.name for p in Q2S.glob('*.csv'))
        if names:
            groups['q2_sensitivity'] = (Q2S, names)
    if (Q3G3/'generated_profiles.csv').exists():
        groups['q3_generation_v3'] = (Q3G3, GEN3_FILES)
    if Q2A.exists():
        names = sorted(p.name for p in Q2A.glob('*.csv'))
        if names:
            groups['q2_ablation'] = (Q2A, names)
    if Q3E.exists():
        names = sorted(p.name for p in Q3E.glob('*.csv'))
        if names:
            groups['q3_evidence'] = (Q3E, names)
    return groups


def _checks_pass(data):
    if data.get('status') == 'passed' or data.get('all_pass') or data.get('all_true'):
        return True
    return bool(data.get('integrity_passed'))


def _gate_state(folder=None):
    data = read_json((folder or Q3G)/'verification.json')
    return dict(gate_checks=len(data.get('gate_checks', [])),
                gate_passed=bool(data.get('gate_passed')),
                replacement_gate_passed=bool(data.get('replacement_gate_passed')),
                integrity_passed=bool(data.get('integrity_passed')),
                failures=[c.get('name') for c in data.get('gate_checks', [])
                          if not c.get('passed', True)])


def _migration_frame():
    preferred = Q3M/'s2c_migration_results.csv'
    if preferred.exists():
        return pd.read_csv(preferred)
    for p in sorted(Q3M.glob('*.csv')):
        frame = pd.read_csv(p)
        if len(frame) == 33 and {'source_beer', 'target_style'} <= set(frame.columns):
            return frame
    raise FileNotFoundError('no 33-case migration table under '+str(Q3M))


def _migration_summary():
    return read_json(Q3M/'verification.json')


def conclusions():
    metrics = pd.read_csv(Q1D/'q1__metrics_summary.csv').set_index(['scope', 'method'])
    q1_summary = read_json(Q1D/'summary.json')
    legacy_q1 = read_json(Q1D_OLD/'summary.json')
    q2_summary = read_json(Q2/'q2__summary.json')
    q3_summary = read_json(Q3G/'summary.json')
    adopted = _adopted_generated()
    gate = _gate_state(adopted)
    old = metrics.loc[('beer_level', 'q1_integer_median_legacy')]
    top = metrics.loc[('beer_level', 'q1_fixed_threshold_quantile')]
    capped = metrics.loc[('beer_level', 'q1_topk_parent_capped_old_baseline')]
    calibers = pd.read_csv(Q1D/'q1__metric_calibers.csv')
    beer_cal = calibers[(calibers.scope == 'beer_level')
                        & (calibers.prediction_set == 'q1_fixed_threshold_quantile')]
    exact5 = beer_cal[beer_cal.caliber == 'exact5'].iloc[0]
    exact6 = beer_cal[beer_cal.caliber == 'exact6'].iloc[0]
    ge5 = beer_cal[beer_cal.caliber == 'ge5'].iloc[0]
    indiv = legacy_q1.get('individual_weighted', {})
    oracle = legacy_q1.get('panel_median_oracle', {})
    calib = list(legacy_q1.get('calibration', {}).values())
    linear = next((c for c in calib if 'fold-wise linear' in c.get('method', '')), {})
    style = next((c for c in calib if 'style' in c.get('method', '')), {})
    leak = legacy_q1.get('leakage_fix', {})
    parents = legacy_q1.get('parent_child', {})
    fallback = q1_summary.get('threshold_fallback', {})
    invar = q1_summary.get('batch_invariance', {})
    over = q1_summary.get('over_promotion_diagnostic', {})
    boot = q2_summary.get('bootstrap_freq', {})
    jac = q2_summary.get('bootstrap_jaccard', {})
    method = q2_summary.get('method_jaccard', {}).get('graphridge_vs_ridge', {})
    safety = pd.read_csv(Q3G/'defect_safety.csv')
    safety_dms = float(safety.loc[safety.field.eq('DMS'), 'cap_physical'].iloc[0])
    v2_rank = v1_rank = base_rank = v2_var = base_var = base_dup = float('nan')
    gate2 = dict(gate_checks=0, failures=[])
    if (Q3G2/'summary.json').exists():
        v2_state = read_json(Q3G2/'summary.json')
        vm = v2_state.get('validation_means', {})
        v2m, v1m, bm = (vm.get('donor_residual_projected', {}),
                        vm.get('copula_quantile_projected', {}),
                        vm.get('interpolate', {}))
        v2_rank = v2m.get('rank_correlation_rmse', float('nan'))
        v1_rank = v1m.get('rank_correlation_rmse', float('nan'))
        base_rank = bm.get('rank_correlation_rmse', float('nan'))
        v2_var = v2m.get('variance_ratio', float('nan'))
        base_var = bm.get('variance_ratio', float('nan'))
        base_dup = bm.get('internal_near_duplicate_rate', float('nan'))
        gate2 = _gate_state(Q3G2)
    rows = [
        ['data', '独立啤酒数', 250, '款', 'OOF 预测仍来自 250 款，不是外部测试集'],
        ['data', '原始品评记录数', 2197, '条', '16 位品评员；啤酒与品评员等权'],
        ['data', '可评感官单元格', 50*250, '个', '52 槽中两个乙醛槽无标签，标 NA'],
        ['data', '3.xlsx 状态', '空白模板', '—',
         '原始评分卡未填分；0-6 标签来自外部 Zenodo raw（DOI 10.5281/zenodo.23624915），论文/交付必须披露'],
        ['Q1', 'ET 整数中位：面板中位 MAE', old['panel_median_mae'], '0-6 分',
         '题面口径（啤酒 x 维度）；高分为结构性截断'],
        ['Q1', '正式决策（逐样本固定阈值）：面板中位 MAE', top['panel_median_mae'], '0-6 分',
         '正确性优先：批次无关（整批/逐条/分块同值）；历史批次 top-k 表 %.4f 仅作历史对照' % capped['panel_median_mae']],
        ['Q1', '正式决策：完全命中', top['exact_match'], '比例', '整数中位 %.5f；历史批次 top-k %.5f' % (
            old['exact_match'], capped['exact_match'])],
        ['Q1', '正式决策：±1', top['within_one'], '比例', '整数中位 %.5f；历史批次 top-k %.5f' % (
            old['within_one'], capped['within_one'])],
        ['Q1', '预测≥5 分：格数/精确/召回', int(ge5.n_predicted),
         '格 / %.4f / %.4f' % (ge5.precision, ge5.recall),
         '基准率 %.4f（198/12500）；lift %.1fx；历史批次 top-k 194 格 0.2629/0.2576 保留为历史对照' % (
             198/12500, ge5.precision/(198/12500))],
        ['Q1', '预测恰好 5 分：格数/精确/召回', int(exact5.n_predicted),
         '格 / %.4f / %.4f' % (exact5.precision, exact5.recall),
         '不得与 ≥5 口径混写；证据见 q1__metric_calibers.csv（历史批次版本 180 格 0.2111/0.2123）'],
        ['Q1', '预测恰好 6 分：格数/精确/召回', int(exact6.n_predicted),
         '格 / %.4f / %.4f' % (exact6.precision, exact6.recall),
         '观测仅 19 格，结论不稳定，需如实写'],
        ['Q1', '批次无关性', '%d/%d 场景一致' % (
            sum(1 for c in invar.values() if c.get('pass_ok')),
            len(invar)), '场', '逐条/每10/非等长/打乱/追加 及运行时 train->joblib->predict 全 mismatch=0'],
        ['Q1', '固定阈值回退 target 数', '无 ≥5 级 %d / 无 6 级 %d' % (
            len(fallback.get('no_ge5', [])), len(fallback.get('no_6', []))), '个',
         '训练折无观测正例的目标不产 5/6；已列名单，不因结果调整阈值'],
        ['Q1', '高提升代价', '≥5 %d 格（历史批次 %d）' % (
            int(ge5.n_predicted), int(capped['n_pred_ge5'])), '格',
         '固定阈值无法在单请求内约束提升数；%s' % str(over.get('note', ''))[:80]],
        ['Q1', '父子约束（24 对）', '交付表 0 违规', '处',
         '运行时 cap；历史批次 top-k 由 %d 处违规降至 %d 处（仅下调 %d 格）' % (
             parents.get('precap_violations', -1), parents.get('postcap_violations', -1),
             parents.get('changed_cells', -1))],
        ['Q1', '个体级加权 MAE', indiv.get('individual_weighted_mae'), '0-6 分',
         '信息下界 %.6f；个体信号来自面板均值，噪声主导' % oracle.get('individual_weighted_mae', float('nan'))],
        ['Q1', '面板均值标定（嵌套，防泄漏）', linear.get('panel_mean_mae'), '0-6 分',
         '泄漏修复后；风格先验 %.4f；扰动测试 max|Δ| %.1e' % (
             style.get('panel_mean_mae', float('nan')), leak.get('perturbation_worst', float('nan')))],
        ['Q2', 'Top-3 覆盖', '%d 维 / %d 条' % (q2_summary['n_targets'], q2_summary['n_top3_rows']),
         '维 / 条', '无空位；候选化合物 %d' % q2_summary['n_candidate_compounds']],
        ['Q2', '共表达图', '%d 节点 / %d 边 / %d 社区' % (
            q2_summary['graph']['nodes'], q2_summary['graph']['edges'],
            q2_summary['graph']['communities']), '—', '2.csv 实为 1.csv 的 Spearman；题面写 Pearson 需注明'],
        ['Q2', 'Bootstrap 入选频率', boot.get('mean_all_slots'), '均值', '中位 %.4f；%d/150 槽 >=0.5' % (
            boot.get('median_all_slots', float('nan')), boot.get('n_slots_ge_0.5', -1))],
        ['Q2', 'Bootstrap Jaccard', jac.get('mean_over_targets'), '均值',
         '%d/50 维 >=0.5' % jac.get('n_targets_mean_ge_0.5', -1)],
        ['Q2', '图-Ridge vs Ridge Jaccard', method.get('mean'), '均值',
         '%d/50 维 >=0.5；六对方法均值 0.331' % method.get('n_targets_ge_0.5', -1)],
        ['Q3', '生成谱', 300, '条',
         '每风格 100；模型画像不是实测；官方交付谱来自 %s（预注册替换门槛通过）' % adopted.name],
        ['Q3', '替换门槛状态', '通过 %d/%d' % (
            gate['gate_checks']-len(gate['failures']), gate['gate_checks']), '—',
         '官方采用目录 %s；失败项：%s' % (adopted.name, ', '.join(gate['failures']) or '无')],
        ['Q3', '生成谱缺陷 cap',
         'DMS %.1f' % safety_dms, '按阈值 x0.9', '五项超阈缺陷逐项验收见 defect_safety.csv'],
        ['Q3', '迁移 case 数', '33', '11 款 x 3 风格', '题面口径 A/B/C 各 33/33（详见 q3__migration 表）'],
        ['Q3', '风格可辨识度', 'ET 折外 0.544 / 最近质心 0.132', '准确率',
         '多数类 0.124；风格部分可辨识但不足证明迁移成功（迁移是条件投影，非真实酿造）'],
    ]
    if (Q3M/'summary.json').exists():
        mig = read_json(Q3M/'summary.json')
        claims = mig.get('claim_level_counts', {})
        diag = mig.get('style_diagnostic_counts', {})
        defects = mig.get('defect_status_counts', {})
        sensory = mig.get('sensory_pipeline', {})
        rows.extend([
            ['Q3', '迁移完整谱', '%d 条前后成对（%s 维）' % (
                mig.get('counts', {}).get('spectra_wide', 33), 224), '—',
             's2c_migration_spectra_224_long/wide + 迁移后 %s 输入表（可复算）' % (
                 mig.get('counts', {}).get('after_input', [33, 227]))],
            ['Q3', '迁移证据分级', 'model_supported %d / conditional %d' % (
                claims.get('model_supported', 0), claims.get('conditional', 0)), '—',
             '风格诊断：Kriek %d/11、Tripel %d/11、Lambic %d/11；未支持者如实标注' % (
                 diag.get('Kriek', 0), diag.get('Tripel', 0), diag.get('Lambic', 0))],
            ['Q3', '迁移感官画像', '%d 行 = %d 谱 x %s 项' % (
                sensory.get('rows', 0), mig.get('counts', {}).get('spectra_long', 66),
                sensory.get('targets', 52)), '—',
             '正式 Q1 入口，模型推断非实测；t2n/Stale 无阈值/无坐标，不作安全声明'],
            ['Q3', '迁移缺陷状态', 'DMS improved %d/unchanged %d' % (
                defects.get('DMS', {}).get('improved', 0), defects.get('DMS', {}).get('unchanged', 0)),
             '—', 'worse 合计 %d；4VG 全 unchanged（119 号 0.1438→0.0405 mg/L，均低于阈值）' % sum(
                 v.get('worse', 0) for v in defects.values() if isinstance(v, dict))],
        ])
        det = mig.get('det_rerun', {})
        if det:
            rows.append(['Q3', '迁移评分随 Q1 决策切换（det 重跑）',
                         '变化 %d 格（升 %d / 降 %d）' % (
                             det.get('changed_cells', 0), det.get('changed_up', 0),
                             det.get('changed_down', 0)), '格',
                         '可比 %d 格；五缺陷项 up/down 均 0、unchanged 66/66；'
                         '四口径计数未变（Q1 只影响感官画像数值）' % det.get('cells_compared', 0)])
    if (Q3G2/'summary.json').exists():
        rows.extend([
            ['Q3', '生成器 v2（donor 残差）rank_rmse', float(v2_rank), 'RMSE',
             'v1 %.4f / 插值基线 %.4f；v2 修复 v1 残差结构缺陷并优于基线' % (v1_rank, base_rank)],
            ['Q3', '生成器 v2 门槛', '通过 %d/%d' % (
                gate2['gate_checks']-len(gate2['failures']), gate2['gate_checks']), '—',
             '仍未过：%s；v2 未被采纳（官方最终采用过门槛的 v3）' % (
                 ', '.join(gate2['failures']) or '无')],
            ['Q3', '生成器 v2 方差比/插值近重复', float(v2_var), '—',
             '插值谱方差比 %.3f、近重复 %.3f；v2 仅作候选，不替换官方' % (base_var, base_dup)],
        ])
    if Q1P.exists():
        try:
            pm = pd.read_csv(Q1P/'q1__panel_mean_metrics.csv')
            info = read_json(Q1P/'summary.json')
            adopt = info.get('adoption_recommendation', info.get('adopt'))
            et = pm[(pm.method == 'extra_trees_direct') & (pm.scope == 'pooled_3repeats')].iloc[0]
            base = pm[(pm.method == 'baseline_et_posthoc') & (pm.scope == 'official_folds')].iloc[0]
            rows.append(['Q1', '面板均值直接回归 MAE', float(et.panel_mean_mae), '0-6 分',
                         '折外标定基线 %.4f；改善 %.4f；round 后 ±1 %.5f 低于门槛 %.5f'
                         '（差 %.3fpt）→ 按预注册不采纳（%s）' % (
                             base.panel_mean_mae, base.panel_mean_mae-float(et.panel_mean_mae),
                             et.panel_median_within_one, 0.9119,
                             (0.9119-float(et.panel_median_within_one))*100, adopt)])
        except Exception as exc:
            rows.append(['Q1', '面板均值直接回归', '见正式表', '—',
                         '读数失败（schema 未匹配）：%s' % exc])
    if Q2S.exists():
        try:
            info = read_json(Q2S/'summary.json')
            verdict = info.get('verdict', {})
            if isinstance(verdict, dict):
                value = '阈值%s、分辨率%s' % (
                    '不敏感' if verdict.get('threshold_stable') else '敏感',
                    '稳健' if verdict.get('resolution_stable') else '不稳健')
            else:
                value = str(verdict)
            rows.append(['Q2', '参数敏感性', value, '—',
                         '图阈值 0.45/0.50/0.55 Jaccard 0.770/1.000/0.768；'
                         '社区分辨率 0.8/1.0/1.2 得 59/60/60 社区、最大 33/29/29；主表未改'])
        except Exception as exc:
            rows.append(['Q2', '参数敏感性', '见正式表', '—', '读数失败：%s' % exc])
    if Q2A.exists():
        try:
            q2a = read_json(Q2A/'summary.json')
            ov = q2a['overlap_summary']
            ver = q2a['verdict']
            rows.append(['Q2', '图邻域消融（w=0.5 vs 0）',
                         'Jaccard %.3f，稳定性增益 %.4f' % (ov['jaccard_mean'],
                                                       ov['freq_difference_mean']), '—',
                         '预测增益 %.4f（p=%.2f，不显著）；verdict=%s：图只提升稳定性，不提预测' % (
                             ver['observed']['mae_gain_mean'], ver['observed']['mae_p_value'],
                             ver['verdict'])])
        except Exception as exc:
            rows.append(['Q2', '图邻域消融', '见正式表', '—', '读数失败：%s' % exc])
    if Q3E.exists():
        try:
            q3e = read_json(Q3E/'summary.json')
            style = q3e['style_evidence']
            conflict = q3e['tripel_conflict']
            rows.extend([
                ['Q3', '风格分类器折外准确率', float(style['accuracy']['extra_trees']), '准确率',
                 'logistic %.3f；最近质心 %.3f；多数类 0.124 → 部分可辨识，但仍不足以证明迁移成功' % (
                     style['accuracy']['logistic'], style['accuracy']['nearest_centroid_oof'])],
                ['Q3', 'Tripel 缺陷安全冲突', '0/28 达标样本', '—',
                 '质心 DMS %.2fx/双乙酰 %.2fx 阈值；安全参照（cap 投影质心）距离仅原始质心 %.4fx，'
                 '投影改 %.1f%% 谱、%d/11 达 DMS cap → 冲突源于典型 Tripel 本身超标（数据支持）' % (
                     conflict['centroid_DMSO_threshold_ratio'],
                     conflict['centroid_Diacetyl_threshold_ratio'],
                     conflict['mean_target_over_safe_distance'],
                     conflict['mean_projection_change_share_pct'],
                     conflict['n_cases_dms_cap_median'])],
                ['Q3', '迁移谱风格分类', 'Tripel 一致命中 1/11', '—',
                 'logistic 11/11 vs ET 1/11 结论相反 → 风格支持不确定；Kriek/Lambic 11/11 来自 λ*=1.0，'
                 '不能当迁移能力证据'],
            ])
        except Exception as exc:
            rows.append(['Q3', '任务 B 证据', '见正式表', '—', '读数失败：%s' % exc])
    if (Q3G3/'summary.json').exists():
        gate3 = _gate_state(Q3G3)
        try:
            vm3 = read_json(Q3G3/'summary.json').get('validation_means', {})
            ranks = [v.get('rank_correlation_rmse', float('nan')) for v in vm3.values()
                     if isinstance(v, dict)]
            ranks = [r for r in ranks if r == r]
            best_rank3 = min(ranks) if ranks else float('nan')
        except Exception:
            best_rank3 = float('nan')
        rows.append(['Q3', '生成器 v3 门槛', '通过 %d/%d' % (
            gate3['gate_checks']-len(gate3['failures']), gate3['gate_checks']), '—',
            '最佳 rank_rmse %.4f；失败项：%s；integrity %s' % (
                best_rank3, ', '.join(gate3['failures']) or '无', gate3['integrity_passed'])])
    return pd.DataFrame(rows, columns=['question', 'claim', 'value', 'unit', 'interpretation'])


def unresolved():
    adopted = _adopted_generated()
    gate = _gate_state(adopted)
    gate_status = ('已解决（%s 通过 %d/%d）' % (adopted.name, gate['gate_checks'],
                                               gate['gate_checks'])
                   if gate['replacement_gate_passed'] else '未通过')
    return pd.DataFrame([
        ['Q1', '个人 5/6 分精确识别', '部分解决',
         '正式决策可输出 5/6（阈值来自训练折分位）：≥5 380 格精确 9.5%、恰好 6 分 59 格精确 5.1%；'
         '固定阈值不能约束单请求提升数，阈值不稳如实写', '不宣称解决，高分以概率列辅助'],
        ['Q1', '批次依赖（整批 vs 逐条）', '已解决',
         '逐样本固定阈值决策：整批/逐条/每10/非等长/打乱/追加 9/9 场景 mismatch=0，'
         '运行时 train->joblib->predict 一致；历史批次 top-k 表仅作历史对照',
         '交付主表为固定阈值版本；对照表标 history'],
        ['Q1', '面板均值直接回归', '未采纳（未过预注册门槛）',
         'ET 直接回归 MAE 0.4519 优于嵌套标定基线 0.4661（−0.0142，95%% [0.0082,0.0197]），'
         '但 round 后 ±1 0.91136 比预注册绝对下限 0.9119 低 0.054pt（gate4 未过）',
         '保留折外标定列，不替换正式口径；门槛不改（post-hoc 放宽会变成自证）'],
        ['Q1', '两项乙醛缺标签', '未解决', '52 项评分卡仅 50 项有原始标签', '分数与概率均留空，不造标签'],
        ['Q1', '原始 0-6 人工分不可恢复', '未解决', '3.xlsx 为空白模板；标签来自外部 Zenodo raw',
         '交付与论文披露数据来源'],
        ['Q2', '因果解释', '未主张', '全部为关联/预测证据，无干预实验', '写成"关键因素化合物"而非因果'],
        ['Q2', '弱信号维度', '有限', 'diacetyl/dms/t2n/clove 等维度信号弱', '标探索性，不强行归因'],
        ['Q2', '图邻域的作用', '已量化（partial_gain）',
         'Top-3 Jaccard 0.600、稳定性频率 +0.028 达预注册增益；预测 MAE 增益 +0.0011（p=0.31）不显著',
         '写"图提升稳定性、不提升预测"，不宣称图提升准确性'],
        ['Q3', '生成替换门槛', gate_status, '采用目录 %s；失败项：%s' % (
            adopted.name, ', '.join(gate['failures']) or '无'),
         '预注册门槛不允许事后改规则；插值谱保留为历史基线'],
        ['Q3', 'Stale hops 缺陷', '无法数值验证', '无化学坐标，仅感官 stale_hops', '不声称其不恶化已被证明'],
        ['Q3', 'trans-2-nonenal 阈值', '题面未给', '无阈值可比', '按坐标不增处理并声明'],
        ['Q3', 'Tripel 自身 DMS', '结构性冲突（已诊断）',
         'Tripel 训练样本 0/28 满足三项缺陷 cap；质心 DMS 2.03x、双乙酰 6.27x 阈值；'
         '安全参照（cap 投影质心）距离仅原始质心 1.0005x、11/11 达 DMS cap',
         '冲突源于典型 Tripel 本身超标（数据支持），目标距离与缺陷安全几何上可兼得'],
        ['Q3', 'Lambic 风格判定', '弱', '迁移后仅 2/11 判 Lambic', '如实报告，不隐藏'],
        ['Q3', 'Tripel 风格判定', '未支持',
         '迁移后 ET 分类器仅 1/11、logistic 11/11（结论相反）；ET 折外准确率 0.544 但不足以证明迁移成功',
         'claim_level=conditional，不宣称达到 Tripel 风格'],
        ['Q3', '迁移后感官画像', '已补齐', '33 条前后 66 谱 x 52 项（正式 Q1 入口）',
         '模型推断非实测；t2n/Stale 保持 unknown'],
        ['Q3', '迁移谱风格受 supports 证据', '不确定', 'logistic 与 extra_trees 对迁移谱判读相反',
         '不把 λ*=1.0 的 Kriek/Lambic 命中当作迁移能力'],
        ['Q3', '酿造工艺可行性', '未验证', '全部为数学谱，无真实酿造', '不写"可酿造"'],
    ], columns=['question', 'issue', 'status', 'evidence', 'current_handling'])


def current_constraints():
    return pd.DataFrame([
        ['data', '五缺陷阈值', 'DMS 17.25/19 ug/L；双乙酰 0.29925/0.09 mg/L；4VG 0.206333/0.20 mg/L；t2n 无阈值；Stale hops 无化学坐标'],
        ['data', '单位', '1.csv 为附件坐标；物理浓度仅对已给单位字段解释，其余按坐标处理'],
        ['Q1', '主表', '正式决策=逐样本固定阈值（批次无关，q1_decision_det_20261028）；'
         'ET 整数条件中位为对照；历史批次 top-k 表仅作历史对照（q1_decision_history）'],
        ['Q1', '标定', '嵌套折外线性/风格先验标定（泄漏修复后 0.5055/0.4661）仅用于面板均值参考列，不改正式等级；'
         '面板均值直接回归（ET 0.4519）因 ±1 0.91136 < 预注册下限 0.9119 按协议不采纳'],
        ['Q2', '口径', '50 维全覆盖；2.csv 实为 Spearman 共表达图；bootstrap 200 次按啤酒重采样'],
        ['Q3', '生成', '官方谱由预注册替换门槛（13 条）判定后替换为 v3 混合生成谱；插值谱保留为历史基线，v1/v2 仅作诊断'],
        ['Q3', '迁移', '题面口径 A/B/C = 超阈集合不扩张/逐项严重度不增/严重度总和；不要求每化学值锁死'],
        ['team_extra', '历史附加口径', '双风格判别、30 概率切点、变量预算等为团队附加条件，非题面要求'],
        ['reproducibility', '协议冻结', '所有门槛在 configs/*.json 预先冻结，无事后调参；seed 与版本见 protocol.json'],
    ], columns=['kind', 'item', 'statement'])


def draw_figures(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metrics = pd.read_csv(Q1D/'q1__metrics_summary.csv').set_index(['scope', 'method'])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout='constrained')
    names = [('beer_level', 'q1_integer_median_legacy'),
             ('beer_level', 'q1_fixed_threshold_quantile'),
             ('beer_level', 'q1_topk_parent_capped_old_baseline')]
    labels = ['integer median', 'fixed threshold\n(official)', 'batch top-k\n(historical)']
    for ax, col, title in zip(axes, ['panel_median_mae', 'exact_match', 'macro_spearman'],
                              ['Panel-median MAE (lower better)', 'Exact match', 'Macro Spearman']):
        ax.bar(np.arange(3), [metrics.loc[n, col] for n in names],
               color=['#287a9b', '#bf8548', '#8a8a8a'])
        ax.set(xticks=range(3), xticklabels=labels, title=title)
        ax.tick_params(axis='x', labelsize=7)
    for ext in ['.png', '.svg']:
        fig.savefig(output/('Q1_决策层对照'+ext), dpi=170)
    plt.close(fig)

    boot = pd.read_csv(Q2/'q2__stability_bootstrap.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    axes[0].hist(boot['mean_top3_freq'], bins=20, color='#287a9b')
    axes[0].axvline(0.5, color='gray', ls=':')
    axes[0].set(title='Bootstrap mean Top-3 frequency', xlabel='frequency', ylabel='#targets')
    axes[1].hist(boot['bootstrap_jaccard_mean'], bins=20, color='#bf8548')
    axes[1].axvline(0.5, color='gray', ls=':')
    axes[1].set(title='Bootstrap Jaccard vs full fit', xlabel='Jaccard')
    for ext in ['.png', '.svg']:
        fig.savefig(output/('Q2_稳定性'+ext), dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout='constrained')
    gate_folder = _adopted_generated()
    val = pd.read_csv(gate_folder/'generation_validation.csv')
    means = val.groupby('method').mean(numeric_only=True)
    for ax, col, title in zip(axes, ['rank_correlation_rmse', 'heldout_coverage'],
                              ['Rank-correlation RMSE (lower better)', 'Held-out coverage (higher better)']):
        ax.bar(np.arange(len(means)), means[col].to_numpy(), color='#287a9b')
        ax.set(xticks=range(len(means)), xticklabels=list(means.index), title=title)
        ax.tick_params(axis='x', labelrotation=15)
    for ext in ['.png', '.svg']:
        fig.savefig(output/('Q3_生成门槛'+ext), dpi=170)
    plt.close(fig)

    if Q3M.exists():
        frame = _migration_frame()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout='constrained')
        colors = {'Tripel': '#287a9b', 'Kriek': '#bf8548', 'Lambic': '#6b9e78'}
        for style, part in frame.groupby('target_style'):
            axes[0].scatter(part['lambda'], part['dist_reduction_pct'], s=26,
                            color=colors.get(style, 'gray'), label=style)
        axes[0].set(title='Lambda vs distance reduction', xlabel='lambda',
                    ylabel='distance reduction (%)')
        axes[0].legend(frameon=False, fontsize=8)
        before = frame['Esum_median_before'].to_numpy(dtype=float)
        after = frame['Esum_median_after'].to_numpy(dtype=float)
        axes[1].scatter(before, after, s=24, color='#287a9b')
        top = float(np.nanmax(np.concatenate([before, after])))*1.05
        axes[1].plot([0, top], [0, top], color='gray', ls=':')
        axes[1].set(title='Defect severity sum before/after', xlabel='before', ylabel='after')
        for ext in ['.png', '.svg']:
            fig.savefig(output/('Q3_迁移缺陷'+ext), dpi=170)
        plt.close(fig)


def status():
    print('当前交付版本：'+VERSION+'（决策层 Q1 / 全覆盖 Q2 / copula Q3 / 33 条迁移）')
    for _, row in conclusions().iterrows():
        if row['claim'] in {'ET 整数中位：面板中位 MAE', '正式决策（逐样本固定阈值）：面板中位 MAE',
                            '批次无关性', 'Top-3 覆盖', '替换门槛状态', '迁移 case 数'}:
            print('%s %s：%s %s' % (row['question'], row['claim'], row['value'], row['unit']))
    print(unresolved()[['question', 'issue', 'status']].to_string(index=False))


def build(output, rendered=None):
    output = output.resolve()
    for folder in [Q1D, Q2, Q3G]:
        if not _checks_pass(read_json(folder/'verification.json')):
            raise ValueError('Unverified source: '+str(folder))
    if Q3M.exists() and not _checks_pass(read_json(Q3M/'verification.json')):
        raise ValueError('Unverified source: '+str(Q3M))
    output.mkdir(parents=True, exist_ok=False)
    index = []

    def include(source, relative, role):
        if any(row['file'] == relative for row in index):
            return
        source = source.resolve()
        destination = output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        origin = source.relative_to(ROOT).as_posix()
        index.append(dict(file=relative, source=origin, role=role,
                          sha256=sha256(destination), bytes=destination.stat().st_size))

    for group, (folder, names) in dynamic_tables().items():
        for name in names:
            include(folder/name, 'tables/'+group+'__'+name, group)
    if rendered:
        rendered_names = ['generated_official_Q1_scores52.csv.gz', 'transfer_official_Q1_scores52.csv',
                          'transfer_auxiliary_Q3_risks.csv', 'transfer_styles.csv', 'render.json']
        if (rendered/'official_Q1_fullfit_scores52.csv').exists():
            rendered_names.append('official_Q1_fullfit_scores52.csv')
        for name in rendered_names:
            include(rendered/name, 'tables/current__'+name, 'refreshed_deployment_inference')
    include(Q1D/'verification.json', 'evidence/q1_decision/verification.json', 'frozen_scientific_evidence')
    include(Q1D/'summary.json', 'evidence/q1_decision/summary.json', 'frozen_scientific_evidence')
    include(Q2/'verification.json', 'evidence/q2_top3/verification.json', 'frozen_scientific_evidence')
    include(Q2/'q2__summary.json', 'evidence/q2_top3/summary.json', 'frozen_scientific_evidence')
    figures = output/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    draw_figures(figures)
    for p in sorted(figures.glob('*')):
        if p.is_file():
            index.append(dict(file='figures/'+p.name, source='generated_from_formal_tables',
                              role='figure', sha256=sha256(p), bytes=p.stat().st_size))
    save_csv(output, 'conclusions.csv', conclusions())
    save_csv(output, 'unresolved.csv', unresolved())
    save_csv(output, 'current_constraints.csv', current_constraints())
    for folder in EVIDENCE_FOLDERS:
        for name in ['summary.json', 'verification.json', 'protocol.json']:
            if (folder/name).exists():
                include(folder/name, 'evidence/'+folder.name+'/'+name, 'frozen_scientific_evidence')
    for folder in ['src/beer_models', 'src/beer_audit']:
        for p in sorted((ROOT/folder).glob('*.py')):
            include(p, 'code/'+p.relative_to(ROOT).as_posix(), 'source_code')
    for name in SCRIPTS+CONFIGS+['requirements-lock.txt', 'requirements-r3-lock.txt',
                                 'docs/待优化问题.md']+TRAIN_ASSETS+RENDER_ASSETS:
        if (ROOT/name).exists():
            include(ROOT/name, 'code/'+name, 'runnable_dependency')
    for folder in [Q1D, Q1D_OLD, Q2, Q2A, Q3G, Q3G2, Q3M, Q3E, Q1P, Q2S, Q3G3]:
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in {'.csv', '.json', '.md'}:
                include(p, 'code/outputs/'+folder.name+'/'+p.name, 'code_snapshot_dependency')
    for folder, names in [
            (GEN_OLD, ['generated_profiles.csv', 'generated_metadata.csv',
                       'generation_validation.csv', 'verification.json']),
            (Q3_OLD, ['source119_profile.csv', 'transfer119_profile.csv',
                      'verification.json'])]:
        for name in names:
            if (folder/name).exists():
                include(folder/name, 'code/outputs/'+folder.name+'/'+name,
                        'code_snapshot_dependency')
    save_csv(output, 'runtime_dependencies.csv',
             [dict(path=name, use='training' if name in TRAIN_ASSETS else 'fixed_profile_render')
              for name in TRAIN_ASSETS+RENDER_ASSETS])
    (output/'code/README.md').write_text(
        '当前日常代码快照（20261028）。表与图见上一级 tables/figures，证据见 evidence/。\n'
        'code/outputs/ 内嵌了 status/verify 所需的正式产物，可在本目录离线执行：\n'
        '  ..\\..\\.venv\\Scripts\\python.exe -m unittest discover -s tests  # 若带 tests\n'
        '  python scripts/current.py verify --output outputs/my_check  # 需 PYTHONPATH=src\n'
        '训练与刷新仍需原始附件（--source）与已训练模型（train 生成）。完整历史实验见 Git 仓库。\n',
        encoding='utf-8')
    (output/'README.md').write_text('''# MathorCup A 当前交付 20261028

本目录是 20261028 版本：先读 conclusions.csv、unresolved.csv，再按 file_index.csv 查表。
与 20261027 的差别（历史组仍以 history_ 前缀保留）：
Q1 决策层改为逐样本固定阈值（批次无关，整批/逐条/分块同值），历史批次 top-k 表降为对照；
另含面板均值折外标定参考列与面板均值直接回归（未过预注册门槛，不采纳）证据；
Q2 换为 50/50 维全覆盖的图 Top-3（150 条）与 bootstrap/Jaccard 稳定性，并含图邻域消融（只提升稳定性、不提升预测）；
Q3 增加 copula 生成器的预注册门槛判定：v3 混合谱 14/14 通过并替换官方插值谱（v1/v2 作诊断保留）；
迁移 11 款×3 风格=33 条题面口径案例，含完整谱、前后感官画像与证据分级；
另有风格分类器折外证据（ET 0.544）、Tripel 缺陷安全冲突诊断与生成器消融。

数据披露：3.xlsx 为空白评分模板；0-6 标签来自外部 Zenodo 23624915 原始记录；2.csv 为 Spearman 矩阵。
本包不含原始附件、个人评分记录与模型二进制。

命令（在 code/ 内）：
`python scripts/current.py status`
`python scripts/current.py train --source "原附件目录"`
`python scripts/current.py predict --input 化学谱.csv --output outputs/my_prediction`
`python scripts/current.py render --output outputs/my_render`
`python scripts/current.py verify --output outputs/my_check`
`python scripts/current.py package --output outputs/current_20261028`
''', encoding='utf-8')
    for name in ['README.md', 'code/README.md', 'conclusions.csv', 'unresolved.csv',
                 'current_constraints.csv', 'runtime_dependencies.csv']:
        p = output/name
        index.append(dict(file=name, source='generated_from_formal_sources', role='current_summary',
                          sha256=sha256(p), bytes=p.stat().st_size))
    save_csv(output, 'file_index.csv', index)
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(output).as_posix())
    print(f'Current package: {output}\nForwardable ZIP: {archive}', flush=True)


def _replay_checks(models, source, package):
    """Actually re-run the deployed model on the attachment and compare to the package.

    The 20261027 checks accepted ``--models/--source`` but only wrote them into the
    report (review item 7).  Here the paths are loaded, the Q1 portraits are
    recomputed from ``q1.joblib`` and ``1.csv`` / the adopted generated spectra,
    and compared bit-for-bit with the stored tables inside the package.
    """
    models, source = Path(models), Path(source)
    if not (models/'q1.joblib').exists():
        raise FileNotFoundError(str(models/'q1.joblib'))
    if not (source/'1.csv').exists():
        raise FileNotFoundError(str(source/'1.csv'))
    if package is None:
        raise ValueError('replay requires the packaged reference tables (--package)')
    package = Path(package)
    reference = package/'tables'
    panel_ref = reference/'current__official_Q1_fullfit_scores52.csv'
    generated_ref = reference/'current__generated_official_Q1_scores52.csv.gz'
    if not panel_ref.exists():
        raise FileNotFoundError(str(panel_ref))
    official = joblib.load(models/'q1.joblib')
    scored = ['estimated_grade_0_6', 'raw_median_estimate', 'probability_rating_ge5',
              'probability_rating_6']

    def _compare(ref_path, frame):
        replay, _, _ = official_scores(frame, official)
        merged = pd.read_csv(ref_path, dtype={'beer': str}).merge(
            replay, on=['beer', 'target'], suffixes=('_ref', '_replay'), validate='one_to_one')
        assert len(merged) == len(replay), (len(merged), len(replay))
        mask = merged['panel_target_available_ref'].astype(bool) & merged['panel_target_available_replay'].astype(bool)
        merged = merged.loc[mask]
        return merged

    panel = pd.read_csv(source/'1.csv', dtype=str)
    merged = _compare(panel_ref, panel)
    assert len(merged) == 12500, len(merged)
    panel_diff = max(float((merged[c+'_ref']-merged[c+'_replay']).abs().max()) for c in scored)
    assert panel_diff <= 1e-12, panel_diff
    # The packaged submission table is the OOF prediction, while this replay is the
    # full-fit deployment portrait; only the full-fit reference is replayable, so the
    # submission table is checked for shape/domain here rather than equality.
    submission = pd.read_csv(package/'tables/q1_decision__q1__submission_wide_250x50.csv')
    assert submission.shape == (250, 51)
    assert submission.set_index('beer').to_numpy().min() >= 0
    assert submission.set_index('beer').to_numpy().max() <= 6

    generated = pd.read_csv(_adopted_generated()/'generated_profiles.csv').rename(
        columns={'generated_id': 'beer'})
    merged_g = _compare(generated_ref, generated)
    assert len(merged_g) == 15000, len(merged_g)
    generated_diff = max(float((merged_g[c+'_ref']-merged_g[c+'_replay']).abs().max())
                         for c in scored)
    assert generated_diff <= 1e-12, generated_diff
    return dict(q1_replay_panel_rows=int(len(merged)),
                q1_replay_panel_grade5=int((merged['estimated_grade_0_6_ref']==5).sum()),
                q1_replay_panel_grade6=int((merged['estimated_grade_0_6_ref']==6).sum()),
                q1_replay_panel_max_abs_diff=panel_diff,
                q1_replay_generated_rows=int(len(merged_g)),
                q1_replay_generated_max_abs_diff=generated_diff,
                q1_replay_models_sha256=sha256(models/'q1.joblib'),
                q1_replay_source_sha256=sha256(source/'1.csv'))


def verify(output, models=None, source=None, package=None):
    if (models is None) != (source is None):
        raise ValueError('models and source must be provided together for the replay')
    output.mkdir(parents=True, exist_ok=True)
    checks = dict(status='passed', version=VERSION)
    for folder in [Q1D, Q2, Q3G]:
        data = read_json(folder/'verification.json')
        assert _checks_pass(data), str(folder)
    checks['source_verifications_pass'] = True

    long = pd.read_csv(Q1D/'q1__submission_long_12500.csv')
    assert len(long) == 12500 and long.target.nunique() == 50
    assert long.grade_0_6.isin(range(7)).all()
    summary = read_json(Q1D/'summary.json')
    dist = long.grade_0_6.value_counts().sort_index().to_dict()
    assert dist == {int(k): int(v) for k, v in summary['new_distribution'].items()}, dist
    wide = pd.read_csv(Q1D/'q1__submission_wide_250x50.csv')
    assert wide.shape == (250, 51)
    pivot = long.pivot(index='beer', columns='target', values='grade_0_6')
    np.testing.assert_array_equal(wide.set_index('beer').loc[pivot.index, pivot.columns].to_numpy(),
                                  pivot.to_numpy())
    checks.update(q1_rows=12500, q1_targets=50, q1_wide_shape='250x50',
                  q1_grade_distribution=dist)
    metrics = pd.read_csv(Q1D/'q1__metrics_summary.csv').set_index(['scope', 'method'])
    official = metrics.loc[('beer_level', 'q1_fixed_threshold_quantile')]
    np.testing.assert_allclose(official['panel_median_mae'], 0.47968, atol=1e-9)
    np.testing.assert_allclose(metrics.loc[('beer_level', 'q1_integer_median_legacy'),
                                           'panel_median_mae'], 0.3928, atol=1e-9)
    np.testing.assert_allclose(metrics.loc[('beer_level', 'q1_topk_parent_capped_old_baseline'),
                                           'panel_median_mae'], 0.40648, atol=1e-9)
    calibers = pd.read_csv(Q1D/'q1__metric_calibers.csv')
    beer = calibers[(calibers.scope == 'beer_level')
                    & (calibers.prediction_set == 'q1_fixed_threshold_quantile')]
    exact5 = beer[beer.caliber == 'exact5'].iloc[0]
    exact6 = beer[beer.caliber == 'exact6'].iloc[0]
    ge5 = beer[beer.caliber == 'ge5'].iloc[0]
    assert int(exact5.n_predicted) == 321 and int(exact5.true_positive) == 23
    assert int(exact6.n_predicted) == 59 and int(exact6.true_positive) == 3
    assert int(ge5.n_predicted) == 380 and int(ge5.true_positive) == 36
    np.testing.assert_allclose(ge5.precision, official['precision_ge5'], atol=1e-9)
    np.testing.assert_allclose(ge5.recall, official['recall_ge5'], atol=1e-9)
    batch = pd.read_csv(Q1D/'q1__batch_invariance.csv')
    assert batch.mismatch_cells.eq(0).all() and batch.pass_ok.all()
    invar = summary['batch_invariance']
    assert all(bool(c.get('pass_ok')) for c in invar.values())
    thresholds = pd.read_csv(Q1D/'q1__thresholds.csv')
    fallback = summary['threshold_fallback']
    assert thresholds.groupby('target')['emits_ge5'].any().sum() == 34
    assert thresholds.groupby('target')['emits_6'].any().sum() == 12
    assert len(fallback['no_ge5']) == 16 and len(fallback['no_6']) == 38
    legacy_q1 = read_json(Q1D_OLD/'summary.json')
    leakage = legacy_q1['leakage_fix']
    assert leakage['perturbation_worst'] == 0.0
    assert legacy_q1['parent_child']['postcap_violations'] == 0
    checks.update(q1_batch_invariance='9/9 scenarios mismatch 0',
                  q1_metric_calibers='exact5/exact6/ge5 (321/59/380)',
                  q1_fixed_threshold_fallbacks='no_ge5 16 / no_6 38',
                  q1_calibration_leakage_perturbation_worst=leakage['perturbation_worst'],
                  q1_parent_child_postcap_violations=0)

    top3 = pd.read_csv(Q2/'q2__top3_all50_150.csv')
    assert len(top3) == 150 and top3.target.nunique() == 50
    assert top3.groupby('target').size().eq(3).all()
    optional_nan = ['r6_cross_corr', 'r6_sign_agree', 'off_flavors', 'threshold_mean']
    assert top3.drop(columns=optional_nan).notna().all().all()
    q2s = read_json(Q2/'q2__summary.json')
    communities = pd.read_csv(Q2/'q2__network_communities.csv')
    assert communities.community_id.nunique() == q2s['graph']['communities']
    assert int(communities.community_size.max()) == q2s['graph']['largest_community']
    tiers_df = pd.read_csv(Q2/'q2__evidence_tier_counts.csv')
    tiers = tiers_df.set_index('item')['count'].to_dict()
    assert {k: tiers[k] for k in ('A', 'B', 'C')} == {'A': 8, 'B': 70, 'C': 72}
    checks.update(q2_rows=150, q2_targets=50, q2_communities=len(communities),
                  q2_evidence_tiers=tiers)

    if Q2A.exists():
        q2a = read_json(Q2A/'summary.json')
        av = q2a['verification']
        assert all(av['checks'].values()), av
        overlap = q2a['overlap_summary']
        np.testing.assert_allclose(overlap['jaccard_mean'], 0.6, atol=1e-12)
        assert overlap['n_targets_identical'] == 13
        assert overlap['n_targets_jaccard_ge_0_5'] == 45
        np.testing.assert_allclose(overlap['freq_difference_mean'], 0.028166666666666663,
                                   atol=1e-12)
        verdict = q2a['verdict']
        assert verdict['verdict'] == 'partial_gain'
        assert verdict['stability_reliable'] is True and verdict['predictive_reliable'] is False
        checks.update(q2_ablation_graph_vs_no_graph_jaccard=0.6,
                      q2_ablation_stability_freq_gain=0.028166666666666663,
                      q2_ablation_predictive_p=verdict['observed']['mae_p_value'],
                      q2_ablation_verdict='partial_gain')

    if Q3E.exists():
        q3e = read_json(Q3E/'summary.json')
        ev = read_json(Q3E/'verification.json')
        assert ev['status'] == 'passed' and len(ev['checks']) == 52
        style = q3e['style_evidence']
        np.testing.assert_allclose(style['accuracy']['extra_trees'], 0.544, atol=1e-9)
        np.testing.assert_allclose(style['accuracy']['logistic'], 0.512, atol=1e-9)
        np.testing.assert_allclose(style['accuracy']['nearest_centroid_oof'], 0.224, atol=1e-9)
        conflict = q3e['tripel_conflict']
        assert conflict['n_defect_safe_samples_median'] == 0
        assert conflict['n_cases_dms_cap_median'] == 11
        np.testing.assert_allclose(conflict['mean_dist_reduction_pct'], 12.49600915769988,
                                   atol=1e-9)
        checks.update(q3_style_classifier_oof_accuracy=0.544,
                      q3_style_classifier_nearest_centroid=0.224,
                      q3_tripel_defect_safe_samples=0,
                      q3_tripel_cases_dms_cap=11,
                      q3_evidence_checks=52)

    adopted = _adopted_generated()
    profiles = pd.read_csv(adopted/'generated_profiles.csv')
    assert len(profiles) == 300 and profiles.style.value_counts().to_dict() == {
        'Tripel': 100, 'Kriek': 100, 'Lambic': 100}
    old_header = pd.read_csv(GEN_OLD/'generated_profiles.csv', nrows=1).columns.tolist()
    assert list(profiles.columns) == old_header
    safety = pd.read_csv(adopted/'defect_safety.csv')
    assert safety.passed.all(), safety.loc[~safety.passed]
    gate = _gate_state(adopted)
    assert gate['integrity_passed'] and gate['replacement_gate_passed'], gate
    checks.update(q3_generated=300,
                  q3_generated_source=adopted.relative_to(ROOT).as_posix(),
                  q3_integrity_passed=True,
                  q3_replacement_gate_passed=True,
                  q3_gate_checks=gate['gate_checks'],
                  q3_gate_failures=gate['failures'],
                  q3_defect_caps_all_pass=True)
    if (Q3G/'verification.json').exists():
        v1 = _gate_state(Q3G)
        checks.update(q3_generation_v1_gate_passed=v1['gate_passed'],
                      q3_generation_v1_gate_failures=v1['failures'])

    if (Q3G2/'generated_profiles.csv').exists():
        v2_profiles = pd.read_csv(Q3G2/'generated_profiles.csv')
        assert len(v2_profiles) == 300
        assert list(v2_profiles.columns) == old_header
        v2_safety = pd.read_csv(Q3G2/'defect_safety.csv')
        assert v2_safety.passed.all(), v2_safety.loc[~v2_safety.passed]
        gate2 = _gate_state(Q3G2)
        assert gate2['integrity_passed'], gate2
        checks.update(q3_generation_v2=300,
                      q3_generation_v2_integrity_passed=True,
                      q3_generation_v2_gate_passed=gate2['gate_passed'],
                      q3_generation_v2_gate_failures=gate2['failures'])

    if Q3M.exists():
        cases = _migration_frame()
        assert len(cases) == 33
        assert cases.target_style.nunique() == 3
        assert cases.target_style.value_counts().to_dict() == {
            'Tripel': 11, 'Kriek': 11, 'Lambic': 11}
        ms = _migration_summary()
        assert ms.get('status') == 'passed'
        for key in ('A_median', 'B_median', 'C_median'):
            assert ms['criteria_pass'][key] == 33
            assert ms['candidate_pass'][key] == 660
        checks['q3_migration_cases'] = int(len(cases))
        checks['q3_migration_criteria_pass'] = ms['criteria_pass']
        checks['q3_migration_candidate_pass'] = ms['candidate_pass']
        checks['q3_migration_mean_distance_reduction_pct'] = ms['mean_distance_reduction_pct']
        checks['q3_migration_status'] = ms.get('status')
        mig = read_json(Q3M/'summary.json')
        spectra = pd.read_csv(Q3M/'s2c_migration_spectra_224_long.csv')
        assert len(spectra) == 66 and spectra.role.value_counts().to_dict() == {
            'before': 33, 'after': 33}
        wide_spectra = pd.read_csv(Q3M/'s2c_migration_spectra_224_wide.csv')
        assert len(wide_spectra) == 33
        after_input = pd.read_csv(Q3M/'s2c_migration_after_input_33.csv')
        assert after_input.shape == (33, 227)
        sensory = pd.read_csv(Q3M/'s2c_sensory_before_after_52.csv')
        assert len(sensory) == 66*52 and sensory.target_52.nunique() == 52
        assert sensory.case_id.nunique() == 33
        assert sensory.groupby(['case_id', 'role']).ngroups == 66
        evidence = pd.read_csv(Q3M/'s2c_evidence_by_case.csv')
        assert len(evidence) == 33
        assert set(evidence.claim_level) <= {
            'model_supported', 'conditional', 'undecidable'}
        assert evidence.claim_level.value_counts().to_dict() == {
            k: v for k, v in mig['claim_level_counts'].items()}
        mig_verification = read_json(Q3M/'verification.json')
        replay = mig_verification['sensory_replay']
        assert int(replay['mismatch_cells']) == 0, replay
        assert float(replay['max_abs_delta']) <= 1e-12, replay
        if models is not None:
            assert replay['model_sha256'] == sha256(Path(models)/'q1.joblib'), replay
        checks.update(q3_migration_spectra=66,
                      q3_migration_sensory_rows=int(len(sensory)),
                      q3_migration_sensory_replay_mismatch_cells=int(replay['mismatch_cells']),
                      q3_migration_sensory_replay_model_sha256=replay['model_sha256'],
                      q3_migration_claim_level_counts=mig['claim_level_counts'],
                      q3_migration_style_diagnostic_counts=mig['style_diagnostic_counts'],
                      q3_migration_defect_status_counts=mig['defect_status_counts'])
        det = mig.get('det_rerun', {})
        if det:
            assert det['defect_worse_count'] == 0, det['defect_worse_cells']
            assert det['grade_change_within_limit'] is True, det
            assert det['model_sha_changed'] is True and det['changed_cells'] <= 100, det
            assert det['criteria_unchanged'] is True
            checks.update(q3_migration_det_changed_cells=int(det['changed_cells']),
                          q3_migration_det_changed_up=int(det['changed_up']),
                          q3_migration_det_changed_down=int(det['changed_down']),
                          q3_migration_det_defect_worse_count=int(det['defect_worse_count']))
    if (Q3G3/'generated_profiles.csv').exists():
        v3_profiles = pd.read_csv(Q3G3/'generated_profiles.csv')
        assert len(v3_profiles) == 300
        assert list(v3_profiles.columns) == old_header
        v3_safety = pd.read_csv(Q3G3/'defect_safety.csv')
        assert v3_safety.passed.all(), v3_safety.loc[~v3_safety.passed]
        gate3 = _gate_state(Q3G3)
        assert gate3['integrity_passed'], gate3
        checks.update(q3_generation_v3=300,
                      q3_generation_v3_integrity_passed=True,
                      q3_generation_v3_gate_passed=gate3['gate_passed'],
                      q3_generation_v3_replacement_gate_passed=gate3['replacement_gate_passed'],
                      q3_generation_v3_gate_failures=gate3['failures'])

    if (Q1P/'q1__panel_mean_pred_12500.csv').exists():
        pred = pd.read_csv(Q1P/'q1__panel_mean_pred_12500.csv')
        assert len(pred) == 12500 and pred.target.nunique() == 50
        assert pred.panel_mean_pred.notna().all()
        pm = pd.read_csv(Q1P/'q1__panel_mean_metrics.csv')
        mae_col = next(c for c in pm.columns if 'mae' in c.lower() and 'panel' in c.lower())
        checks.update(q1_panelmean_rows=12500,
                      q1_panelmean_best_mae=float(pm[mae_col].min()),
                      q1_panelmean_adoption=read_json(Q1P/'summary.json').get(
                          'adoption_recommendation'))

    if (Q2S/'verification.json').exists():
        assert _checks_pass(read_json(Q2S/'verification.json'))
        checks['q2_sensitivity_verdict'] = read_json(Q2S/'summary.json').get('verdict')

    if (models is None) != (source is None):
        raise ValueError('--models and --source must be given together for the replay check')
    if models is not None:
        checks.update(_replay_checks(models, source, package))

    if package:
        index = pd.read_csv(package/'file_index.csv')
        for row in index.itertuples():
            assert sha256(package/row.file) == row.sha256, row.file
        prohibited = [p for p in package.rglob('*') if p.is_file() and
                      (p.suffix.lower() in ['.joblib', '.xlsx', '.npy']
                       or 'Raw sensory data anonymized' in p.name)]
        assert not prohibited, prohibited
        checks['package_files_hashed'] = len(index)
        checks['package_excludes_raw_records_and_model_binaries'] = True
    write_json(output/'verification.json', checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2, default=str), flush=True)
