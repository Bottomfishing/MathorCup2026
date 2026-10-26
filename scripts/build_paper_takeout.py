"""Package verified results for teammates; this does not write a competition paper."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd


TABLES = {
    'scope': ['scorecard_scope.csv', 'defect_scope.csv'],
    'raw_sensory': ['target_coverage52.csv'],
    'q1': ['point_comparison.csv', 'point_uncertainty.csv', 'defect_risk_validation.csv',
           'risk_metrics.csv', 'oof_scores_all52.csv', 'full_fit_scores_all52.csv'],
    'q2_standardized': ['top3_panel_all50.csv', 'top3_all52_slots.csv'],
    'q2_literature': ['top3_all50_literature.csv', 'source_catalog.csv'],
    'q2_raw': ['predictive_metrics.csv', 'top3_raw_all50.csv', 'top3_raw_all52_slots.csv',
               'rank_stability.csv', 'historical_rank_comparison.csv'],
    'q3': ['generated_profiles.csv', 'generated_metadata.csv', 'sensory_fingerprints52.csv',
           'generation_validation.csv', 'generation_replacement_checks.csv',
           'migration_constraint_attrition.csv', 'transfer_results.csv',
           'original_fixed_path_five_defect_checks.csv', 'fixed_path_joint_refit_comparison.csv'],
    'validation': ['q1_metrics_by_repeat.csv', 'q1_repeat_summary.csv', 'risk_repeat_summary.csv',
                   'paired_beer_averaged_intervals.csv', 'q2_rank_coverage.csv', 'q2_rank_stability.csv',
                   'generation_repeat_summary.csv', 'fixed_historical_path_joint.csv'],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_records(frame):
    return json.loads(frame.to_json(orient='values', force_ascii=False))


def build(run, output):
    run, output = run.resolve(), output.resolve()
    checks = {}
    for name in ('raw_sensory', 'r2', 'q1', 'q2_standardized', 'q2_literature', 'q2_raw', 'q3', 'validation'):
        checks[name] = json.loads((run/name/'verification.json').read_text(encoding='utf-8'))
        if checks[name]['status'] != 'passed':
            raise ValueError('Unverified source: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    tables, figures = output/'tables', output/'figures'
    tables.mkdir(); figures.mkdir()
    index = []
    def include(source, destination, stage, kind):
        shutil.copy2(source, destination)
        entry = dict(file=destination.relative_to(output).as_posix(),
                     source=source.relative_to(run).as_posix(), stage=stage, kind=kind,
                     sha256=digest(destination), bytes=destination.stat().st_size)
        if kind == 'table':
            df = pd.read_csv(source)
            entry.update(rows=len(df), columns=len(df.columns))
        index.append(entry)
    # File names are explicitly selected; missing deliverables must fail visibly.
    for stage, names in TABLES.items():
        for name in names:
            include(run/stage/name, tables/(stage+'__'+name), stage, 'table')
    for stage in ('q1', 'q2_raw', 'q3', 'validation'):
        for source in sorted((run/stage).iterdir()):
            if source.suffix.lower() in ('.png', '.svg'):
                include(source, figures/(stage+'__'+source.name), stage, 'figure')
    def read(stage, name):
        return pd.read_csv(run/stage/name, float_precision='round_trip')
    val = json.loads((run/'validation/summary.json').read_text(encoding='utf-8'))
    raw = json.loads((run/'raw_sensory/summary.json').read_text(encoding='utf-8'))
    r3 = json.loads((run/'q2_raw/summary.json').read_text(encoding='utf-8'))
    means = read('validation', 'q1_repeat_summary.csv')
    et = means[(means.method == 'extra_trees') & (means.metric == 'raw_rating_mae')].iloc[0]
    q3 = read('q3', 'transfer_results.csv')
    path_sensitivity = read('validation', 'fixed_historical_path_joint.csv')
    claims = [
        ['数据', '唯一啤酒数', val['unique_beers'], '款', 'validation/summary.json', '重复验证的750次预测仍来自250款'],
        ['数据', '原始品评记录数', raw['records'], '条', 'raw_sensory/summary.json', '同一啤酒全部记录分在同一折'],
        ['Q1', 'ET重复验证MAE均值', float(et['mean']), '原始0～6分', 'validation/q1_repeat_summary.csv', '按啤酒和品评员等权；50个有标签目标'],
        ['Q1', 'ET重复间样本标准差', float(et['sample_sd']), '原始0～6分', 'validation/q1_repeat_summary.csv', '3次重复的样本SD，不是置信区间'],
        ['Q1', '两项乙醛标签数', 0, '项', 'scope/scorecard_scope.csv', 'A_acetaldehyde、F_acetaldehyde分数和概率留空'],
        ['Q2', 'R3 GBR非零因素排名数', 51, '条', 'q2_raw/top3_raw_all50.csv', '17目标；99个零归因空位及6个无标签空位不补排名'],
        ['Q2', 'R3 GBR有历史预测支持目标数', r3['supported_targets']['gbr_selected'], '项', 'q2_raw/summary.json', '预测支持及SHAP不等于因果或迁移控制证据'],
        ['Q3', '主生成谱数', len(read('q3', 'generated_profiles.csv')), '条', 'q3/generated_profiles.csv', '三种风格插值；覆盖、方差收缩和近重复限制保留'],
        ['Q3', '当前完整筛选通过源啤酒数', int(q3.selected.sum()), '款', 'q3/transfer_results.csv', '8293个历史合格路径点；77组权重；当前0个完整候选'],
        ['敏感性', '历史固定路径仅感官共同通过数', int(path_sensitivity.all_five_sensory_guards_pass.sum()), '个/33上下文', 'validation/fixed_historical_path_joint.csv', '仅固定路径感官敏感性；不得改写Q3当前0完整候选'],
    ]
    # Tail recall is taken from the repeated-validation table, not inferred from MAE.
    metrics = read('validation', 'q1_metrics_by_repeat.csv')
    et_metrics = metrics[metrics.method == 'extra_trees']
    for rating in (5, 6):
        values = et_metrics[f'raw_recall_{rating}']
        assert values.nunique() == 1
        claims.append(['Q1', f'个人{rating}分精确召回', float(values.iloc[0]), '比例',
                       'validation/q1_metrics_by_repeat.csv', '所有3次重复均为0；不能称高分预测已解决'])
    ledger = pd.DataFrame(claims, columns=['question', 'claim', 'value', 'unit', 'source', 'interpretation'])
    ledger.to_csv(output/'conclusions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(index).to_csv(output/'file_index.csv', index=False, encoding='utf-8-sig')
    # A compact, static reader view; complete machine-readable results remain CSV.
    sheets = []
    def sheet(name, title, df, widths, note):
        sheets.append(dict(name=name, title=title, headers=list(df.columns), rows=frame_records(df), widths=widths, note=note))
    sheet('结论与边界', 'MathorCup A｜最终取数概览', ledger[['claim','value','unit','interpretation']],
          [34, 19, 21, 78], '数值来自已核验结果；来源逐项见 conclusions.csv。')
    sheet('Q1重复评分', 'Q1｜原始0～6分 · 三次重复',
          metrics[['repeat','method','raw_rating_mae','raw_recall_5','raw_recall_6']],
          [12, 25, 22, 22, 22], 'MAE与个人5/6分精确召回不同；SD只在完整汇总CSV中报告。')
    intervals = read('validation', 'paired_beer_averaged_intervals.csv')
    risk = intervals[intervals.metric != 'raw_MAE_gain'][['metric','method','target','mean_gain','lower','upper']]
    sheet('缺陷概率区间', '缺陷概率｜相对训练先验的Brier改善', risk,
          [25, 24, 24, 18, 18, 18], '95%固定OOF配对bootstrap区间；先按250款啤酒平均重复损失；不含完整重拟合不确定性。')
    ranks = read('q2_standardized','top3_panel_all50.csv')
    cols = ['target','rank','field','chemical_name','bootstrap_frequency','target_passes_BY_0_05']
    sheet('Q2标准化主表', 'Q2｜历史标准化响应关联 · 150条', ranks[cols],
          [25, 10, 40, 45, 24, 28][:len(cols)], '响应是历史标准化值；与原始分GBR分别引用。全部统计证据及文献见CSV，不作因果解释。')
    raw_ranks = read('q2_raw','top3_raw_all50.csv')
    raw_ranks = raw_ranks[raw_ranks.method == 'gbr_selected']
    cols = ['target','rank','field','ranking_score','predictive_support_status']
    sheet('Q2原始GBR', 'Q2｜原始量尺GBR · 51条非零SHAP排名', raw_ranks[cols],
          [25, 10, 40, 22, 53], '17目标有非零归因；其余零归因与无标签空位见all52台账；SHAP不是化合物因果作用。')
    gen = read('validation','generation_repeat_summary.csv')
    gen = gen[gen.metric.isin(['energy','variance_ratio','rank_correlation_rmse','heldout_coverage','internal_near_duplicate_rate'])]
    sheet('Q3生成诊断', 'Q3｜重复验证的生成诊断', gen[['method','metric','mean','sample_sd']],
          [25, 39, 20, 20], 'Copula未通过训练内冻结替换条件；主方案仍为插值。sample_sd是3次重复的样本标准差。')
    sheet('Q3迁移检查', 'Q3｜完整检查 · 0个当前通过方案',
          q3[['beer','n_historically_eligible','n_pass_current','selected','actual_after_measured']],
          [12, 29, 22, 17, 29], '历史11款逐一报告；任务5的1/33仅为固定路径感官敏感性，未完成新迁移候选验证。')
    (output/'workbook_data.json').write_text(json.dumps(dict(sheets=sheets),ensure_ascii=False,indent=2,allow_nan=False)+'\n', encoding='utf-8')
    readme = '''# 最终论文取数包

本文件夹供队友核对数值、选用表格及引用图，不包含论文正文。

先读 `conclusions.csv`，再按 `file_index.csv` 查完整CSV和PNG/SVG。
`workbook_data.json`保存Excel静态查看版的结构化输入；CSV是完整科学结果。
Excel若随交付提供，只展示这些已核验数值，不重新训练或计算机器学习结果。

## 使用口径

- 数据：250款啤酒，226种测量化学属性加5种派生字段共231个输入；原始模型排除7个未解决汇总字段，使用224列。2197条记录、16名品评员，50项有原始标签；评分卡52项，两个乙醛项目留空。
- Q1：当前为原始0～6分整数条件中位预测，按啤酒/品评员等权。完整拟合值用于部署展示，性能必须引用折外结果。三次重复ET MAE均值约0.794561、样本SD约0.000682；个人5/6分精确召回均为0。
- 缺陷：四项非零概率相对先验有历史支持；Stale区间跨0。校准前后都保留，不能把概率当安全认证。
- Q2：历史标准化响应的Ridge主表150条与原始分GBR 51条分别列出；原始均值Ridge是另一统计泛函。不得混成156个已识别因素。零归因不排序；关联、SHAP、同源原论文和外部化学依据均不自动构成数据集内因果证据。
- Q3：三种风格共300条插值谱；收缩Copula未通过冻结替换门槛。模型评分画像不是新酿造实测。8293个历史合格路径点、77组权重及原11路径的当前完整筛选均无通过方案；这不证明真实酿造无解。
- 重复验证：3个固定种子×3外折/2内折。750次折外预测仍只有250款独立啤酒单位。重复间SD不是CI；配对区间先按啤酒平均重复损失，仅为固定OOF损失bootstrap，不覆盖完整重训练或新数据不确定性。
- 历史固定路径：任务5仅感官敏感性中1/33上下文通过，未重新做范围、风格或候选搜索；不得将其写成当前完整迁移候选。

## 文件与图

`tables/`：按阶段前缀区分，保留原字段与数值。`figures/`：沿用各阶段已核验数据绘图，PNG便于预览、SVG便于排版。图源CSV在索引中按同一阶段查找；图中旧阶段数值不得替代重复验证汇总。

`source_checks.json`保留各阶段核验范围；复现总报告及日志在本包上一级。Q2历史200次重训/1999次置换、文献标注和历史迁移池按版本复用；这次重建不宣称重新做过所有历史实验。

原始附件、公开评分CSV/工作簿及模型二进制不打入本包；代码、锁定依赖及重建命令见仓库和任务6说明。
'''
    (output/'README.md').write_text(readme, encoding='utf-8')
    (output/'source_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Paper takeout: {len(index)} indexed files; {len(sheets)} workbook views; {len(claims)} claims.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    build(a.run, a.output)
