"""One current result selection, compact teammate export and numerical checks."""
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import joblib
import numpy as np
import pandas as pd
from beer_audit.audit import sha256, save_csv, write_json
from .current_runtime import ROOT, Q3, DEFECTS, read_json, auxiliary_defects
from .raw_panel import portrait

VERSION = '20261027'
Q1 = ROOT/'outputs/q1_selective_tail_20261026'
TARGET = ROOT/'outputs/q1_target_20261026'
GEN = ROOT/'outputs/task4_raw_q3_20261026'

# Deliberate selection, not a recursive copy of experimental output directories.
TABLES = {
    'q1': (Q1, ['estimated_grades_all52.csv', 'estimated_grades_primary47.csv',
        'full_fit_grades_all52.csv', 'comparison_summary.csv', 'metrics_by_repeat.csv',
        'metrics_by_target.csv', 'metrics_by_observed_grade.csv', 'confusion.csv',
        'panel_confusion.csv', 'tail_precision.csv']),
    'q1_diagnosis': (TARGET, ['label_distribution.csv', 'panel_disagreement.csv', 'observed_panel_oracle.csv']),
    'q1_risk': (ROOT/'outputs/r2b_tail_risk_20261026', ['defect_risk_validation.csv']),
    'q1_repeated': (ROOT/'outputs/task5_repeated_nested_20261026',
        ['q1_repeat_summary.csv', 'paired_beer_averaged_intervals.csv', 'generation_repeat_summary.csv']),
    'q1_independent': (ROOT/'experiments/q1_two_stage_20261026',
        ['comparison.csv', 'comparison_by_repeat.csv', 'six_by_target.csv', 'tail_gain_intervals.csv']),
    'q2_standardized': (ROOT/'outputs/q2_task4_20261025',
        ['top3_panel_all50.csv', 'top3_all52_slots.csv', 'network_nodes.csv', 'network_edges_0_6.csv']),
    'q2_literature': (ROOT/'outputs/task3_q2_literature_20261025', ['top3_all50_literature.csv', 'source_catalog.csv']),
    'q2_raw': (ROOT/'outputs/r3_raw_explanation_20261026',
        ['predictive_metrics.csv', 'top3_raw_all50.csv', 'top3_raw_all52_slots.csv', 'rank_stability.csv']),
    'q3_generation': (GEN, ['generated_profiles.csv', 'generated_metadata.csv',
        'generation_validation.csv', 'generation_replacement_checks.csv', 'generation_profile_diagnostics.csv']),
    'q3_transfer': (Q3, ['current_plan_status.csv', 'source119_profile.csv', 'transfer119_profile.csv',
        'official_Q1_missing_scores52.csv.gz', 'missing_aware_defect_before_after.csv', 'nominal_styles.csv',
        'matched_plan_comparison.csv', 'paired_intervals.csv', 'style_accuracy.csv',
        'transfer_checks.csv', 'parameter_checks.csv']),
    'q3_stale': (ROOT/'outputs/q3_stale_risk_20261027', ['metrics.csv', 'paired_intervals.csv']),
    'historical_coordinate_evidence': (ROOT/'outputs/q3_stage2_20261026', ['source_defect_evidence.csv']),
}

TRAIN_ASSETS = [
    'configs/r2_raw_panel_protocol_20261026.json', 'configs/r2b_tail_risk_protocol_20261026.json',
    'configs/q3_style_protocol.json', 'data/reference/zenodo23624915/download_manifest.json',
    'data/current/risk_full_training.csv.gz', 'data/current/risk_provenance.json',
    'outputs/p0_audit_20261023_final/data_inventory.csv',
    'outputs/p0_audit_20261023_final/sensory_mapping.csv',
    'outputs/raw_sensory_audit_20261025/beer_identity_mapping.csv',
    'outputs/task1_acceptance_20261025/scorecard_scope.csv',
    'outputs/r2b_tail_risk_20261026/defect_risk_validation.csv',
]
RENDER_ASSETS = ['outputs/task4_raw_q3_20261026/generated_profiles.csv',
    'outputs/q3_missing_aware_20261027/source119_profile.csv',
    'outputs/q3_missing_aware_20261027/transfer119_profile.csv']
EVIDENCE_FOLDERS = [Q1, TARGET, GEN, Q3, ROOT/'outputs/r3_raw_explanation_20261026',
    ROOT/'outputs/task5_repeated_nested_20261026', ROOT/'outputs/q3_stale_risk_20261027']
EVIDENCE_CHECKS = {folder: (Q3 if folder.name == 'q3_stale_risk_20261027' else folder)/'verification.json'
                   for folder in EVIDENCE_FOLDERS}


def unresolved():
    return pd.DataFrame([
        ['Q1', '个人5/6分识别', '未解决，用户暂缓', '正式模型两等级精确召回均0；独立候选未采用', '保留正式中位预测，后续单独优化'],
        ['Q1', '两项乙醛缺标签', '未解决', '52项评分卡仅50项有原始标签', '分数和概率留空，不造标签'],
        ['Q3', 'Stale预测支持', '未解决', '专用流程Brier改善0.000357，99%区间跨0', '保留未知，不用于真实不恶化认证'],
        ['Q3', '完整五缺陷迁移', '未解决', '119号仅四非零风险代理条件案例；其余10款受限', '不声称所有源样本迁移成功'],
        ['Q3', '参数与重采样稳定性', '部分解决', '119号18/20和9/12共同通过；附加扰动仅22/32', '不推广为全参数情景可靠'],
        ['Q3', '源119真实双乙酰', '数值仍缺失', '已消除预测填0依赖，实际浓度未恢复', '两端保留NA，要求保持实际未知值'],
        ['Q3', '生成多样性与工艺可行性', '未解决', '300条插值谱有方差收缩和近重复；未真实酿造', '模型画像不当实测或可酿造证明'],
        ['Q2', '因果解释与排名覆盖', '有限', '标准化关联150条；原始GBR仅17目标51条非零排名', '三种统计口径分开，零归因与缺标签不补位'],
    ], columns=['question', 'issue', 'status', 'evidence', 'current_handling'])


def status():
    print('当前交付版本：'+VERSION+'；第三阶段代码与交付收敛')
    print('Q1：正式ET整数条件中位数；高分候选未采用。')
    print('Q2：标准化关联、原始均值Ridge、原始分GBR分别引用。')
    print('Q3：300条插值谱；119号ET223四代理条件方案；Stale及完整五缺陷未解决。')
    print(unresolved()[['question', 'issue', 'status']].to_string(index=False))


def conclusions():
    metrics = pd.read_csv(Q1/'metrics_by_repeat.csv').query("method == 'shared_et'")
    s = read_json(Q3/'current_plan_status.json')
    return pd.DataFrame([
        ['data', '独立啤酒数', 250, '款', '750次重复预测仍来自250款，非外部测试'],
        ['data', '原始品评记录数', 2197, '条', '同一啤酒全部记录分组，啤酒与品评员等权'],
        ['Q1', '正式ET重复验证MAE', metrics.raw_rating_mae.mean(), '原始0～6分', '平均个人评分误差，不等于高分识别'],
        ['Q1', '个人5分精确召回', metrics.raw_recall_5.mean(), '比例', '仍未解决'],
        ['Q1', '个人6分精确召回', metrics.raw_recall_6.mean(), '比例', '仍未解决'],
        ['Q1', '无标签评分卡项', 2, '项', '两项乙醛留空'],
        ['Q2', '原始GBR非零SHAP排名', 51, '条', '17目标；与标准化关联和原始均值Ridge分开'],
        ['Q3', '三风格生成谱', 300, '条', '插值；模型评分不是实测'],
        ['Q3', '119号可用辅助输入', 223, '列', '不使用缺失双乙酰；化学保持约束仍保留'],
        ['Q3', '119号改动坐标', s['changed_coordinates'], '列', '数学目标谱，工艺未认证'],
        ['Q3', '119号配对共同通过', s['validation_joint_fraction'], '比例，9/12', '同250款开发敏感性，非新独立验证'],
        ['Q3', '119号开发参数扰动通过', s['development_parameter_fraction'], '比例，32/32', '冻结采用定义'],
        ['Q3', '119号附加对照扰动通过', s['validation_parameter_fraction_additional_diagnostic'], '比例，22/32', '未达80%，仍有限制'],
        ['Q3', '完整五缺陷真实效果已证明', False, '布尔', 'Stale不足，无迁移后实测'],
    ], columns=['question', 'claim', 'value', 'unit', 'interpretation'])


def current_constraints():
    return pd.DataFrame([
        ['question', 'Pils/Lager迁移至Tripel，按风格生成谱及画像', '题意要求；画像是模型估计'],
        ['question', '指定超阈值缺陷不恶化', '题意要求；不等同所有概率切点不得升高'],
        ['data_implementation', '已知DMS/4VG/t2n坐标不增加；缺失双乙酰保持源实际未知值', '原坐标不自动当物理浓度，双乙酰两端NA；真实工艺未认证'],
        ['data_implementation', '223输入辅助模型四项未校准P(评分≥1)不增加', '预测关联代理，不是阈值超量或真实安全；与正式Q1分列'],
        ['interface_consistency', '五项辅助中位评分不增加', 'Stale及近零整数不构成安全证据'],
        ['team_extra', '双风格判别、目标距离至少改善10%、经验支持与改动≤144列', '团队建模条件，119号固定改动130列；不是题面定理或独立生产旋钮'],
        ['team_extra', '预先固定重采样联合条件及开发参数扰动门槛', '同250款开发敏感性：18/20、9/12；32/32，附加对照22/32不足'],
        ['team_sensitivity', '全部五缺陷、校准前后各三切点，共30概率条件', '历史附加保守情景，不冒充题面；未继承旧网格为新223模型结论'],
        ['unresolved', 'Stale、真实双乙酰、完整五缺陷、酿造效果', '保持未知；较高等级事件仍可能增加，不能写全部严重度改善'],
    ], columns=['kind', 'condition', 'interpretation'])


def draw_generation(output):
    """Replot retained generation evidence without the obsolete migration panel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = pd.read_csv(GEN/'generation_validation.csv')
    styles = ['Tripel', 'Kriek', 'Lambic']
    means = data.groupby(['method', 'style']).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for offset, method, label, color in [(-.18, 'interpolate', 'Retained interpolation', '#287a9b'),
        (.18, 'shrinkage_copula', 'Unadopted copula', '#bf8548')]:
        for ax, metric in zip(axes, ['variance_ratio', 'energy']):
            values = [means.loc[(method, style), metric] for style in styles]
            ax.bar(np.arange(3)+offset, values, width=.36, label=label, color=color)
            ax.set(xticks=range(3), xticklabels=styles)
    axes[0].axhline(1, color='gray', ls=':')
    axes[0].set(ylabel='Within-style variance ratio', title='Variance shrinkage remains')
    axes[1].set(ylabel='Held-out energy distance (lower is better)', title='Generation evidence only; no brewing test')
    axes[1].legend(fontsize=8, loc='upper right')
    for ext in ['.png', '.svg']:
        fig.savefig(output/('Q3_生成诊断'+ext), dpi=170)
    plt.close(fig)
    svg = output/'Q3_生成诊断.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')


def build(output, rendered=None):
    output = output.resolve()
    # Check selected sources before creating the export; never modify historical runs.
    for folder, check in EVIDENCE_CHECKS.items():
        if read_json(check)['status'] != 'passed':
            raise ValueError('Unverified source: '+str(folder))
    output.mkdir(parents=True, exist_ok=False)
    index = []
    def include(source, relative, role):
        if any(row['file'] == relative for row in index):
            return
        source = source.resolve()
        destination = output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        origin = source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source)
        index.append(dict(file=relative, source=origin, role=role,
            sha256=sha256(destination), bytes=destination.stat().st_size))
    for group, (folder, names) in TABLES.items():
        for name in names:
            include(folder/name, 'tables/'+group+'__'+name, group)
    if rendered:
        for name in ['generated_official_Q1_scores52.csv.gz', 'transfer_official_Q1_scores52.csv',
                     'transfer_auxiliary_Q3_risks.csv', 'transfer_styles.csv', 'render.json']:
            include(rendered/name, 'tables/current__'+name, 'refreshed_deployment_inference')
    else:
        include(ROOT/'outputs/q3_stage2_20261026/generated_sensory52.csv.gz',
                'tables/current__generated_official_Q1_scores52.csv.gz', 'reused_same_official_Q1_inference')
        # Keep the same daily filenames when exporting directly from verified tables.
        stored = pd.read_csv(Q3/'official_Q1_missing_scores52.csv.gz')
        save_csv(output/'tables', 'current__transfer_official_Q1_scores52.csv', stored)
        defect = pd.read_csv(Q3/'missing_aware_defect_before_after.csv')
        nominal = pd.read_csv(Q3/'nominal_styles.csv')
        risks, styles = [], []
        for suffix, beer in [('before','119'), ('after','transfer119')]:
            for row in defect.to_dict('records'):
                risks.append(dict(beer=beer, target=row['target'], auxiliary_grade=row['grade_'+suffix],
                    probability_ge1=row['ge1_'+suffix], probability_ge5=row['ge5_'+suffix],
                    probability_6=row['six_'+suffix], nonzero_validation_supported=row['nonzero_validation_supported'],
                    tail_validation_established=False, probability_source=row['risk_source'], safety_certified=False))
            for row in nominal.to_dict('records'):
                styles.append(dict(beer=beer, classifier=row['model'],
                    predicted_style=row['source_prediction' if suffix=='before' else 'after_prediction'],
                    Tripel_probability=row['Tripel_'+suffix], calibration_established=False))
        save_csv(output/'tables', 'current__transfer_auxiliary_Q3_risks.csv', risks)
        save_csv(output/'tables', 'current__transfer_styles.csv', styles)
        write_json(output/'tables/current__render.json', dict(status='reused_verified_predictions',
            training_required=False, search_required=False, generated_profiles=300, transfer_profiles=2,
            actual_after_measured=False))
        for name in ['transfer_official_Q1_scores52.csv','transfer_auxiliary_Q3_risks.csv','transfer_styles.csv','render.json']:
            relative = 'tables/current__'+name
            path = output/relative
            index.append(dict(file=relative, source='outputs/q3_missing_aware_20261027',
                role='reused_verified_deployment_inference', sha256=sha256(path), bytes=path.stat().st_size))
    for folder, filename, label in [(Q1, 'q1_before_after', 'Q1_未采用候选对照'),
        (Q3, 'evidence', 'Q3_缺失处理与Stale证据'),
        (ROOT/'outputs/r3_raw_explanation_20261026', 'raw_explanation_examples', 'Q2_原始量尺解释')]:
        for ext in ['.png', '.svg']:
            include(folder/(filename+ext), 'figures/'+label+ext, 'figure')
    draw_generation(output/'figures')
    for ext in ['.png', '.svg']:
        name = 'figures/Q3_生成诊断'+ext
        index.append(dict(file=name, source='outputs/task4_raw_q3_20261026/generation_validation.csv',
            role='generation_only_replot', sha256=sha256(output/name), bytes=(output/name).stat().st_size))
    for folder in EVIDENCE_FOLDERS:
        for name in ['summary.json', 'verification.json', 'protocol.json']:
            if (folder/name).exists():
                include(folder/name, 'evidence/'+folder.name+'/'+name, 'frozen_scientific_evidence')
    for name in ['current_plan_status.json', 'transfer_rule.json', 'predictive_qualification.json']:
        include(Q3/name, 'evidence/current_Q3/'+name, 'adoption_scope')
    save_csv(output, 'conclusions.csv', conclusions())
    save_csv(output, 'unresolved.csv', unresolved())
    save_csv(output, 'current_constraints.csv', current_constraints())
    # Chemical deltas are in supplied coordinates; no invented physical units.
    source, target = [pd.read_csv(Q3/name).iloc[0] for name in ['source119_profile.csv', 'transfer119_profile.csv']]
    fields = list(pd.read_csv(Q3/'transfer119_profile.csv').columns.drop('beer'))
    changes = pd.DataFrame([dict(field=f, before=source[f], after=target[f],
        delta=target[f]-source[f] if pd.notna(source[f]) and pd.notna(target[f]) else np.nan,
        scale='supplied_attachment_coordinate_not_automatically_physical_concentration',
        missing_rule='preserve_actual_unknown_source_value' if f=='X2.3.butanedione' else '') for f in fields])
    save_csv(output/'tables', 'q3_transfer__chemical_changes.csv', changes)
    # Forwardable source: only daily entry points, required src, and explicitly used assets.
    for folder in ['src/beer_models', 'src/beer_audit']:
        for p in sorted((ROOT/folder).glob('*.py')):
            include(p, 'code/'+p.relative_to(ROOT).as_posix(), 'source_code')
    for name in ['scripts/current.py', 'scripts/predict_q1.py', 'scripts/fetch_raw_sensory.py',
        'requirements-lock.txt', 'requirements-r3-lock.txt',
        'docs/待优化问题.md', 'docs/历史依赖与复现.md', 'docs/阶段三交付说明.md']+TRAIN_ASSETS+RENDER_ASSETS:
        include(ROOT/name, 'code/'+name, 'runnable_dependency')
    # Only the aggregates read by current verification are needed in the code snapshot.
    for name in ['outputs/q1_selective_tail_20261026/estimated_grades_all52.csv',
        'outputs/task4_raw_q3_20261026/generated_metadata.csv',
        'outputs/r3_raw_explanation_20261026/top3_raw_all50.csv',
        'outputs/q3_missing_aware_20261027/official_Q1_missing_scores52.csv.gz',
        'outputs/q3_missing_aware_20261027/missing_aware_defect_before_after.csv',
        'outputs/q3_missing_aware_20261027/nominal_styles.csv']:
        include(ROOT/name, 'code/'+name, 'verification_dependency')
    for folder in EVIDENCE_FOLDERS:
        for name in ['summary.json', 'verification.json', 'protocol.json']:
            if (folder/name).exists():
                include(folder/name, 'code/'+(folder/name).relative_to(ROOT).as_posix(), 'verification_dependency')
    for name in ['current_plan_status.json', 'transfer_rule.json', 'predictive_qualification.json']:
        include(Q3/name, 'code/'+(Q3/name).relative_to(ROOT).as_posix(), 'verification_dependency')
    # Model replay uses stored aggregate predictions only.
    for name in ['outputs/q1_selective_tail_20261026/full_fit_scores_all52.csv',
        'outputs/q3_stage2_20261026/generated_sensory52.csv.gz']:
        include(ROOT/name, 'code/'+name, 'prediction_replay_reference')
    save_csv(output, 'runtime_dependencies.csv', [dict(path=name, use='training' if name in TRAIN_ASSETS else 'fixed_profile_render')
                                                for name in TRAIN_ASSETS+RENDER_ASSETS])
    (output/'code/README.md').write_text('当前日常代码快照。结果与转发说明见上一级README.md、tables和figures。\n'
        '在本目录按上一级命令训练、预测、刷新和核验；完整历史研究与重新打包使用Git仓库。\n'
        '模型、原始附件与原始评分不在包内，需本机获取。\n', encoding='utf-8')
    (output/'README.md').write_text('''# MathorCup A 当前交付 20261027

本目录是当前取数版本。先读 conclusions.csv、unresolved.csv，再按 file_index.csv 找 tables、figures 和 evidence。
完整代码在 code/；不包含原始附件、个人评分记录、下载参考工作簿或模型二进制。

Q1正式中位评分不变，个人5/6召回仍0。Q2三种口径分开。Q3仅119号四代理条件案例；Stale、完整五缺陷及真实工艺未解决。9/12共同通过与22/32附加扰动同时报告。

队友运行：进入 code，Python3.12创建.venv并安装 requirements-r3-lock.txt。
查看结果不需训练；`python scripts/current.py status`。
本地取回固定原始评分：`python scripts/fetch_raw_sensory.py --no-proxy`。
训练：`python scripts/current.py train --source "原附件目录"`。
预测：`python scripts/current.py predict --input "化学谱.csv" --output outputs/my_prediction`。
刷新固定谱画像：`python scripts/current.py render --output outputs/my_render`。
核验：`python scripts/current.py verify --models outputs/current_models --source "原附件目录" --output outputs/my_check`。
每个输出目录须新建。Q3辅助预测需显式 --task q3，不能当正式Q1。

历史实验复现需完整GitHub仓库，见 code/docs/历史依赖与复现.md。本目录没有重新包装旧任务6的结论。
''', encoding='utf-8')
    # Generated summaries are also indexed; avoid self-referential file hashes.
    for name in ['README.md', 'code/README.md', 'conclusions.csv', 'unresolved.csv', 'current_constraints.csv', 'runtime_dependencies.csv', 'tables/q3_transfer__chemical_changes.csv']:
        p = output/name
        index.append(dict(file=name, source='generated_from_selected_sources', role='current_summary', sha256=sha256(p), bytes=p.stat().st_size))
    save_csv(output, 'file_index.csv', index)
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(output).as_posix())
    print(f'Current package: {output}\nForwardable ZIP: {archive}', flush=True)


def _numeric_equal(actual, expected, keys, columns):
    actual, expected = actual.copy(), expected.copy()
    for key in keys:
        actual[key], expected[key] = actual[key].astype(str), expected[key].astype(str)
    a = actual.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    if a.index.has_duplicates or b.index.has_duplicates or not a.index.equals(b.index):
        raise AssertionError('Prediction keys mismatch')
    delta = 0.
    for name in columns:
        av, bv = a[name].to_numpy(float, na_value=np.nan), b[name].to_numpy(float, na_value=np.nan)
        np.testing.assert_allclose(av, bv, atol=1e-9, rtol=0, equal_nan=True, err_msg=name)
        if np.isfinite(av-bv).any():
            delta = max(delta, float(np.nanmax(abs(av-bv))))
    return delta


def verify(output, models=None, source=None, package=None):
    """Current model/result contract. Reuse valid historical scientific checks."""
    if models is not None and source is None:
        raise ValueError('--models replay also requires --source')
    output.mkdir(parents=True, exist_ok=False)
    s = read_json(Q3/'current_plan_status.json')
    assert s['stale_evidence_sufficient'] is False and s['all_five_true_nonworsening_certified'] is False
    assert s['exact_zero_assumption_required'] is False and s['source_diacetyl_numeric_value'] is None
    assert s['validation_joint_fraction'] == 9/12 and s['validation_parameter_fraction_additional_diagnostic'] == 22/32
    assert read_json(ROOT/'outputs/q3_stale_risk_20261027/summary.json')['adopted'] is False
    assert read_json(Q1/'summary.json')['adopted'] is False
    official = pd.read_csv(Q1/'estimated_grades_all52.csv')
    unlabelled = ['A_acetaldehyde', 'F_acetaldehyde']
    assert len(official) == 250 and official[unlabelled].isna().all().all()
    gen = pd.read_csv(GEN/'generated_profiles.csv')
    assert len(gen) == 300
    metadata = pd.read_csv(GEN/'generated_metadata.csv')
    assert metadata.groupby('style').size().to_dict() == {'Kriek':100, 'Lambic':100, 'Tripel':100}
    slots = pd.read_csv(ROOT/'outputs/r3_raw_explanation_20261026/top3_raw_all50.csv')
    gbr = slots.loc[slots.method.eq('gbr_selected')]
    assert len(gbr) == 51 and gbr.target.nunique() == 17
    source_frame, after = [pd.read_csv(Q3/name) for name in ['source119_profile.csv', 'transfer119_profile.csv']]
    assert source_frame['X2.3.butanedione'].isna().all() and after['X2.3.butanedione'].isna().all()
    for field in ['DMS', 'X4VG', 'X.E..2.nonenal']:
        assert float(after[field].iloc[0]) <= float(source_frame[field].iloc[0])+1e-10
    fields = source_frame.columns.drop('beer')
    changed = np.abs(after[fields].to_numpy()-source_frame[fields].to_numpy()) > 1e-10
    assert changed.sum() == 130
    checks = dict(status='passed', data_beers=250, original_ratings=2197, generated_profiles=300,
        official_Q1_adoption_record_unchanged=True, current_models_replayed=False,
        missing_acetaldehyde_blank=True, raw_GBR_nonzero_ranks=51,
        source119_diacetyl_NA_preserved=True, changed_coordinates=130,
        Stale_unresolved=True, all_five_effect_unresolved=True,
        current_tables_checked=True, independent_new_external_validation=False,
        historical_scientific_evidence_reused=sorted({p.relative_to(ROOT).as_posix() for p in EVIDENCE_CHECKS.values()}))
    if models:
        b, q3 = [joblib.load(models/name) for name in ['q1.joblib', 'q3.joblib']]
        assert b['selected_model'] == 'extra_trees__median' and b['raw_panel_model'].penalties is None
        assert len(b['features']) == 224 and len(q3['features']) == 223 and q3['omitted_input'] not in q3['features']
        actual, _, _, _ = portrait(pd.read_csv(source/'1.csv', dtype=str), b)
        expected = pd.read_csv(Q1/'full_fit_scores_all52.csv')
        cols = ['estimated_grade_0_6', 'raw_median_estimate']+[n for n in expected if n.startswith(('probability_', 'uncalibrated_probability_'))]
        checks['Q1_full_fit_13000_rows_max_delta'] = _numeric_equal(actual, expected, ['beer','target'], cols)
        actual, _, _, _ = portrait(gen.rename(columns={'generated_id':'beer'}), b)
        expected = pd.read_csv(ROOT/'outputs/q3_stage2_20261026/generated_sensory52.csv.gz')
        checks['generated_15600_rows_max_delta'] = _numeric_equal(actual, expected, ['beer','target'], cols)
        frame = pd.concat([source_frame, after], ignore_index=True)
        actual, _, _, _ = portrait(frame, b)
        expected = pd.read_csv(Q3/'official_Q1_missing_scores52.csv.gz')
        checks['transfer_104_rows_max_delta'] = _numeric_equal(actual, expected, ['beer','target'], cols)
        actual, styles = auxiliary_defects(frame, q3, b['targets'])
        expected = pd.read_csv(Q3/'missing_aware_defect_before_after.csv').set_index('target')
        for i, suffix in enumerate(['before','after']):
            part = actual.loc[actual.beer.eq(str(frame.beer.iloc[i]))].set_index('target')
            for col, old in [('auxiliary_grade','grade'), ('probability_ge1','ge1'), ('probability_ge5','ge5'), ('probability_6','six')]:
                np.testing.assert_allclose(part.loc[DEFECTS,col], expected.loc[DEFECTS,old+'_'+suffix], atol=1e-9, rtol=0)
        expected_style = pd.read_csv(Q3/'nominal_styles.csv').set_index('model')
        for i, suffix in enumerate(['before','after']):
            part = styles.loc[styles.beer.eq(str(frame.beer.iloc[i]))].set_index('classifier')
            np.testing.assert_allclose(part.loc[expected_style.index,'Tripel_probability'], expected_style['Tripel_'+suffix], atol=1e-9, rtol=0)
            label = 'source_prediction' if i == 0 else 'after_prediction'
            assert (part.loc[expected_style.index,'predicted_style'] == expected_style[label]).all()
        checks.update(independent_full_data_model_rebuild_verified=True, current_models_replayed=True,
                      nominal_Q3_auxiliary_and_styles_replayed=True)
    if package:
        index = pd.read_csv(package/'file_index.csv')
        for row in index.itertuples():
            assert sha256(package/row.file) == row.sha256, row.file
        prohibited = [p for p in package.rglob('*') if p.is_file() and
            (p.suffix.lower() in ['.joblib','.xlsx','.npy'] or 'Raw sensory data anonymized' in p.name)]
        assert not prohibited, prohibited
        checks['package_files_hashed'] = len(index)
        checks['package_excludes_raw_records_and_model_binaries'] = True
    write_json(output/'verification.json', checks)
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
