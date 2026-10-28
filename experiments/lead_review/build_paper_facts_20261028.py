"""Build the paper-facing fact sheet (r2) from the official outputs.

Read-only with respect to outputs/; writes only to
deliverables/paper_handoff_20261028/ (tables, figures, two markdown files,
paper_claims.csv, verification.json).

Run:
    $env:PYTHONPATH='src'; .venv\\Scripts\\python.exe experiments\\lead_review\\build_paper_facts_20261028.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "deliverables" / "paper_handoff_20261028"
TAB = OUT / "tables"
FIG = OUT / "figures"
PKG = ROOT / "deliverables" / "current_20261028"

Q1D = ROOT / "outputs" / "q1_decision_det_20261028"
Q1H = ROOT / "outputs" / "q1_decision_20261028"
Q1P = ROOT / "outputs" / "q1_panelmean_20261028"
Q2 = ROOT / "outputs" / "q2_top3_20261028"
Q2A = ROOT / "outputs" / "q2_ablation_20261028"
Q3G = ROOT / "outputs" / "q3_copula_v3_20261028"
Q3E = ROOT / "outputs" / "q3_evidence_20261028"
Q3M = ROOT / "outputs" / "q3_migration_20261028"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pct(x) -> str:
    return f"{100.0 * float(x):.2f}%"


def f4(x) -> str:
    return f"{float(x):.4f}"


q1 = load(Q1D / "summary.json")
cal = pd.read_csv(Q1D / "q1__metric_calibers.csv")
inv = pd.read_csv(Q1D / "q1__batch_invariance.csv")
q1h = load(Q1H / "summary.json")
q2 = load(Q2 / "q2__summary.json")
q2a = load(Q2A / "summary.json")
q3g = load(Q3G / "summary.json")
q3e = load(Q3E / "summary.json")
q3m = load(Q3M / "summary.json")

M = q1["metrics"]
ET = M["beer_level|q1_integer_median_legacy"]
DET = M["beer_level|q1_fixed_threshold_quantile"]
HIST = M["beer_level|q1_topk_parent_capped_old_baseline"]
IND_DET = M["individual_taster_weighted|q1_fixed_threshold_quantile"]
ORACLE = q1h["panel_median_oracle"]
PAR = q1h["parent_child"]
DIST = q1["new_distribution"]
FALLBACK = q1["threshold_fallback"]
OVERPROMO = q1["over_promotion_diagnostic"]["promoted_by_target_top10"]

Q1P_M = pd.read_csv(Q1P / "q1__panel_mean_metrics.csv")
Q1P_ET = Q1P_M[(Q1P_M.method == "extra_trees_direct") & (Q1P_M.scope == "pooled_3repeats")].iloc[0]
Q1P_CMP = float(Q1P_ET["panel_median_within_one"])

q2_pred = q2a["predictive"]
q2_over = q2a["overlap_summary"]
q2_verdict = q2a["verdict"]

q3_val = q3g["validation_means"]
q3_style = {row["style"]: row for row in q3g["quality_by_style"]}
q3_ablation = {row["generator"]: row for row in q3e["ablation"]["headline"]}
q3_conf = q3e["tripel_conflict"]
q3_style_ev = q3e["style_evidence"]
q3_det = q3m["det_rerun"]

# ----------------------------------------------------------------------------
# tables
for _dir in (TAB, FIG):
    _dir.mkdir(parents=True, exist_ok=True)
    for _old in _dir.iterdir():
        if _old.is_file():
            _old.unlink()
COPY = {
    Q1D / "q1__submission_long_12500.csv": "q1__submission_long_12500.csv",
    Q1D / "q1__submission_wide_250x50.csv": "q1__submission_wide_250x50.csv",
    Q1D / "q1__evaluation_long_12500.csv": "q1__evaluation_long_12500.csv",
    Q1D / "q1__metrics_summary.csv": "q1__metrics_summary.csv",
    Q1D / "q1__metric_calibers.csv": "q1__metric_calibers.csv",
    Q1D / "q1__batch_invariance.csv": "q1__batch_invariance.csv",
    Q1D / "q1__thresholds.csv": "q1__thresholds.csv",
    Q1H / "q1__metrics_summary.csv": "q1_history__metrics_summary.csv",
    Q1H / "q1__metric_calibers.csv": "q1_history__metric_calibers.csv",
    Q1P / "q1__panel_mean_metrics.csv": "q1_panelmean__metrics.csv",
    Q1P / "q1__panel_mean_comparison.json": "q1_panelmean__comparison.json",
    Q2 / "q2__top3_all50_150.csv": "q2__top3_all50_150.csv",
    Q2 / "q2__top3_wide_50.csv": "q2__top3_wide_50.csv",
    Q2 / "q2__stability_bootstrap.csv": "q2__stability_bootstrap.csv",
    Q2 / "q2__method_jaccard.csv": "q2__method_jaccard.csv",
    Q2 / "q2__network_edges_0_5.csv": "q2__network_edges_0_5.csv",
    Q2 / "q2__network_communities.csv": "q2__network_communities.csv",
    Q2 / "q2__evidence_tier_counts.csv": "q2__evidence_tier_counts.csv",
    Q2 / "q2__summary.json": "q2__summary.json",
    Q2A / "q2__ablation_overlap.csv": "q2_ablation__overlap.csv",
    Q2A / "q2__ablation_stability.csv": "q2_ablation__stability.csv",
    Q2A / "q2__ablation_predictive.csv": "q2_ablation__predictive.csv",
    Q2A / "q2__ablation_top3.csv": "q2_ablation__top3.csv",
    Q2A / "summary.json": "q2_ablation__summary.json",
    Q3G / "generation_validation.csv": "q3_generation__validation.csv",
    Q3G / "generation_gate.csv": "q3_generation__gate.csv",
    Q3G / "generation_quality_by_style.csv": "q3_generation__quality_by_style.csv",
    Q3G / "summary.json": "q3_generation__summary.json",
    Q3M / "s2c_migration_results.csv": "q3_migration__results_33.csv",
    Q3M / "s2c_candidate_counts.csv": "q3_migration__candidates.csv",
    Q3M / "s2_source_defects_11beers.csv": "q3_migration__source_defects.csv",
    Q3M / "s2c_evidence_by_case.csv": "q3_migration__evidence.csv",
    Q3M / "s2c_sensory_before_after_52.csv": "q3_migration__sensory_3432.csv",
    Q3M / "s2c_migration_spectra_224_wide.csv": "q3_migration__spectra_wide.csv",
    Q3M / "s2c_det_grade_changes.csv": "q3_migration__det_grade_changes.csv",
    Q3M / "summary.json": "q3_migration__summary.json",
    Q3E / "style_classifier_oof.csv": "q3_evidence__style_classifier.csv",
    Q3E / "tripel_conflict.csv": "q3_evidence__tripel_conflict.csv",
    Q3E / "generator_ablation.csv": "q3_evidence__generator_ablation.csv",
    Q3E / "summary.json": "q3_evidence__summary.json",
}
missing = [str(src) for src in COPY if not src.exists()]
if missing:
    sys.exit("missing official tables: " + ", ".join(missing))
for src, name in COPY.items():
    shutil.copyfile(src, TAB / name)
for fig in sorted((PKG / "figures").glob("*.png")):
    shutil.copyfile(fig, FIG / fig.name)
for fig in sorted((PKG / "figures").glob("*.svg")):
    shutil.copyfile(fig, FIG / fig.name)

# ----------------------------------------------------------------------------
# claims
claims = [
    ("Q1", "Q1 的正式决策是逐样本固定阈值，与请求批次大小无关", "batch_invariance 5 个场景 mismatch_cells 全 0", "tables/q1__batch_invariance.csv"),
    ("Q1", "固定阈值主表的面板中位 MAE", f4(DET["panel_median_mae"]), "tables/q1__metrics_summary.csv"),
    ("Q1", "固定阈值主表完全命中率与 ±1 命中率", f"{pct(DET['exact_match'])} / {pct(DET['within_one'])}", "tables/q1__metrics_summary.csv"),
    ("Q1", "固定阈值主表预测 ≥5 分格数与精确率/召回率", f"{int(DET['n_pred_ge5'])} 格 / {pct(DET['precision_ge5'])} / {pct(DET['recall_ge5'])}", "tables/q1__metric_calibers.csv"),
    ("Q1", "固定阈值主表恰好 5 分与恰好 6 分的格数与精确率", f"321 格 {pct(0.071651)}；59 格 {pct(0.050847)}", "tables/q1__metric_calibers.csv"),
    ("Q1", "ET 整数中位基线的面板中位 MAE（无任何 5/6 预测）", f4(ET["panel_median_mae"]), "tables/q1__metrics_summary.csv"),
    ("Q1", "历史批次 top-k 表（固定整批 250 款）的面板中位 MAE 与 ≥5 格数", f"{f4(HIST['panel_median_mae'])} / {int(HIST['n_pred_ge5'])} 格", "tables/q1_history__metrics_summary.csv"),
    ("Q1", "个体品评员加权的 MAE 与信息下界", f"{f4(IND_DET['panel_median_mae'])}（下界 {f4(ORACLE['individual_weighted_mae'])}）", "tables/q1__metrics_summary.csv"),
    ("Q1", "嵌套折外标定到面板均值（未采纳为主表，仅作对照）", f"线性 {f4(0.5054899)}；风格+线性 {f4(0.4660719)}", "tables/q1_panelmean__metrics.csv"),
    ("Q1", "面板均值直接回归相对嵌套基线的改善与门槛", f"+{f4(0.0142134)}；gate4 ±1 {f4(Q1P_CMP)} < 0.9119（未采纳）", "tables/q1_panelmean__metrics.csv"),
    ("Q1", "父子约束：24 对 child<=parent 的违规数", f"cap 前 {PAR['precap_violations']} → cap 后 {PAR['postcap_violations']}（仅下调 {PAR['changed_cells']} 格）", "tables/q1__evaluation_long_12500.csv"),
    ("Q1", "固定阈值的过度提升诊断（clove 单维贡献）", f"{OVERPROMO['clove']} / {int(DET['n_pred_ge5'])} 个 ≥5 分", "outputs/q1_decision_det_20261028/summary.json"),
    ("Q2", "Q2 Top-3 覆盖与行数", f"{q2['coverage']['targets_with_3_rows']}/{q2['coverage']['total_targets']} 维、{q2['n_top3_rows']} 行、{q2['n_candidate_compounds']} 候选", "tables/q2__summary.json"),
    ("Q2", "共表达图规模", f"{q2['graph']['nodes']} 节点 / {q2['graph']['edges']} 边 / {q2['graph']['communities']} 社区（最大 {q2['graph']['largest_community']}）", "tables/q2__summary.json"),
    ("Q2", "bootstrap 入选频率（Top-3 槽位均值）与 Jaccard", f"{f4(q2['bootstrap_freq']['mean_all_slots'])} / {f4(q2['bootstrap_jaccard']['mean_over_targets'])}", "tables/q2__stability_bootstrap.csv"),
    ("Q2", "graph-ridge 与 ridge 的 Top-3 方法 Jaccard", f"{f4(0.6)}（45/50 维≥0.5）", "tables/q2__method_jaccard.csv"),
    ("Q2", "证据分级", f"A {q2['evidence']['tier_counts']['A']} / B {q2['evidence']['tier_counts']['B']} / C {q2['evidence']['tier_counts']['C']}", "tables/q2__evidence_tier_counts.csv"),
    ("Q2", "图邻域消融：Top-3 Jaccard 与稳定性增益", f"{f4(q2_over['jaccard_mean'])}；频率 +{f4(q2_over['freq_difference_mean'])}", "tables/q2_ablation__overlap.csv"),
    ("Q2", "图邻域对预测的作用（不显著）", f"MAE 增量 +{f4(q2_pred['mae']['gain_mean'])}（p={f4(q2_pred['mae']['p_value'])}）→ {q2_verdict['verdict']}", "tables/q2_ablation__predictive.csv"),
    ("Q3", "v3 生成器的预注册替换门槛", f"通过 14/14（gate_failures = {len(q3g['gate_failures'])}）", "tables/q3_generation__gate.csv"),
    ("Q3", "v3 生成谱质量：方差比 / 秩相关误差 / 近重复", f"{f4(q3_val['hybrid_support_projected']['variance_ratio'])} / {f4(q3_val['hybrid_support_projected']['rank_correlation_rmse'])} / {pct(q3_val['hybrid_support_projected']['internal_near_duplicate_rate'])}", "tables/q3_generation__validation.csv"),
    ("Q3", "插值基线对照（方差比 / 秩相关误差 / 近重复）", f"{f4(q3_val['interpolate']['variance_ratio'])} / {f4(q3_val['interpolate']['rank_correlation_rmse'])} / {pct(q3_val['interpolate']['internal_near_duplicate_rate'])}", "tables/q3_generation__validation.csv"),
    ("Q3", "生成谱缺陷 cap 达标（三风格 × 三缺陷）", "defect_safety 全 pass（DMS 17.1 / 双乙酰 0.081 / 4VG 按风格 cap）", "tables/q3_generation__validation.csv"),
    ("Q3", "迁移三口径（A/B/C）通过数", f"各 {q3m['criteria_pass']['A_median']}/33（候选级 {q3m['candidate_pass']['A_median']}/660）", "tables/q3_migration__results_33.csv"),
    ("Q3", "迁移到目标质心的平均距离降幅", f"Tripel {f4(q3m['mean_distance_reduction_pct']['Tripel'])}% / Kriek {f4(q3m['mean_distance_reduction_pct']['Kriek'])}% / Lambic {f4(q3m['mean_distance_reduction_pct']['Lambic'])}%", "tables/q3_migration__results_33.csv"),
    ("Q3", "迁移后缺陷状态（五缺陷）", "improved/unchanged 66/66，worse 0；t2n 与 Stale hops 均 unknown", "tables/q3_migration__results_33.csv"),
    ("Q3", "迁移感官画像（66 条 × 52 维）与决策层一致性", f"{q3m['counts']['sensory']} 行，replay mismatch {q3m['sensory_replay']['mismatch_cells']}", "tables/q3_migration__sensory_3432.csv"),
    ("Q3", "迁移证据分级 claim_level", f"model_supported {q3m['claim_level_counts']['model_supported']} / conditional {q3m['claim_level_counts']['conditional']} / undecidable 0", "tables/q3_migration__evidence.csv"),
    ("Q3", "风格分类器折外准确率（ET / logistic / 最近质心）", f"{f4(q3_style_ev['accuracy']['extra_trees'])} / {f4(q3_style_ev['accuracy']['logistic'])} / {f4(q3_style_ev['accuracy']['nearest_centroid_oof'])}", "tables/q3_evidence__style_classifier.csv"),
    ("Q3", "Tripel 训练样本自身满足三缺陷 cap 的比例", f"{q3_conf['n_defect_safe_samples_median']}/{q3_conf['n_style_samples']}（质心 DMS {f4(q3_conf['centroid_DMSO_threshold_ratio'])}×、双乙酰 {f4(q3_conf['centroid_Diacetyl_threshold_ratio'])}×）", "tables/q3_evidence__tripel_conflict.csv"),
    ("Q3", "缺陷安全参照与源距离（几何可兼得）", f"距离比 {f4(q3_conf['mean_target_over_safe_distance'])}，投影只改 {f4(q3_conf['mean_projection_change_share_pct'])}% 坐标，{q3_conf['n_cases_dms_cap_median']}/11 达 DMS cap", "tables/q3_evidence__tripel_conflict.csv"),
    ("Q3", "生成器消融（插值 → v2 → v3，秩相关误差）", f"{f4(q3_ablation['interpolate']['rank_correlation_rmse'])} → {f4(q3_ablation['donor_residual_projected']['rank_correlation_rmse'])} → {f4(q3_ablation['hybrid_support_projected']['rank_correlation_rmse'])}", "tables/q3_evidence__generator_ablation.csv"),
    ("Q3", "迁移评分随 Q1 决策切换的格数变化", f"{q3_det['changed_cells']} 格（升 {q3_det['changed_up']} / 降 {q3_det['changed_down']}，可比 {q3_det['cells_compared']}）", "tables/q3_migration__det_grade_changes.csv"),
]
claims_df = pd.DataFrame(claims, columns=["question", "claim", "value", "evidence"])
claims_df.insert(0, "claim_id", [f"C{i:02d}" for i in range(1, len(claims_df) + 1)])
claims_df.to_csv(OUT / "paper_claims.csv", index=False, encoding="utf-8-sig")

# ----------------------------------------------------------------------------
doc = rf"""# MathorCup 2026 A 题：论文取数与口径（r2 / 20261028）

> **版本说明（先读这一行）**：本文件是 **r2**，取代 r1 的全部数字。r2 对应提交 `0a70627`；
> r1 对应 `9fb9e57` 之前。r1→r2 的主要差异：Q1 主表由"批次 top-k"改为"逐样本固定阈值"（批次无关），
> 因此头条 MAE 由 0.4065 变为 0.4797；Q3 生成谱采用通过预注册门槛的 v3，迁移表按新 Q1 决策重跑。
> 所有数字都能在 `tables/` 同名文件或 `deliverables/current_20261028/` 中核对；`paper_claims.csv` 是逐条主张表。

## 0. 一页综述

| 问题 | 一句话结论 | 头条数字 | 最强证据 | 主要限制 |
|---|---|---|---|---|
| Q1 感官评分 | 用化合物谱预测 0–6 分级；正式决策为批次无关的逐样本固定阈值 | 面板中位 MAE **{f4(DET['panel_median_mae'])}**、完全命中 **{pct(DET['exact_match'])}**、±1 **{pct(DET['within_one'])}**；历史批次 top-k 对照 0.4065、ET 中位对照 0.3928 | 12,500 格全覆盖 + 批次一致性 5/5 + 父子约束 0 违规 | 个体品评员层面噪声大（MAE {f4(IND_DET['panel_median_mae'])}，下界 {f4(ORACLE['individual_weighted_mae'])}）；5/6 分的精确率低 |
| Q2 关键因素 | 50/50 维 × Top-3 = 150 条，图邻域加权 + bootstrap 稳定性 + 证据分级 | 150 条无空位；bootstrap 频率均值 0.474；graph vs ridge Jaccard 0.600 | 200× bootstrap + 四种方法六对 Jaccard + 化学合理性分级 A8/B70/C72 | 图提升的是稳定性不是预测（消融 partial_gain）；只能写关联不能写因果 |
| Q3 生成与迁移 | 风格条件生成器通过预注册替换门槛；迁移按题面口径 33/33 | 生成谱 300 条、门槛 14/14；迁移 33/33、候选 660/660、缺陷 worse 0 | 门槛逐条判定 + 前后完整 224 维谱 + 52 维感官画像 + 折外风格分类器 | 风格判别弱（Tripel 迁移后 1/11 判为 Tripel）；非酿造配方、无真实品评 |

## 1. 三条全局口径（论文必须披露）

1. **数据规模**：题面称"226 款啤酒"，附件实际为 250 款。本研究按 250 款 × 50 个可评感官维度 = 12,500 个 0–6 单元格建模；评分卡 52 项中两个乙醛槽（Aroma/Flavour 各一）无任何标签，标为 NA，不计入 12,500。
2. **0–6 标签来源**：附件 3.xlsx 是**空白评分模板**（12 张评分卡与 summary 全为 0，无人工分）。0–6 标签由外部 Zenodo 记录（DOI 10.5281/zenodo.23624915，Raw sensory data anonymized.csv，2,197 条、16 位品评员）按"啤酒内相对名次"还原为 0–6 等级。论文必须写成"标签为外部原始品评记录的相对等级近似"，**不能写成 3.xlsx 实测分**。
3. **2.csv 的性质**：题面写 Pearson，但 2.csv 实测是 1.csv 的 **Spearman** 矩阵（按位置复算 max|Δ|=6.66e-16，且 231 列 26,565 对与 Spearman 一致）。共表达图按 |ρ|≥0.5 建图，写论文时注明"题面表述与实际矩阵口径不一致，我们按实测 Spearman 使用"。

其余数据事实：1.csv 缺失 73 格（0.125%，其中 amyl.acetate 27 格）；7.csv 750 格缺失集中在 3 个 palate 列；4.csv 无缺失；总共 231 列输入（剔除 7 个派生汇总列后 224 列进入模型）。

## 2. Q1：感官评分

### 2.1 主张
本题要求"以 1.csv 化合物数据给出 3.xlsx 中约 40 个感官维度的 0–6 分级"。我们的建模链条是：
先预测**面板级中位数**（16 位品评员加权的啤酒级参考），再把连续预测**决策**为 0–6 整数；正式决策规则为逐样本固定阈值：
`p(6)≥tau6 → 6；p(≥5)≥tau5 → 5；否则 min(ET 整数中位, 4)`，最后施加 24 对父子约束（child≤parent）。
阈值 tau 只由训练折的分位估计（不在验证折上调），因此**同一份模型对"整批 250 款"和"逐条 1 款"给出完全相同的结果**（5 个批次场景 mismatch=0）。

### 2.2 三套口径并排（同一个 12,500 格口径，别混用）

| 口径 | 面板中位 MAE | 完全命中 | ±1 | ≥5 预测格数 | ≥5 精确/召回 | 6 分格数 |
|---|---|---|---|---|---|---|
| ET 整数条件中位（历史基线，永不输出 5/6） | {f4(ET['panel_median_mae'])} | {pct(ET['exact_match'])} | {pct(ET['within_one'])} | 0 | 0 / 0 | 0 |
| 历史批次 top-k + 父子 cap（固定整批 250 款） | {f4(HIST['panel_median_mae'])} | {pct(HIST['exact_match'])} | {pct(HIST['within_one'])} | {int(HIST['n_pred_ge5'])} | {pct(HIST['precision_ge5'])} / {pct(HIST['recall_ge5'])} | {int(HIST['n_pred_6'])} |
| **正式：逐样本固定阈值 + 父子 cap** | **{f4(DET['panel_median_mae'])}** | **{pct(DET['exact_match'])}** | **{pct(DET['within_one'])}** | {int(DET['n_pred_ge5'])} | {pct(DET['precision_ge5'])} / {pct(DET['recall_ge5'])} | {int(DET['n_pred_6'])} |

写作建议：**列正式口径为主表，另两列作为对照**，并在正文用一句话解释 trade-off：
"批次 top-k 表在固定整批 250 款时可多得约 0.073 的 MAE 优势，但它把'本批有多少格'当成决策输入，换批次或逐条预测结果会改变（逐条预测时 ≥5 分全部消失）；正式交付选择批次无关的固定阈值，牺牲部分指标换取可部署的一致性。"

### 2.3 逐维 5/6 与约束
- 固定阈值分布 `{DIST}`；非 0 分格数 {12500 - int(DIST['0']):,}，0 分 {int(DIST['0']):,} 格。
- ≥5 分 {int(DET['n_pred_ge5'])} 格中，恰好 5 分 {int(cal[cal.caliber == 'exact5']['n_predicted'].iloc[0])} 格（精确 {pct(cal[cal.caliber == 'exact5']['precision'].iloc[0])}、召回 {pct(cal[cal.caliber == 'exact5']['recall'].iloc[0])}），恰好 6 分 {int(DET['n_pred_6'])} 格（精确 {pct(DET['precision_6'])}、召回 {pct(DET['recall_6'])}）。
- 阈值回退：训练折无 5 分正例的维度 {len(FALLBACK['no_ge5'])} 个、无 6 分正例的 {len(FALLBACK['no_6'])} 个（这些维度按规则只可能输出 ≤4，论文可在附录列出，避免"为什么某维没有 5 分"的追问）。
- 父子约束：24 对 child≤parent；历史批次表违规 12 对 → cap 后 0 对（仅下调 12 格，主表分布不受其他影响）。
- 诚实点：固定阈值无法约束"一次请求里判出多少 5 分"；最坏情况是 clove 一个维度贡献了 {OVERPROMO['clove']} 个 ≥5 分（训练折正例 1/166、阈值 0.005356，验证折该维 84 格 p_ge5 同为 0.006574）。论文应把这条写进"局限"，不要假装 5 分精确率是可控的。

### 2.4 个体品评员层级（说明为什么不做个体预测）
- 面板级（啤酒×维度）是本题要求的口径；个体级预测 MAE {f4(IND_DET['panel_median_mae'])}，即使把"已观察面板中位"当作 oracle，个体 MAE 也只有 {f4(ORACLE['individual_weighted_mae'])}（信息下界）。
- 个体级 5/6 分精确识别在机制上不可达：个体 5 分质量中约 82% 来自面板中位 <5 的啤酒，啤酒内评分方差占逐维均值约 69.5%。论文用它解释"为什么高分召回低"，而不是当作失败。

### 2.5 未采纳的两个候选（可写进"消融/负结果"）
1. **嵌套折外线性标定到面板均值**：MAE {f4(0.5054899)}；加风格先验收缩 {f4(0.4660719)}。未替换主表（主表口径是 0–6 整数，标定列仅作辅助）。
2. **ET 直接回归面板均值**：相对嵌套基线改善 {f4(0.0142134)}（bootstrap 95% [0.0082, 0.0197]），但 round 后 ±1 {f4(Q1P_CMP)} 低于预注册绝对下限 0.9119（差 0.05pt）→ 不采纳，门槛不放宽。

### 2.6 Q1 可写结论（一句话）与"不要写"
可写："在批次无关的固定阈值决策下，12,500 格的面板中位 MAE 为 0.4797、完全命中 72.2%、±1 89.6%；5 分与 6 分可被输出但精确率有限（9.5%/5.1%），高分识别受品评噪声限制。"
不要写：①"个体 5/6 分可被精确识别"；②"3.xlsx 提供了 0–6 标签"；③"我们的 MAE 0.39 是交付口径"（那是 ET 中位历史基线，不含 5/6）；④"clove 的 84 格是模型有效"（那是过度提升）。

## 3. Q2：关键因素化合物 Top-3

### 3.1 主张
按 |ρ|≥0.5 建化合物共表达图（{q2['graph']['nodes']} 节点、{q2['graph']['edges']} 边、{q2['graph']['communities']} 社区），
在 219 个候选化合物上做 Ridge（alpha=1000）回归，得分 `score = |β| + 0.5 × 邻居均值|β|`，
每个感官维度取 Top-3，得 **50/50 维、150 条无空位**（旧表 17/50 只是 GBR 原始分子表，不是覆盖上限）。

### 3.2 数字
- 稳定性：200× 按啤酒 bootstrap，Top-3 槽位入选频率均值 {f4(q2['bootstrap_freq']['mean_all_slots'])}（中位 {f4(q2['bootstrap_freq']['median_all_slots'])}），{q2['bootstrap_freq']['n_slots_ge_0.5']}/150 槽 ≥0.5；bootstrap Top-3 Jaccard 均值 {f4(q2['bootstrap_jaccard']['mean_over_targets'])}。
- 方法稳健：graph-ridge vs ridge 的 Top-3 Jaccard {f4(0.6)}（45/50 维 ≥0.5，重合 2.16/3）；六对方法平均 Jaccard {f4(q2['method_jaccard_overall']['target_mean_mean'])}。
- 证据分级：A {q2['evidence']['tier_counts']['A']} / B {q2['evidence']['tier_counts']['B']} / C {q2['evidence']['tier_counts']['C']}（A 级全部 high confidence；A+B 占 {pct(q2['evidence']['tier_AB_share'])}）。
- 图的作用（消融）：有图 vs 无图 Top-3 Jaccard {f4(q2_over['jaccard_mean'])}（{q2_over['n_targets_identical']}/50 维完全相同、{q2_over['n_targets_jaccard_ge_0_5']}/50 维 ≥0.5）；稳定性频率 +{f4(q2_over['freq_difference_mean'])}、Jaccard +{f4(q2_verdict['observed']['jaccard_gain'])}；**留出预测无显著提升**（MAE 增量 +{f4(q2_pred['mae']['gain_mean'])}，p={f4(q2_pred['mae']['p_value'])}）→ 结论 `partial_gain`：图的价值在"更稳、更可解释"，不在"更准"。

### 3.3 可写结论与"不要写"
可写："在 50 个感官维度上给出 150 条 Top-3 因素，bootstrap 入选频率均值 0.474，四种方法间 Jaccard 0.60，图邻域加权使稳定性频率提升 0.028 而不改变预测精度——因此把图定位为'提升稳定性和化学可解释性'的结构先验。"
不要写：①任何因果表述（"X 导致 Y"）；②"图提升了预测精度"（消融 p=0.31）；③把 17/50 当覆盖不足（那是子表口径）。

## 4. Q3：风格条件生成与跨风格迁移

### 4.1 生成
- 方案演进（同折同 seed）：插值基线 → v2（donor 残差 bootstrap）→ **v3（混合：0.5 插值内部分量 + 0.5 donor 残差 + 支撑投影）**。
- 指标（逐方法均值，300 条 = 3 风格 × 100）：方差比 {f4(q3_val['interpolate']['variance_ratio'])} → {f4(q3_val['donor_residual_projected']['variance_ratio'])} → **{f4(q3_val['hybrid_support_projected']['variance_ratio'])}**；秩相关误差 {f4(q3_val['interpolate']['rank_correlation_rmse'])} → {f4(q3_val['donor_residual_projected']['rank_correlation_rmse'])} → **{f4(q3_val['hybrid_support_projected']['rank_correlation_rmse'])}**；近重复 {pct(q3_val['interpolate']['internal_near_duplicate_rate'])} → 0 → {pct(q3_val['hybrid_support_projected']['internal_near_duplicate_rate'])}；覆盖率 {pct(q3_val['hybrid_support_projected']['heldout_coverage'])}。
- 预注册替换门槛：v3 **14/14 通过**（13 条实质规则 + 1 条汇总），据此替换官方插值谱；门槛与基线在协议里预先冻结，未做事后调参。
- 逐风格质量（v3）：Tripel 方差比 {f4(q3_style['Tripel']['variance_ratio'])}、Kriek {f4(q3_style['Kriek']['variance_ratio'])}、Lambic {f4(q3_style['Lambic']['variance_ratio'])}；覆盖率 95% 区间均 1.0；近重复 ≤{pct(max(r['near_duplicate_rate'] for r in q3g['quality_by_style']))}。
- 缺陷 cap：DMS 17.1、双乙酰 0.081、4VG 按风格 cap，`defect_safety` 全 pass。

### 4.2 迁移（题面口径：把 Pils/Lager 谱迁移为目标风格且缺陷不恶化）
- 11 款源酒 × 3 目标 = **33 条**；三口径（超阈集合不扩张+严重度不增 / 逐项 e_after≤e_before / 严重度加总）各 **33/33**，候选级 660/660。
- 平均到目标质心距离降幅：Tripel {f4(q3m['mean_distance_reduction_pct']['Tripel'])}%、Kriek {f4(q3m['mean_distance_reduction_pct']['Kriek'])}%、Lambic {f4(q3m['mean_distance_reduction_pct']['Lambic'])}%；Tripel 的 λ*=0.55（9 款）或 0.50（2 款，因其在网格内无更大 λ 更近质心），Kriek/Lambic λ*=1.0。
- 缺陷：DMS improved {q3m['defect_status_counts']['DMS']['improved']} / unchanged {q3m['defect_status_counts']['DMS']['unchanged']}；双乙酰 improved {q3m['defect_status_counts']['Diacetyl']['improved']} / unchanged {q3m['defect_status_counts']['Diacetyl']['unchanged']} / unknown {q3m['defect_status_counts']['Diacetyl']['unknown']}；4VG 全 unchanged；t2n 与 Stale hops 全 unknown（无阈值/无化学坐标）；**worse 0**。
- 证据分级：model_supported {q3m['claim_level_counts']['model_supported']} / conditional {q3m['claim_level_counts']['conditional']} / undecidable 0；风格诊断 Tripel {q3m['style_diagnostic_counts']['Tripel']}/11、Kriek {q3m['style_diagnostic_counts']['Kriek']}/11、Lambic {q3m['style_diagnostic_counts']['Lambic']}/11。
- 补了迁移后感官画像：66 条（前/后各 33）× 52 维 = 3,432 行，全部走正式 `inference.run` 决策层；与交付表重放一致（mismatch 0）。
- 随 Q1 决策切换重跑：可比 {q3_det['cells_compared']} 格中 {q3_det['changed_cells']} 格变化（升 {q3_det['changed_up']}/降 {q3_det['changed_down']}），缺陷 worse 仍 0，三口径计数不变。

### 4.3 风格可辨识度（要如实写）
- 折外风格分类器：ET 准确率 {f4(q3_style_ev['accuracy']['extra_trees'])}（宏召回 {f4(q3_style_ev['macro_recall']['extra_trees'])}）、logistic {f4(q3_style_ev['accuracy']['logistic'])}、最近质心 {f4(q3_style_ev['accuracy']['nearest_centroid_oof'])}（22 类多数类基线 0.124）。
- 也就是说：风格是**部分可辨识**，不是"不可辨识"（旧文档说法过强），但**不足以单独证明迁移成功**——迁移谱被判为 Tripel 的只有 1/11（logistic 判 11/11、ET 判 1/11，两种分类器结论相反 → 风格支持不确定）。
- Tripel 冲突的根源：Tripel 训练集 **{q3_conf['n_defect_safe_samples_median']}/{q3_conf['n_style_samples']}** 满足三缺陷 cap（质心 DMS {f4(q3_conf['centroid_DMSO_threshold_ratio'])}×、双乙酰 {f4(q3_conf['centroid_Diacetyl_threshold_ratio'])}× 阈值）；把质心投影到 cap 后到源的距离只多 {f4(q3_conf['mean_target_over_safe_distance'])} 倍、投影只改 {f4(q3_conf['mean_projection_change_share_pct'])}% 坐标、{q3_conf['n_cases_dms_cap_median']}/11 达 DMS cap → **"迁移到典型 Tripel"与"缺陷不恶化"在几何上可兼得，冲突来自数据本身**（典型 Tripel 样本自身超标）。

### 4.4 可写结论与"不要写"
可写："风格条件生成器在预注册门槛下替换了插值基线（方差比 0.84、秩相关误差 0.136、近重复 7.2%、覆盖率 94.9%）；33 条迁移全部满足缺陷不恶化，其中 9 条可由模型支持、24 条为条件性结论。"
不要写：①"生成谱是可酿造的配方"（统计参考，无物理/感官验证）；②"迁移后风格被识别为目标风格"（Tripel 1/11）；③"缺陷安全已穷尽"（t2n/Stale 不可判定）；④"Lambic 迁移成功"（距离降 81%，但仅 2/11 判 Lambic）。

## 5. 建议论文结构（正文 ≤30 页）
1. 摘要：三问各 1 句结论 + 一个方法论亮点（预注册门槛 + 批次无关决策 + 全量重放）。
2. 问题重述与分析：三问关系（Q1 是监督预测、Q2 是关联解释、Q3 是生成+约束）。
3. 数据预处理：口径披露（§1）、缺失处理、标签构造、2.csv 口径修正。
4. Q1 模型与决策：嵌套验证、风格先验、固定阈值决策、父子约束、三口径对照。
5. Q2 共表达图与 Top-3：建图、加权、bootstrap 稳定性、方法 Jaccard、证据分级。
6. Q3 生成与迁移：生成器、预注册门槛、迁移三判据、缺陷状态、风格证据。
7. 结果与讨论：三口径并排表、消融（Q2 partial_gain、Q3 生成器消融）、失败分析（个体噪声、Tripel 冲突）。
8. 局限与展望、参考文献、附录（全部源程序、表索引、复现命令）。

图表建议（均已在 `tables/` 与 `figures/` 提供）：Q1 决策层对照（`Q1_决策层对照`）、Q2 稳定性（`Q2_稳定性`）、Q3 生成门槛（`Q3_生成门槛`）、Q3 迁移缺陷（`Q3_迁移缺陷`）。

## 6. 评委可能追问的 8 个点与应答
1. **为什么不用 3.xlsx 的分数？** 3.xlsx 是空白模板；标签来自外部 Zenodo 2197 条原始记录（§1）。这是本题最大前提，必须主动说。
2. **MAE 0.48 是否偏低？** 0–6 尺度上完全命中 72%、±1 89.6%；历史批次口径 0.407 但批次相关；两者都为真，trade-off 见 §2.2。
3. **为什么 5 分精确率只有 9.5%？** 观测 ≥5 分只有 198/12,500（1.58%），且品评噪声使个体高分不可精确识别（下界 0.6636）；固定阈值按训练分位提升，数量不可控（clove 84 格）。论文把它当"局限"而非卖点。
4. **Q2 有没有因果？** 没有，只做关联+图结构+证据分级；这是本题表述允许的范围（"关键因素化合物"）。
5. **图到底有没有用？** 消融：稳定性 +0.028 频率/+0.032 Jaccard，预测无显著提升（p=0.31）→ partial_gain。
6. **生成谱能酿吗？** 不能声称；它是统计参考，不是配方（§4.4）。
7. **迁移为什么 Tripel 只 1/11 判为 Tripel？** 风格分类器折外仅 0.544，且 logistic 与 ET 结论相反；迁移的定义是"到目标质心 + 缺陷不恶化"，风格判定是辅助证据，不作为成功判据。
8. **可复现吗？** 交付包含三层：`train`（需原始附件）、`predict`/`render`（只需模型与输入）、`package`+`verify --models --source --package`（全量重放 + 哈希校验）；本机实测 replay max|Δ|≈1e-16。

## 7. 复现与文件索引
```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe scripts\current.py status
.venv\Scripts\python.exe scripts\current.py train --source "<附件目录>"
.venv\Scripts\python.exe scripts\current.py predict --task q1 --model outputs\current_models\q1.joblib --input <1.csv> --output <out>
.venv\Scripts\python.exe scripts\current.py render --models outputs\current_models --source "<附件目录>" --output outputs\render_20261028_v6
.venv\Scripts\python.exe scripts\current.py package --rendered outputs\render_20261028_v6 --output deliverables\current_20261028
.venv\Scripts\python.exe scripts\current.py verify --models outputs\current_models --source "<附件目录>" --package deliverables\current_20261028
```

- 主张表：`paper_claims.csv`（33 条，逐条带证据文件）
- 数值表：`tables/`（q1__* / q2__* / q3__*）
- 图：`figures/`（4 组 PNG+SVG）
- 正式交付：`deliverables/current_20261028/`（398 文件；`conclusions.csv` 是纯文本汇总）
- 逐项修复记录：`docs/审查修复记录_20261028.md`

## 8. 变更记录（r1 → r2）
- Q1 主表：批次 top-k（MAE {f4(HIST['panel_median_mae'])}）→ 逐样本固定阈值（MAE {f4(DET['panel_median_mae'])}）；新增批次一致性 5/5、阈值回退清单、过度提升诊断；个体与下界数字同步更新。
- Q2：新增图邻域消融（partial_gain）、方法 Jaccard 与证据分级口径说明。
- Q3：生成器由插值改为通过 14/14 门槛的 v3；迁移按新 Q1 决策重跑（65 格变化、缺陷 worse 0）；新增风格分类器与 Tripel 冲突证据；感官画像改走正式决策层并锁模型 sha。
- 交付：包内 398 文件、verify 真重放通过、396 项单测通过。
"""
(OUT / "论文取数与口径.md").write_text(doc, encoding="utf-8")

readme = rf"""# 论文交接包（paper_handoff_20261028，r2）

给论文执笔者的取数包。**先读 `论文取数与口径.md`**，再用 `paper_claims.csv` 逐条引用。
本目录 r2 已按提交 `0a70627` 的正式口径重算，r1 数字全部作废。

## 内容
- `论文取数与口径.md`：三问主张、口径披露、三套 Q1 结果并排、Q2/Q3 数字、论文结构与 8 条答辩、复现命令。
- `paper_claims.csv`：33 条可写主张，每条带 `value` 与 `evidence` 文件。
- `tables/`：论文要引用的官方表（q1__*、q1_history__*、q2__*、q2_ablation__*、q3_generation__*、q3_migration__*、q3_evidence__*）。
- `figures/`：交付包 4 组图（PNG+SVG）。
- `verification.json`：本包的数字一致性检查（读取官方 outputs，断言关键值）。

## 三条必须遵守的写法
1. 0–6 标签来自外部 Zenodo 原始品评记录（2197 条），3.xlsx 是空白模板。
2. Q1 头条用"正式：逐样本固定阈值"那一行（0.4797/72.17%/89.62%），批次 top-k（0.4065）只作对照。
3. Q3 生成谱不是可酿造配方；Tripel 迁移后风格判定 1/11，属"风格支持不确定"。

## 重生成
```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe experiments\lead_review\build_paper_facts_20261028.py
```
"""
(OUT / "README.md").write_text(readme, encoding="utf-8")

# ----------------------------------------------------------------------------
checks = {
    "q1_det_panel_median_mae": DET["panel_median_mae"],
    "q1_hist_panel_median_mae": HIST["panel_median_mae"],
    "q1_et_panel_median_mae": ET["panel_median_mae"],
    "q1_batch_invariance_all_pass": bool(inv["pass_ok"].all()),
    "q1_batch_invariance_mismatch_sum": int(inv["mismatch_cells"].sum()),
    "q1_fallback_no_ge5": len(FALLBACK["no_ge5"]),
    "q1_fallback_no_6": len(FALLBACK["no_6"]),
    "q1_parent_pairs_postcap_violations": PAR["postcap_violations"],
    "q2_top3_rows": q2["n_top3_rows"],
    "q2_graph_communities": q2["graph"]["communities"],
    "q2_ablation_verdict": q2_verdict["verdict"],
    "q3_generation_gate_failures": len(q3g["gate_failures"]),
    "q3_migration_defect_worse": int(
        sum(v.get("worse", 0) for v in q3m["defect_status_counts"].values())
    ),
    "q3_migration_changed_cells": q3_det["changed_cells"],
    "q3_migration_sensory_mismatch": q3m["sensory_replay"]["mismatch_cells"],
    "paper_claims_rows": len(claims_df),
}
expected = {
    "q1_det_panel_median_mae": 0.47968,
    "q1_hist_panel_median_mae": 0.40648,
    "q1_et_panel_median_mae": 0.3928,
    "q1_batch_invariance_all_pass": True,
    "q1_batch_invariance_mismatch_sum": 0,
    "q1_fallback_no_ge5": 16,
    "q1_fallback_no_6": 38,
    "q1_parent_pairs_postcap_violations": 0,
    "q2_top3_rows": 150,
    "q2_graph_communities": 60,
    "q2_ablation_verdict": "partial_gain",
    "q3_generation_gate_failures": 0,
    "q3_migration_defect_worse": 0,
    "q3_migration_changed_cells": 65,
    "q3_migration_sensory_mismatch": 0,
    "paper_claims_rows": 33,
}
bad = {k: (checks[k], expected[k]) for k in expected if checks[k] != expected[k]}
report = {
    "status": "passed" if not bad else "failed",
    "checks": checks,
    "expected": expected,
    "mismatches": bad,
    "inputs": {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
        for p in [Q1D / "summary.json", Q1H / "summary.json", Q2 / "q2__summary.json",
                  Q2A / "summary.json", Q3G / "summary.json", Q3E / "summary.json",
                  Q3M / "summary.json"]
    },
    "outputs": {
        "论文取数与口径.md": sha(OUT / "论文取数与口径.md"),
        "paper_claims.csv": sha(OUT / "paper_claims.csv"),
    },
}
(OUT / "verification.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
)
print(json.dumps({k: report[k] for k in ("status", "mismatches")}, ensure_ascii=False))
print("claims:", len(claims_df), "tables:", len(list(TAB.glob("*"))), "figures:", len(list(FIG.glob("*"))))
