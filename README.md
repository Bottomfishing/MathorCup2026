# MathorCup 2026 A：当前代码与交付

当前版本 **20261028，三问正式化版**。队友取数与转发统一使用 [deliverables/current_20261028](deliverables/current_20261028/README.md)，本机生成同名 ZIP 供直接转发。方法与入口以本页为准；历史记录见[待优化问题](docs/待优化问题.md)、[分阶段进度](docs/分阶段进度.md)、[阶段三说明](docs/阶段三交付说明.md)。

相对 20261027 的变化：Q1 决策层改为**逐样本固定阈值**（批次无关：整批/逐条/分块同值；历史批次 top-k 表降为 `q1_decision_history` 对照），并保留面板均值嵌套标定与直接回归（未过预注册门槛，不采纳）证据；Q2 从 150 条/50 维重建为图-Ridge 主表（覆盖 50/50 维），补参数敏感性证据包（阈值/分辨率网格）与图邻域消融（只提升稳定性、不提升预测）；Q3 迁移从"119 号单案例"升级为 11 款 × 3 风格的 33 条题面口径案例；Q3 生成器迭代到 v3 混合谱，通过预注册 14/14 替换门槛，**官方生成谱已由 v3 替换插值谱**（v1/v2 与插值谱保留为诊断与历史基线）；补风格分类器折外证据（ET 准确率 0.544）与 Tripel 缺陷安全冲突诊断。**这些是口径、覆盖与生成质量的修正，不是个体级预测质量的提升**；个体级高分识别、Stale 证据、物理可酿造性等限制不变。Q1 标定改为严格嵌套（防泄漏：扰动测试 max|Δ|=0），旧 0.5027/0.4622 已作废。正式决策表分布 {0:8096,1:856,2:1355,3:1667,4:146,5:321,6:59}，面板中位 MAE 0.4797；与历史批次 top-k（0.4065、194/180/14）相比，固定阈值把 ≥5/6 预测从 194/14 提到 380/59 但精确率下降（9.47%/5.08%），批次无关性的代价如实写。面板均值直接回归（ET 0.4519 vs 嵌套标定基线 0.4661，改善 0.0142，95% [0.0082,0.0197]）因 round 后 ±1 0.91136 低于预注册绝对下限 0.9119（差 0.05pt，gate4 未过）按协议不采纳；门槛不事后放宽。

## 查看结果与论文取数

无需安装或重训。先读交付包 `conclusions.csv`、`unresolved.csv`、`current_constraints.csv`，再按 `file_index.csv` 找 `tables/`、`figures/`、`evidence/`。CSV 可直接在 Excel 打开，PNG/SVG 用于插图，不另造易过期的 Excel 副本。

| 内容 | 包内 `tables/` 文件 | 使用边界 |
|---|---|---|
| Q1 题面口径主表（正式） | `q1_decision__q1__submission_long_12500.csv` / `q1_decision__q1__submission_wide_250x50.csv` | 250 款 × 50 个可评维度 = 12,500 格；逐样本固定阈值决策（父子 cap 后），面板中位 MAE 0.4797、完全命中 72.17%、±1 89.62% |
| Q1 整数中位对照 | `q1_decision__q1__evaluation_long_12500.csv` 的 `grade_integer_median_old` 列 | 旧正式算法，面板中位 MAE 0.3928、完全命中 74.13%、±1 91.69%；高分 5/6 为 0 是取整截断 |
| Q1 历史批次决策（对照） | `q1_decision_history__q1__submission_long_12500.csv`、`q1__metrics_summary.csv` | 20261028 前批次 top-k（依赖整批行数）：MAE 0.4065、194 个 ≥5、180 个 5、14 个 6；仅作历史对照，不再作部署口径 |
| Q1 高分与标定（四口径） | `q1_decision__q1__metrics_summary.csv`、`q1__metric_calibers.csv`、`q1__batch_invariance.csv`、`q1__high_grade_coverage_by_target.csv`、`q1__calibrated_panel_mean_12500.csv` | 预测≥5 分 380 格 精确 9.47%/召回 18.18%；恰好 5 分 321 格 7.17%/12.85%；恰好 6 分 59 格 5.08%/15.79%（观测仅 19 格，不稳）；批次一致性 5/5 场景 mismatch=0；父子 24 对 0 违规；面板均值嵌套标定 0.6123→0.5055→0.4661 |
| Q1 面板均值直接回归（诊断） | `q1_panelmean__q1__panel_mean_metrics.csv`、`q1__panel_mean_pred_12500.csv` | ET 直接回归 0.4518585 vs 嵌套标定基线 0.4660719（改善 0.0142，95% [0.0082,0.0197]，g1-g3 过）；round 后 ±1 0.91136 < 预注册下限 0.9119（g4 未过，差 0.05pt）→ 按协议不采纳；HGB 0.4691 全门不过；ET 宏 Spearman 0.4145 vs 基线 0.3540 |
| Q2 Top-3 主表 | `q2_top3__q2__top3_all50_150.csv` | 50/50 维、150 条、0 空位；共表达图 231 节点/645 边/60 社区；bootstrap 频率均值 0.474、Jaccard 0.361、图-Ridge vs Ridge Jaccard 0.600 |
| Q2 稳定性与证据 | `q2_top3__q2__stability_bootstrap.csv`、`q2__method_jaccard.csv`、`q2__evidence_tier_counts.csv` | 证据 A/B/C=8/70/72；弱信号维（diacetyl/dms/t2n/clove）标探索性 |
| Q2 参数敏感性 | `q2_sensitivity__q2__threshold_sweep.csv`、`q2__resolution_sweep.csv`、`q2__variant_overlap.csv` | 阈值 0.45/0.50/0.55 Top-3 Jaccard 0.770/1.000/0.768；分辨率 0.8/1.0/1.2 → 59/60/60 社区、最大 33/29/29；默认参数与主表逐行一致 → 主结论不是调参产物 |
| Q3 生成谱（官方） | `q3_generation__generated_profiles.csv`（v3 混合谱，300 条）、`generation_validation.csv`、`generation_gate.csv`、`defect_safety.csv` | 每风格 100 条；v3 rank_rmse 0.1361、方差比 0.840、近重复 7.2%、marginal W1 0.3851、coverage 0.9488；预注册 14/14 门槛通过 → 替换官方插值谱 |
| Q3 生成谱（诊断） | `q3_generation_v2__generated_profiles.csv`（donor 残差 v2）、`q3_generation_v1__generated_profiles.csv`（copula v1）、`history_q3_generation__generated_profiles.csv`（插值谱） | v2 rank_rmse 0.0975/方差比 0.983/近重复 0 但 Kriek/Lambic coverage 差约 1 谱级（11/14）；v1 10/14；插值谱方差比 0.726/近重复 26.9%；三者仅作诊断，不是官方谱 |
| Q3 迁移案例 | `q3_migration__s2c_migration_results.csv`（33 行）、`s2c_candidate_counts.csv`、`s2_source_defects_11beers.csv`、`s2_style_defect_exceedance.csv` | 题面口径 A/B/C 各 33/33、候选级 660/660；Tripel λ*=0.55、Kriek λ*=1.0；Lambic 迁移后仅 2/11 判为 Lambic（如实报） |
| Q3 迁移闭环（新增） | `q3_migration__s2c_migration_spectra_224_long.csv`（66 条前后谱）、`s2c_migration_after_input_33.csv`（33×227）、`s2c_sensory_before_after_52.csv`（3432 行）、`s2c_evidence_by_case.csv`（33 行）、`s2c_det_grade_changes.csv`（65 行） | 33 条迁移的前后完整 224 维谱 + 迁移后可直接重算的输入表 + 正式 Q1 入口的 52 项画像（66 谱）；claim_level model_supported 9/conditional 24，Tripel 判定 0/11 标 conditional；t2n/Stale 保持 unknown；Q1 决策切换后重打分 65/3300 格变化（升 19/降 46），五缺陷项 0 worse |
| Q2 图邻域消融（新增） | `q2_ablation__q2_ablation_summary.json`、`q2__variant_overlap.csv` 等 | 图(w=0.5) vs 无图(w=0)：Top-3 Jaccard 0.600（13/50 维相同、45/50 维 ≥0.5）；稳定性频率 +0.028、Jaccard +0.032；预测 MAE 增益 +0.0011（p=0.31，不显著）→ verdict=partial_gain：图只提升稳定性，不提升预测 |
| Q3 风格与冲突证据（新增） | `q3_evidence__summary.json`、`q3_style_classifier_*`、`q3_tripel_conflict_*` 等表 | 风格分类器按啤酒折外 ET 0.544/logistic 0.512/最近质心 0.224（旧"0.132 vs 0.124 不可辨识"过强）；Tripel 训练样本 0/28 满足三缺陷 cap（质心 DMS 2.03x、双乙酰 6.27x）；缺陷安全参照距离仅原始质心 1.0005x、11/11 达 DMS cap → 冲突源于典型 Tripel 本身超标 |
| Q3 两种画像 | `current__transfer_official_Q1_scores52.csv`、`current__transfer_auxiliary_Q3_risks.csv` | 正式 Q1 的 224 输入评分与 Q3 辅助 223 输入概率分开；双乙酰 NA 保持未知 |

`history_` 前缀的表来自 20261027 交付，保留作历史对照与可追溯性，不再作为当前口径。

## 数据与环境

Python **3.12**；`requirements-r3-lock.txt` 含核心与 SHAP 锁定依赖，无新增依赖。原附件保留仓库外，本机路径为 `C:/Users/wb198/Desktop/2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题/2026年MathorCup大数据挑战赛-赛道A初赛/附件`。固定训练需要附件 1.csv、4.csv、8.xlsx，校验原审计哈希；原始评分可从固定公开版本获取或用 `--raw` 指定本地文件。

实际 250 款啤酒、226 种测量属性加 5 个派生字段共 231 列；正式 Q1 排除 7 个汇总字段使用 224 列。2197 条原始记录覆盖 50 目标；3.xlsx 为空白评分模板，0–6 标签来自外部 Zenodo raw（DOI 10.5281/zenodo.23624915），论文与交付必须披露。评分卡 52 项的两项乙醛无标签，分数及概率始终留空。原始附件、逐品评员记录、模型二进制均不上传。

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-r3-lock.txt
.venv/Scripts/python.exe scripts/fetch_raw_sensory.py --no-proxy
$env:PYTHONPATH='src'
.venv/Scripts/python.exe scripts/current.py status
```

## 训练、预测与刷新

从仓库根目录运行。每个输出目录须不存在，默认模型目录 `outputs/current_models/`。

```powershell
.venv/Scripts/python.exe scripts/current.py train --source '原始附件目录'
.venv/Scripts/python.exe scripts/current.py predict --input '新化学谱.csv' --output outputs/my_prediction
.venv/Scripts/python.exe scripts/current.py render --source '原始附件目录' --output outputs/my_render
```

输入列名/坐标与附件 1 一致，可含唯一 beer 列；缺失用训练填补，不能恢复测量，也不自动取指数。Q1 输出 52 项评分（`estimated_grades_all52.csv`）、校准前后风险及缺失情况；`--task q3` 显式输出辅助风险与风格，不当正式 Q1 或安全认证。`render` 的生成谱来源由 `current_runtime.generated_source()` 决定：按 `outputs/q3_copula_v3_20261028` → `outputs/q3_copula_v2_20261028` → `outputs/q3_copula_20261028` → `outputs/task4_raw_q3_20261026` 顺序取第一个**通过采纳判定**（integrity + replacement gate）的目录；当前 v3 已通过，故官方画像使用 v3 谱，`render.json` 记录 `generated_source` 与 sha256。自选模型目录用 `--models`。

## 核验与打包

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -m unittest discover -s tests
.venv/Scripts/python.exe scripts/current.py verify --models outputs/current_models --source '原始附件目录' --package deliverables/current_20261028 --output outputs/my_check
.venv/Scripts/python.exe scripts/current.py package --rendered outputs/my_render --output outputs/my_teammate_delivery
```

`verify --package 交付目录` 检查包内逐文件 sha256 与排除项（当前 398 文件）；`verify --models --source --package` 会真正重放：用 `q1.joblib`（当前 sha `82cf7496…`，固定阈值版）对 1.csv 与 adopted 生成谱重算官方画像，与包内参照逐格比对（要求 max|Δ|≤1e-12；实测 panel 12500 行 ≈1e-16、generated 15000 行 ≈1e-16），并校验 Q3 迁移感官表 `sensory_replay.mismatch_cells=0` 与模型 sha、迁移 det 重跑 `defect_worse_count=0`，缺参数即报错而不是静默通过；部署入口对无固定阈值决策层的旧 bundle 显式报错（`legacy_decision=True` 才能回放历史批次规则）。`--version 20261027` 可回到旧交付口径。全量单元测试 396 项（25 skip）。实现检查不替代模型质量。

## 采用理由与限制

Q1 正式决策改为逐样本固定阈值（阈值只由训练折概率分位估出）：批次无关（整批/逐条/分块同值，5/5 场景 mismatch=0），可部署；代价是面板中位 MAE 0.4797（历史批次 top-k 0.4065、整数中位对照 0.3928），≥5 预测由 194 提到 380、6 分由 14 提到 59，但精确率降到 9.47%/5.08%，这是阈值无法在单请求内约束提升数的直接后果，如实写。历史批次表保留为 `q1_decision_history` 对照。个体级高分的上限受品评噪声限制（面板中位 oracle 个体 MAE 仍 0.6636），不宣称解决。面板均值直接回归（ET 0.4519 vs 嵌套基线 0.4661）因 round 后 ±1 0.91136 低于预注册下限 0.9119 按协议不采纳（门槛不 post-hoc 放宽）。Q2 主表为图-Ridge，稳定性用 200 次按啤酒 bootstrap 与四方法 Jaccard 报告，另以阈值 0.40–0.60、分辨率 0.6–1.6 网格证明主结论不敏感，并用图邻域消融量化图的真实作用（稳定性 +0.028、预测增益不显著）；非因果。Q3 生成器 v3（插值内部分量 + donor 残差 bootstrap + 支撑投影）以同一预注册门槛 14/14 通过并替换官方插值谱：rank_rmse 0.1361（插值 0.1473）、方差比 0.840（插值 0.726）、近重复 7.2%（插值 26.9%）、coverage 0.9488；v1/v2 作为诊断保留。风格分类器折外 ET 0.544/最近质心 0.224 → 风格部分可辨识但不足以证明迁移成功；Tripel 冲突诊断为"典型样本本身超标"（0/28 达标），而非目标不可达。迁移是坐标空间内的统计投影，不是酿造配方，无迁移后酿造/品评实测。Stale 无化学坐标，按未知处理。9/12、22/32 等旧数字见历史表，不再作为当前结论。

题目要求、实现条件与额外保守情景见包内 `current_constraints.csv`；同源原论文不算外部验证。旧 `task6_final_20261026`、旧"交付"目录与旧日期文档是历史快照，不再作当前取数入口；原"交付"目录未修改。
