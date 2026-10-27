# MathorCup 2026 A：当前代码与交付

当前版本 **20261028，三问正式化版**。队友取数与转发统一使用 [deliverables/current_20261028](deliverables/current_20261028/README.md)，本机生成同名 ZIP 供直接转发。方法与入口以本页为准；历史记录见[待优化问题](docs/待优化问题.md)、[分阶段进度](docs/分阶段进度.md)、[阶段三说明](docs/阶段三交付说明.md)。

相对 20261027 的变化：Q1 增加决策层 top-k 与面板均值标定（原整数中位表保留为对照，不删）；Q2 从 150 条/50 维重建为图-Ridge 主表（覆盖 50/50 维）；Q3 迁移从"119 号单案例"升级为 11 款 × 3 风格的 33 条题面口径案例；Q3 增加 copula 生成器 v1/v2 各 300 条（v2 修好 v1 的残差结构缺陷并优于插值，但预注册门槛仍差 2 条 coverage，官方继续用插值谱，如实保留）。**这些是口径与覆盖的修正，不是模型质量提升**；个体级高分识别、Stale 证据、物理可酿造性等限制不变。

## 查看结果与论文取数

无需安装或重训。先读交付包 `conclusions.csv`、`unresolved.csv`、`current_constraints.csv`，再按 `file_index.csv` 找 `tables/`、`figures/`、`evidence/`。CSV 可直接在 Excel 打开，PNG/SVG 用于插图，不另造易过期的 Excel 副本。

| 内容 | 包内 `tables/` 文件 | 使用边界 |
|---|---|---|
| Q1 题面口径主表 | `q1_decision__q1__submission_long_12500.csv` / `q1_decision__q1__submission_wide_250x50.csv` | 250 款 × 50 个可评维度 = 12,500 格；top-k 决策，面板中位 MAE 0.4080、完全命中 73.88%、±1 91.31% |
| Q1 整数中位对照 | `q1_decision__q1__evaluation_long_12500.csv` 的 `grade_integer_median_old` 列 | 旧正式算法，面板中位 MAE 0.3928、完全命中 74.13%、±1 91.69%；高分 5/6 为 0 是取整截断 |
| Q1 高分与标定 | `q1_decision__q1__metrics_summary.csv`、`q1__high_grade_coverage_by_target.csv`、`q1__calibrated_panel_mean_12500.csv` | 5 分 204 格 精确 25.49%/召回 26.26%；6 分 16 格 6.25%/5.26%（仅 19 格，不稳）；面板均值折外标定 0.6123→0.5027→0.4622 |
| Q2 Top-3 主表 | `q2_top3__q2__top3_all50_150.csv` | 50/50 维、150 条、0 空位；共表达图 231 节点/645 边/60 社区；bootstrap 频率均值 0.474、Jaccard 0.361、图-Ridge vs Ridge Jaccard 0.600 |
| Q2 稳定性与证据 | `q2_top3__q2__stability_bootstrap.csv`、`q2__method_jaccard.csv`、`q2__evidence_tier_counts.csv` | 证据 A/B/C=8/70/72；弱信号维（diacetyl/dms/t2n/clove）标探索性 |
| Q3 生成谱 | `q3_generation__generated_profiles.csv`（copula v1，300 条）、`q3_generation_v2__generated_profiles.csv`（donor 残差 v2，300 条）、`history_q3_generation__generated_profiles.csv`（官方插值谱） | 每风格 100 条；v2 的 rank_rmse 0.0975 优于插值 0.1473、方差比 0.983、近重复 0，但 Kriek/Lambic coverage 仍差约 1 个谱级（门禁 11/14），**官方仍用插值谱**（方差比 0.726、近重复 26.9%）；预注册 "otherwise retain"，无 post-result retuning |
| Q3 迁移案例 | `q3_migration__s2c_migration_results.csv`（33 行）、`s2c_candidate_counts.csv`、`s2_source_defects_11beers.csv`、`s2_style_defect_exceedance.csv` | 题面口径 A/B/C 各 33/33、候选级 660/660；Tripel λ*=0.55、Kriek λ*=1.0；Lambic 迁移后仅 2/11 判为 Lambic（如实报） |
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
.venv/Scripts/python.exe scripts/current.py render --output outputs/my_render
```

输入列名/坐标与附件 1 一致，可含唯一 beer 列；缺失用训练填补，不能恢复测量，也不自动取指数。Q1 输出 52 项评分（`estimated_grades_all52.csv`）、校准前后风险及缺失情况；`--task q3` 显式输出辅助风险与风格，不当正式 Q1 或安全认证。`render` 的生成谱来源由 `current_runtime.generated_source()` 决定：按 `outputs/q3_copula_v2_20261028` → `outputs/q3_copula_20261028` → `outputs/task4_raw_q3_20261026` 顺序取第一个**通过采纳判定**（integrity + replacement gate）的目录；当前 copula 门槛未过，因此仍用官方插值谱，`render.json` 记录 `generated_source` 与 sha256。自选模型目录用 `--models`。

## 核验与打包

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -m unittest discover -s tests
.venv/Scripts/python.exe scripts/current.py verify --models outputs/current_models --source '原始附件目录' --output outputs/my_check
.venv/Scripts/python.exe scripts/current.py package --rendered outputs/my_render --output outputs/my_teammate_delivery
```

`verify --package 交付目录` 检查包内逐文件 sha256 与排除项；`--version 20261027` 可回到旧交付口径。已从零模型缓存重建固定模型并完成 Q1 12500 格、生成谱 15600 格、迁移 104 格与 Q3 辅助风险/风格的逐值对照；全量单元测试（含新增 Q1 决策层、Q2 Top-3、Q3 copula、Q3 迁移、delivery_current 测试）通过。实现检查不替代模型质量。

## 采用理由与限制

Q1 保留整数中位为对照、把 top-k 决策层作为题面口径主表：后者用概率排序补出 5/6 分格（204/16 格），代价是面板中位 MAE +0.0152；个体级高分的上限受品评噪声限制（面板中位 oracle 个体 MAE 仍 0.6636），不宣称解决。Q2 主表为图-Ridge，稳定性用 200 次按啤酒 bootstrap 与四方法 Jaccard 报告，非因果。Q3 生成保留官方插值谱，copula 作为未过门槛的候选如实交付；迁移是坐标空间内的统计投影，不是酿造配方，无迁移后酿造/品评实测。Stale 无化学坐标，按未知处理。9/12、22/32 等旧数字见历史表，不再作为当前结论。

题目要求、实现条件与额外保守情景见包内 `current_constraints.csv`；同源原论文不算外部验证。旧 `task6_final_20261026`、旧"交付"目录与旧日期文档是历史快照，不再作当前取数入口；原"交付"目录未修改。
