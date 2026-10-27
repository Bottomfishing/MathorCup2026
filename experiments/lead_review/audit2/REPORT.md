# 独立第三方复核（task-14）：46ca932 九项审查问题整改验证

- 复核对象：`85b59cd Fix audit findings: nested calibration, capped decision, runtime decision, Q3 migration closure, verify replay, gitattributes`（整改提交）；前序被审提交 `46ca932`
- 复核人：q2-analyst（未参与 q1/q3 整改）
- 方式：只读仓库；所有证据由 `experiments/lead_review/audit2/` 下脚本从原始数据/产物独立重算，非引用整改者自述
- 环境：Windows，`.venv\Scripts\python.exe`，`PYTHONPATH=src`；仓库根 `C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件`
- 结论速览：**9 项中 7 项 fixed、2 项 partial**；另发现 1 个高优先级新问题（交付包"正式"全拟合表仍是 legacy 口径，与运行时决策层不一致）和 1 个中优先级新问题（Q3 迁移感官画像表无法由当前模型+谱复现）

| # | 审查问题 | verdict | 关键独立证据 |
|---|---|---|---|
| 1 | Q1 风格均值标定验证泄漏 | **fixed** | 折结构独立复算吻合；泄漏版→修复版 7166/12500 格变化 |
| 2 | top-k 决策未接入运行时 | **partial** | predict 已接入（0 不一致）；但包内"official"全拟合表仍是 legacy（192/12500 不一致） |
| 3 | 新 Q1 表破坏父子约束 12 处 | **fixed** | precap 12 → capped 0，仅降 12 格 |
| 4 | 高分四口径混写 | **fixed** | 三口径独立重算逐位一致；文档旧数字 0 处 |
| 5 | Q3"33/33"判据范围不足 | **fixed** | claim_level 24 conditional / 9 model_supported；Tripel 0/11、Kriek 11/11、Lambic 2/11 |
| 6 | 缺"完整谱→52 项画像"闭环 | **partial** | 表齐全且自洽；但交付画像表 57/3300 格不可由当前模型+谱复现 |
| 7 | verify 忽略 models/source 假通过 | **fixed** | 坏 models 路径 exit 1；有效组合 passed 且报 max\|Δ\| |
| 8 | 门槛图仍读 v1 | **fixed** | 交付 SVG 方法名 = v3 三个，无 copula_quantile |
| 9 | 交付 file_index 哈希不通过 | **fixed** | fresh clone 内 334 行哈希独立复算 0 不匹配 |

---

## 1) Q1 标定验证泄漏 — fixed

**代码路径（已读）**：`scripts/q1_official_20261028.py:321-374`。`nested_calibrations()` 对每个外层折只用 `train_beer_index`（不含本折验证啤酒）的 inner OOF 单元拟合 4 个校准器；若验证啤酒进入拟合集合，`scripts/q1_official_20261028.py:338-339` 直接 `raise RuntimeError('outer-validation beer leaked into the calibration fit')`。

**独立折结构复算**（不依赖整改产物）：`outputs/r2_raw_q1_20261026/rating_weights_and_folds.csv` 去重得 250 啤酒，外层折 84/83/83；由此训练啤酒 166/167/167，×50 个有标签目标 = 8300/8350/8350 个拟合单元，验证单元 4200/4150/4150。与 `outputs/q1_decision_20261028/summary.json::leakage_fix.calibration_fit_rows` 完全吻合（`{0: 8300/4200, 1: 8350/4150, 2: 8350/4150}`），即拟合单元里没有本折验证格。

**旁证（独立产物）**：`outputs/q1_panelmean_20261028/verification.json` → `all_pass: true`，其中 `baseline_calibration_is_nested`、`no_validation_beer_in_calibration_fit`、`fold_count_3`、`seed_per_fold_deterministic` 均为 true；诊断 `calibration_fit_beers: 166/167/167`、`inner_cells: 8300/8350/8350`、`inner_fits: 2`。

**泄漏版 vs 修复版差异表**（`outputs/q1_decision_20261028_pre_leakfix/q1__calibrated_panel_mean_12500.csv` vs 修复版）：12500 格中 7166 格改变（57.3%），max|Δ|=2.287；按折 2352/4200、2324/4150、2490/4150。`nested_style_linear` 面板均值 MAE 0.462234 → **0.466072**、中位 MAE 0.689469 → 0.697462、完全命中 0.475360 → 0.463440——修复后指标变差，方向与"去掉乐观泄漏"一致。

**残余限制（如实）**：本次未重跑 6 次内层 `fit_shared`（运行时长>10 分钟），扰动证据引用 `summary.json::leakage_fix`（beer `'2'`、delta 0.25、四方法 max|Δ|=0.0）并做上述独立折结构与差异表核对。verdict fixed。

复现：见附录 A 的折结构对照命令。

## 2) top-k 决策运行时接入 — partial（新问题 N1）

**predict 路径已接入（fixed 的一面）**：`scripts/current.py predict --task q1 --models outputs/current_models --input <附件>\1.csv` exit 0，250 样本 ×52 目标（50 有标签）。用 `probability_rating_ge5_y`/`probability_rating_6_y`/`grade_legacy_integer_median` 按 bundle 目标顺序独立调用 `beer_models.q1_decision.runtime_decision`：与 `estimated_grade_0_6` **0 格不一致 max|Δ|=0.0**，父子违规 **0**（24 对），bundle 声明的 `implementation_sha256 = e972cfe4…` 与 `src/beer_models/q1_decision.py` 实际 sha256 **完全一致**。

**但交付包"正式"全拟合表仍是 legacy（broken 的一面）**：`deliverables/current_20261028/tables/current__official_Q1_fullfit_scores52.csv`（13000 行 = 250×52）与 predict 决策结果在 **192/12500 有标签格不同（1.54%）**，而与 runtime 的 legacy 列 **0 格不同**。

- 包表分布 `{0:8146, 1:886, 2:1423, 3:1575, 4:438, 5:32, 6:0}`
- predict 决策分布 `{0:8132, 1:885, 2:1415, 3:1549, 4:305, 5:195, 6:19}`
- legacy 分布 `{0:8146, …, 5:32, 6:0}` = 包表分布

根因（代码已读）：`src/beer_models/current_runtime.py` 的 `render()` 写该表时走 `portrait()`（raw panel legacy 输出），未应用 `inference._official_grade_frame` 的决策层；而 `verify` 的 `_replay_checks` 恰是拿 `portrait()` 与该表比对（13000 行 max|Δ|=1.96e-15 通过）→ **verify 通过反而把 legacy 口径固化**。表名 `official` 与内容不符，是与 46ca932 审查第 2 项同性质的问题，未随整改彻底消除。

证据：`experiments/lead_review/audit2/check2_runtime.json`、`check2b_package_vs_runtime.json`。

## 3) 24 对父子约束 — fixed

从 `outputs/q1_decision_20261028/q1__evaluation_long_12500.csv` + bundle 的 24 对目标位置独立重算：

- precap 违规 **12**、cap 后违规 **0**、改动格 **12**、**只降不升**（`changed_cells_only_down: true`）
- 分布 `grade_0_6` = `{0:8279, 1:856, 2:1363, 3:1655, 4:153, 5:180, 6:14}`（与 summary 一致）
- precap 分布 `{0:8279, 1:856, 2:1361, 3:1648, 4:152, 5:188, 6:16}`（= 旧 5 分 188、6 分 16）

证据：`experiments/lead_review/audit2/check3_4_calibers.json::parent_child`。

## 4) 高分四口径 — fixed

**beer-level**（用 `grade_0_6` + `panel_median_reference` 独立重算，n=12500）：

| 口径 | n_pred | true_positive | precision | n_observed | recall |
|---|---|---|---|---|---|
| exact5 | 180 | 38 | 0.211111 | 179 | 0.212291 |
| exact6 | 14 | 1 | 0.071429 | 19 | 0.052632 |
| ge5 | 194 | 51 | 0.262887 | 198 | 0.257576 |

与 `outputs/q1_decision_20261028/q1__metric_calibers.csv`（beer_level cap 行）逐位一致，亦与 46ca932 审查要求的目标值一致。

**individual-taster weighted**：从 `data/reference/zenodo23624915/Raw sensory data anonymized.csv`（cp1252，109850 条）+ `outputs/r2_raw_q1_20261026/rating_weights_and_folds.csv`（weight_sum=12500、unmapped=0）独立加权重算：exact5 180/41.6829/0.231572/524.989837/0.079398；exact6 14/2.195887/0.156849/124.847615/0.017589；ge5 194/70.210714/0.361911/649.837452/0.108044 —— 与交付 individual 行逐位一致。

**文档核对**：`README.md`、`docs/待优化问题.md` 中旧混写数字（204、25.49、26.26、188、20.74、21.79、16、1、6.25）出现次数 **0**；新数字（180、21.11、21.23、194、26.29、25.76、14、1、7.14、5.26）均存在。

证据：`check3_4_calibers.json::beer_level / individual_weighted / docs_scan`。

## 5) Q3 33/33 判据范围与如实标注 — fixed（标注层）

结构全部齐全：`s2c_migration_spectra_224_long.csv` 66 行（before 33 + after 33）×224 特征（与 bundle `features` 224 名称完全匹配）；`s2c_migration_after_input_33.csv` 33×227；`s2c_sensory_before_after_52.csv` **3432 行 = 33 case × 2 role × 52 目标**（before 1716 / after 1716）；`s2c_evidence_by_case.csv` 33 行×36 列。

如实性核对：

- `claim_level`：conditional **24**、model_supported **9**、undecidable **0**（与 verification 一致；9 个 model_supported 全为 Kriek）
- 风格诊断（每风格 11 源啤酒）：Tripel **0/11**、Kriek **11/11**、Lambic **2/11**
- 缺陷状态：DMS unchanged 24/improved 9；Diacetyl unchanged 18/improved 9/unknown 6；4VG unchanged 33；t2n unknown 33；Stale_hops unknown 33；**defects_worse_total=0**
- before 谱 = 附件 1.csv（11 个唯一 source beer）max|Δ|=0.0；after 谱 = after_input max|Δ|=0.0
- migration `verification.json` claims 与表逐项一致（`verification_claims_match_table: true`）

证据：`check56_q3.json::structure / claim_level_counts / style_diagnostics / defect_status_counts`。

## 6) 完整谱→52 画像闭环 — partial（新问题 N2）

表齐全（见第 5 项），但**交付画像表不能由当前模型 + 谱复现**：

- 用 `beer_models.inference.run(outputs/current_models/q1.joblib, 66 谱)` 独立重放，与交付 `s2c_sensory_before_after_52.csv` 的 3300 个有标签格比较：**57 格不一致（1.7%），max|Δ|=5**（after 31、before 26，涉及 18 个 case；如 S209_TO_Tripel after F_malt_all 交付 2 vs 重放 5）
- 又用官方脚本原函数 `scripts/q3_official_migration_20261028.py::build_sensory_profiles`（importlib，`data.official_features = bundle['features']`）复现，得到**同样的 57 格不一致**（另 132 格为 2 个无标签 Acetaldehyde 的 NaN）
- `outputs/q3_migration_20261028/protocol.json::sensory_pipeline` 只记录模型路径，**未记录 q1.joblib 的 sha256**；实际模型 sha `3312c314bf26ca8f5f08b995ba0a1a8b701db23ed3b3359433890e95a5b9c3ae` 不在 protocol 中
- `git log` 显示 `85b59cd` 与 `6d00d06` 都改动过 `outputs/current_models/q1.joblib` 与 `outputs/q3_migration_20261028`，与"画像表基于旧模型/旧谱生成且未重生成"一致

即：闭环在"表结构+标注规则"层面 fixed，在"可复现"层面 partial。建议：重生成该表并把 model sha256 写进 protocol，verify 增加该表的逐格 replay。

证据：`check56_q3.json::sensory_replay`、`experiments/lead_review/audit2/q3_sensory_replay/`、`tmp_official_replay/`。

## 7) verify 真重放 — fixed（但有盲区，见 N3）

- **有效组合**：`python scripts/current.py verify --models outputs/current_models --source <附件> --package deliverables/current_20261028 --output experiments/lead_review/audit2/verify_full` → `status: passed`，exit 0；`q1_replay_panel_rows=13000, max|Δ|=1.96e-15`；生成谱 15600 行 `max|Δ|=5.34e-16`；models sha `3312c314…`、source sha `419849…`；`package_files_hashed=334`
- **坏 models 必须失败**：`--models outputs/current_models_nonexistent` → `FileNotFoundError: outputs\current_models_nonexistent\q1.joblib`，**exit 1**
- **package-only**：`current.py verify --package deliverables/current_20261028` → exit 0，334 文件哈希

盲区：`_replay_checks` 只重放 `portrait()`（legacy 层）并与同为 legacy 的包表比对，故无法发现第 2 项/第 6 项的"表与决策层/模型不一致"。证据：`verify_full/verification.json`、`verify_bad/`、`verify_package/`。

## 8) 门槛图方法名来自 v3 — fixed

- `outputs/q3_copula_v3_20261028/generation_validation.csv` 方法集合 = `[donor_residual_projected, hybrid_support_projected, interpolate]`
- v1 `outputs/q3_copula_20261028/generation_validation.csv` = `[copula_quantile_projected, interp_quantile_projected, interpolate]`
- 交付 `deliverables/current_20261028/figures/Q3_生成门槛.svg` 含 `hybrid_support_projected`，**不含** `copula_quantile`；图中方法 token 集合 = v3 三个

证据：`check56_q3.json::gate_figure`。

## 9) fresh clone + file_index 哈希 — fixed

- `git clone` 仓库到 `C:\Users\wb198\Desktop\mc_audit_clone`，clone HEAD = `85b59cd`
- clone 内 `python scripts/current.py verify --package deliverables/current_20261028 --output .../verify_clone` → `status: passed`，`package_files_hashed=334`，exit 0
- **独立复算** `deliverables/current_20261028/file_index.csv`：334 行全部存在、sha256 **0 不匹配**
- `.gitattributes:22` 存在 `deliverables/current_20261028/** -text` 规则

证据：`verify_clone/verification.json`、附录 B 命令。

---

## 新问题（单列）

### N1（高）交付包"正式"Q1 全拟合表仍是 legacy 口径，verify 检测不到
- 现象：`current__official_Q1_fullfit_scores52.csv` 的 `estimated_grade_0_6` 列 = legacy ET 中位数（无 6 分、5 分仅 32），与 `current.py predict` 决策结果差 192/12500 格（1.54%）
- 根因：`src/beer_models/current_runtime.py::render` 用 `portrait()` 写"official"表，未走 `inference._official_grade_frame`；`delivery_current._replay_checks` 也以 `portrait()` 为准，故 verify 全绿
- 建议：要么让 `render` 输出决策后结果并重生成包表（同时改 replay 基准），要么把该表改名/加列明确标注 legacy，并在 README/conclusions 中说明差异

### N2（中）Q3 迁移 52 项画像表不可复现，且 protocol 未锁模型哈希
- 57/3300 格不一致（max|Δ|=5），官方函数复现结果相同 → 表是旧模型/旧谱的产物
- `protocol.json` 未记录 q1.joblib sha256；verify 未 replay 该表
- 建议：用当前 q1.joblib 重生成 `s2c_sensory_before_after_52.csv`，protocol 记录模型 sha，verify 增补该表逐格 replay

### N3（中，与 N1/N2 同源）verify 的重放基准选错层级
- `_replay_checks` 以 `portrait()`（legacy）与包表比对，而不是决策后产物；因此"verify passed"不能证明交付物就是运行时 predict 会给出的结果
- 建议：replay 直接调用 `inference.run` 的对外输出（含决策层）作为基准，并纳入 Q3 画像表

### N4（低）`outputs/current_models/q1.joblib` 在整改提交中被替换过，历史产物未全部重生成
- `85b59cd` 与 `6d00d06` 都改过 q1.joblib 与 q3_migration 输出；建议全量重跑依赖该模型的正式产物（含 q1 面板均值、Q3 画像表）并重新 package

---

## 附录 A：复现命令（仓库根，PowerShell）

```powershell
$env:PYTHONPATH='src'
# 复核 2：运行时规则接入（本轮曾以 py 脚本重算；结果见 check2_runtime.json）
.venv\Scripts\python.exe scripts/current.py predict --task q1 --models outputs/current_models --input 'C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件\1.csv' --output experiments/lead_review/audit2/runtime_predict
# 复核 2 的包表 vs 运行时比较
.venv\Scripts\python.exe experiments/lead_review/audit2/check2b_package_vs_runtime.py
# 复核 3/4
.venv\Scripts\python.exe experiments/lead_review/audit2/check3_4_calibers.py
# 复核 5/6/8
.venv\Scripts\python.exe experiments/lead_review/audit2/check56_q3.py
# 复核 7
.venv\Scripts\python.exe scripts/current.py verify --models outputs/current_models --source 'C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件' --package deliverables/current_20261028 --output experiments/lead_review/audit2/verify_full
.venv\Scripts\python.exe scripts/current.py verify --models outputs/current_models_nonexistent --source 'C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件' --package deliverables/current_20261028 --output experiments/lead_review/audit2/verify_bad
# 复核 1 的折结构与 fit_rows 对照
.venv\Scripts\python.exe -c "import pandas as pd,json;w=pd.read_csv('outputs/r2_raw_q1_20261026/rating_weights_and_folds.csv');print(w[['beer','outer_fold']].drop_duplicates().outer_fold.value_counts().to_dict());print(json.load(open('outputs/q1_decision_20261028/summary.json',encoding='utf-8'))['leakage_fix']['calibration_fit_rows'])"
```

## 附录 B：fresh clone 复核

```powershell
cd C:\Users\wb198\Desktop; git clone -q 'C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件' mc_audit_clone; cd mc_audit_clone
git rev-parse --short HEAD   # 85b59cd
$env:PYTHONPATH='src'
& 'C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件\.venv\Scripts\python.exe' scripts/current.py verify --package deliverables/current_20261028 --output 'C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件\experiments\lead_review\audit2\verify_clone'
# 独立哈希复算：334 行、0 不匹配（脚本内联，见 REPORT 正文）
```

## 附录 C：本轮新增文件（仅 `experiments/lead_review/audit2/`）

- `check2_runtime.py` / `check2_runtime.json`
- `check2b_package_vs_runtime.py` / `check2b_package_vs_runtime.json`
- `check3_4_calibers.py` / `check3_4_calibers.json`
- `check56_q3.py` / `check56_q3.json`
- `runtime_predict/`、`verify_full/`、`verify_bad/`、`verify_package/`、`verify_clone/`、`q3_sensory_replay/`、`tmp_official_replay/`
- `REPORT.md`（本文件）

## 复核局限（如实声明）

1. 第 1 项未重跑 6 次内层模型拟合（成本>10 分钟）；以代码路径、独立折结构复算、泄漏版/修复版差异表为证据，官方扰动测试数值（max|Δ|=0.0）为自述引用。
2. 第 2 项中"192 格差异"依赖 `current.py predict` 在附件 1.csv 上的输出；该输出与包表使用的输入一致（同一 1.csv、同一 q1.joblib sha）。
3. 第 6 项判定"不可复现"基于官方函数 + 当前模型/谱两种独立复现路径均得同一 57 格差异；不排除交付表生成时的谱/模型与仓库当前版本不同（正因如此才判 partial）。
