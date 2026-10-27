## 17. task-13：迁移交付闭环——完整谱、52 项感官画像、逐 case 证据分级（2026-10-28）

### 17.1 动机：审查问题清单第 5/6 项
`docs/审查问题清单_46ca932_20261027.md` 指出 Q3 迁移交付有两个真问题：
- **第 5 项（过度声称）**：33/33 只是化学判据 A/B/C 通过，风格诊断弱（Tripel 最近质心 0/11、Lambic 2/11，全库最近质心 LOO 仅 0.132 ≈ 多数类 0.124），不能写成"33 条完整迁移成功且五类缺陷安全"。
- **第 6 项（交付物不全）**：迁移算法内部有完整向量，但输出只存距离/判据/部分缺陷浓度；33 条完整谱与对应感官画像没有落盘；正式 render 仍固定读旧 119 号两谱（`Q3/source119_profile.csv`、`Q3/transfer119_profile.csv`，仅 104 行 = 2 谱 × 52 项）。
本节记录闭环结果：**33 条迁移前后完整 224 维谱已导出；66 条谱（33 前 + 33 后）已用正式 Q1 入口生成 52 项感官画像；每条 case 的证据分级与缺陷状态已落表；旧 119 号作为历史案例保留。**

### 17.2 新增产物（`outputs/q3_migration_20261028/`，全部真实跑出）
| 文件 | 形状 | 内容 |
|---|---|---|
| `s2c_migration_spectra_224_long.csv` | 66 × 229 | `case_id, source_beer, target_style, lambda, role∈{before,after}` + 完整 224 维官方特征谱；每 case 严格成对 |
| `s2c_migration_spectra_224_wide.csv` | 33 × 455 | meta + `dist_reduction_pct, style_before, style_after` + 448 列（224 个 `<feature>_before` / `<feature>_after`） |
| `s2c_migration_after_input_33.csv` | 33 × 227 | 迁移后谱的 Q1 输入表，列序与官方 `generated_profiles.csv` 一致：`generated_id, style, method` + 224 特征；`generated_id = S{beer:03d}_TO_{style}`，`method = q3_migration_projected` |
| `s2c_sensory_before_after_52.csv` | 3432 × 15 | 66 谱 × 52 项感官：`case_id, role, target_52, grade_raw, prob_ge1, prob_ge5, prob_6, raw_median_estimate, panel_target_available, score_status, risk_*, calibration_*, safety_certified` |
| `s2c_evidence_by_case.csv` | 33 × 36 | 每 case 证据分级（见 17.3） |
| `summary.json` / `summary.md` / `verification.json` | — | 增计数、claim_level/风格诊断/缺陷状态计数；verification **72 项全 True，status=passed** |
| 既有 4 张表 | — | `s2c_migration_results.csv`(33×~40)、`s2c_candidate_counts.csv`、`s2_source_defects_11beers.csv`、`s2_style_defect_exceedance.csv` 内容不变 |

完整 224 维特征集 = 迁移模块 231 列坐标减去官方排除列（`GFS, unfermentables, kcalperc, esters_sum, aroma_hops_sum, acids_sum, sulfur_sum`），已逐列断言与官方 generated_profiles 同序。

### 17.3 证据分级规则（写入 `protocol.json` 的 `evidence_grading`，跑之前写死）
- **缺陷状态** `defect_status`（逐缺陷）：以"超阈值项"为准，`e(c,T)=max(c/T-1,0)`；`e_after < e_before-1e-12 → improved`、`> +1e-12 → worse`、否则 `unchanged`；无阈值（trans-2-nonenal）或无坐标（Stale hops）→ `unknown`，且不参与 A/B/C。
- **风格诊断**：`style_diagnostic_supported = (nearest_centroid == target) and (target 风格 support 数 == 11/11)`；实际 support：**Tripel 0/11、Kriek 11/11、Lambic 2/11**。
- **claim_level**：
  - `undecidable`：A/B/C 任一不过（本次 0 条）；
  - `model_supported`：A/B/C 全过 **且** 最近质心=目标风格 **且** 该目标风格 support=11/11 **且** 无 worse 且无 conditional cap；
  - `conditional`：化学判据全过，但风格诊断未支持 / 目标非最近质心 / 存在条件性双乙酰 cap。

### 17.4 关键计数（`verification.json` / `summary.json`）
- **claim_level：`conditional` 24、`model_supported` 9、`undecidable` 0。** 即：**9 条（全部 Kriek 目标）可以说"化学判据 + 风格诊断双双支持"；24 条只能说"化学判据通过、风格诊断未支持"**——这正是问题 5 要求的收敛，不再写"33 条完整成功"。
- 按目标风格：Kriek 9 supported / 2 conditional；Tripel 11 conditional；Lambic 11 conditional。
- **style_nearest_centroid**：target 13（Kriek 11 + Lambic 2）、other 20、source_baseline 0。
- **缺陷状态计数**：DMS improved 9 / unchanged 24；Diacetyl improved 9 / unchanged 18 / unknown 6（119、28 两源 ×3 目标，源缺失）；4VG 全部 unchanged 33（含 119 的 0.1438→0.0405，两者均低于阈值 0.20，按"超阈值项"口径不计改善）；t2n unknown 33；Stale hops unknown 33；**defects_worse_count 合计 0**。
- 距离缩短（距目标风格质心）：Tripel 平均 12.496%（11.637–13.175）、Kriek 99.346%（99.286–99.507）、Lambic 81.165%（65.082–92.027）。
- 化学判据：A/B/C × median/mean 两情景全部 33/33；候选级 660/660；λ* Tripel ∈{0.50, 0.55}、Kriek/Lambic = 1.0。

### 17.5 示例（真实数值）
- `S119_TO_Tripel`（旧 119 号案例，`claim_level=conditional`）：DMS 16.661495→16.661495 µg/L（状态 unchanged，源未超阈）；Diacetyl NaN→0.081 mg/L（条件性 cap = min(目标中位, 0.9×0.09)，状态 unknown，**不宣称双乙酰安全**）；4VG 0.143755→0.040508 mg/L（状态 unchanged，两者均低于 0.20 阈值）；t2n 0.004838→0.004838 mg/L（unknown，无阈值）；Stale hops 无法评级（无化学坐标）。52 项画像中：`dms` 等级 1→0、`X4vg` 的 P(≥5) 0.015453→0.027592、`A_hops_all` 的 P(≥5) 0.021470→0.062775、`overall` 2→3。
- `S124_TO_Tripel`：chemical_pass_A=True、距离缩短 12.679%、最近质心 other、三项有阈值缺陷全 unchanged、global claim `conditional`——"化学安全但风格诊断不支持"的典型条目。

### 17.6 与旧 119 号案例的衔接
旧 `Q3/source119_profile.csv` + `Q3/transfer119_profile.csv`（2 谱 × 52）是历史单例，**保留不动**；新表把它吸收为 33 条 case 中的 `S119_TO_Tripel` 一行，并补齐完整 224 维谱、前后对比与 claim_level。正式 render 目前仍固定读旧 119 号文件；把 render 切到 33 条新表属于交付层改动（不在本任务写范围），已在回报中提请 lead 安排。

### 17.7 诚实表述（可直接进论文/答辩）
> 33 条迁移谱在预注册化学判据 A/B/C（有阈值的 DMS/双乙酰/4VG，按"超阈值项"口径）下全部不恶化；其中 9 条（Kriek 目标）同时获得风格诊断支持，24 条化学判据通过但风格诊断未支持（Tripel 0/11、Lambic 2/11），全部以 `conditional` 标注。trans-2-nonenal 与 Stale hops 因无阈值/无化学坐标，缺陷状态记为 unknown，不作"五类缺陷全部安全"的声明。52 项感官画像由全训练集 Q1 多任务模型给出，是模型推断而非新啤酒实测，也不是 OOS 精度估计。

### 17.8 复现命令（仓库根，Windows PowerShell）
```powershell
$env:PYTHONIOENCODING='utf-8'
$env:Q3_SOURCE='<附件目录，含 1.csv 与 8.xlsx>'
.venv\Scripts\python.exe scripts\q3_official_migration_20261028.py --source $env:Q3_SOURCE   # 2.7 s，72 项 checks 全 True
.venv\Scripts\python.exe -m unittest tests.test_q3_migration                                  # 32 tests OK（含 24 项数据相关）
```
感官画像由正式入口 `beer_models.inference.run(outputs/current_models/q1.joblib, <66 行输入>, <tmp 输出>)` 生成；输入/输出临时目录固定在 `experiments/lead_review/q3/tmp_q1_migration`、`tmp_q1_output`，每次运行前清空以保证可复现，不写入正式 outputs。
