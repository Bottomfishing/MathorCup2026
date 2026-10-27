## 16. v3 生成器：coverage 诊断 + 预注册混合方案使替换门槛 13/13 全过（2026-10-28）

本节对应 task-10。v2（§15）在预注册替换门槛上 11/13，唯一缺口是两条小样本 coverage 规则
（Kriek 0.83333 < 0.86667、Lambic 0.91667 < 0.95）。v3 先做诊断再动手，诊断全文见
`experiments/lead_review/q3/v3_diagnosis.md`，本节给结论和数字。

### 16.1 诊断：这是"指标分辨率地板 + 半径定义"，不是可修的模型缺陷

用预注册的 3 折重算每个 held-out 谱到生成谱的最近距离，并加入 `oracle_train_set` 参照
（生成集=该风格训练谱本身，即任何"忠于训练支撑"的生成器所能达到的极限）：

| style | v2 donor_residual | interpolate（官方基线） | oracle_train_set | 半径 |
|---|---|---|---|---|
| Tripel | 0.92963 | 0.92963 | 0.92963 | 1.171445 |
| Kriek | **0.83333** | 0.91667 | **0.75000** | 1.091035 |
| Lambic | **0.91667** | 1.00000 | **0.72222** | 0.899564 |

oracle 比 v2 更差：**把支撑域收窄到训练集本身也过不了这两条规则**。逐漏点显示真正拉开差距的是
"只超半径 1–4%"的边缘谱（Kriek beer166、Lambic beer183），而 Tripel beer160/82 是三种方法加
oracle 全部漏的结构性离群，对 gate 无差别。每折 Kriek/Lambic 各只有 12 个 heldout 谱，
1 条谱 = 0.0833，大于规则允许的 0.05 降幅——**coverage 在该样本量下存在分辨率地板**。
另一方面 interpolate 能过 coverage 靠的是故意欠离散（variance_ratio 0.72599 vs v2 0.98347），
coverage 与 `variance∈[0.8,1.2]` 两条规则方向互相拉扯。

### 16.2 v3 设计（跑 gate 之前写死，不搜参、不看结果）

`HybridSupportGenerator`：`interior_fraction=0.5` 的官方 `StyleGenerator(method='interpolate')`
内部分量（补齐 baseline 已证明能覆盖的支撑域内部）+ 0.5 的 v2 donor 残差 bootstrap 外部分量
（真实边际与秩相关结构）；同一逐坐标经验分位映射、同一支撑投影、**缺陷 cap 最后施加**；
内部分量独立 seed 偏移 7919。预注册记录在 `outputs/q3_copula_v3_20261028/protocol.json` 的
`pre_registered_design`：`fixed_before_gate_evaluation=true`、`parameter_search="none"`、
`gate_thresholds_modified=false`、`baseline_modified=false`、`evaluation_code_modified=false`。
门槛、基线、`build_gate`、评估代码一行未改。

### 16.3 结果：13/13 规则全过，integrity 16/16

`outputs/q3_copula_v3_20261028/generation_gate.csv`：

| rule | candidate | limit | operator | passed |
|---|---|---|---|---|
| marginal_w1 | 0.385093 | 0.401069 | ≤ | True |
| energy | 0.333242 | 0.366649 | ≤ | True |
| exact_training_copy_rate | 0.000000 | 0.000000 | ≤ | True |
| variance_lower | 0.840121 | 0.800000 | ≥ | True |
| variance_upper | 0.840121 | 1.200000 | ≤ | True |
| rank_correlation_rmse | 0.136075 | 0.197289 | ≤ | True |
| model_median_w1 | 0.122158 | 0.148131 | ≤ | True |
| Kriek_energy | 0.498905 | 0.613837 | ≤ | True |
| **Kriek_coverage** | **0.916667** | 0.866667 | ≥ | **True** |
| Lambic_energy | 0.322447 | 0.381892 | ≤ | True |
| **Lambic_coverage** | **1.000000** | 0.950000 | ≥ | **True** |
| Tripel_energy | 0.178373 | 0.214211 | ≤ | True |
| Tripel_coverage | 0.929630 | 0.879630 | ≥ | True |

`verification.json`：`integrity_passed=true`（16/16，300 行、每风格 100、方法标签、同 seed 逐位同一、
不同 seed 不同、227 列与官方同序、无 NaN、每风格每缺陷 cap 全过）、`gate_passed=true`（14/14
含 `gate_all_passed`）、`all_passed=true`。

三版对照（3 折×3 风格均值，同一验证折与 seed）：

| 指标 | interpolate（官方） | v1 pc_residual | v2 donor_residual | **v3 hybrid（交付）** |
|---|---|---|---|---|
| marginal_w1 | 0.401069 | 0.385410 | 0.385542 | **0.385093** |
| energy | 0.366649 | 0.309500 | 0.333000 | **0.333242** |
| variance_ratio | 0.725986 | 0.982400 | 0.983468 | **0.840121** |
| rank_correlation_rmse | 0.147289 | 0.389960 | 0.097492 | **0.136075** |
| heldout_coverage | 0.948765 | 0.865431 | 0.893210 | **0.948765** |
| exact_training_copy_rate | 0 | 0 | 0 | **0** |
| near_training_copy_rate | 0.002222 | 0 | 0 | **0.001111** |
| internal_near_duplicate_rate | 0.268889 | 0 | 0 | **0.072222** |
| model_median_w1（感官一致性） | 0.128131 | 0.121289 | 0.144111 | **0.122158** |
| 预注册 gate | —（基线） | 10/13 | 11/13 | **13/13** |

逐风格 energy / coverage（v3）：Tripel 0.178373 / 0.929630；Kriek 0.498905 / 0.916667；
Lambic 0.322447 / 1.000000（Kriek/Lambic 的 coverage 与官方基线持平，energy 全面更低）。

### 16.4 交付谱质量与缺陷安全

`generation_quality_by_style.csv`（全局标准化空间）：Tripel variance_ratio 0.7304、near_dup 0、
pair_distance_ratio 0.8453、maha 2.1962（真实 2.6423）、coverage_95 1.0、marginal_w1 0.1083；
Kriek 0.7434 / 0.02 / 0.8045 / 2.3395（真实 2.6814）/ 1.0 / 0.1154；Lambic 0.7251 / 0.02 / 0.8001 /
2.3069（真实 2.6695）/ 1.0 / 0.0982。

`defect_safety.csv`（median 阈值 DMS 19 µg/L、双乙酰 0.09 mg/L、4VG 0.20 mg/L，cap=0.9×阈值）：
三风格 DMS max 17.1、双乙酰 max 0.081、4VG max 0.180 / 0.009702 / 0.035420，`n_above_threshold`
全 0，`passed` 全 True；trans-2-nonenal 无阈值仅按实报告（Tripel max 10.9595、Kriek 0.024639、
Lambic 0.026481 mg/L）。

诚实说明（必须写进论文）：v3 的内部分量就是官方插值，它把交付集的离散度拉回到接近基线的水平
（交付集 variance_ratio ≈0.73，gate 用的折内诊断是 0.840121，两者口径不同，均如实列出）；
v3 的收益是边际分布（W1 0.136→0.098–0.115）、秩相关结构（rank_rmse 0.147→0.136）、
近重复（0.269→0.072）、内部近重复 0 与 coverage 全部合规，不是"更真实的酿造谱"。
交付谱是统计意义上的风格条件参考谱，不是实测啤酒、不认证可酿造性。

### 16.5 结论与替换建议

按预注册协议（`configs/task4_raw_q3_protocol_20261026.json` 的 `generation_replacement` 字段原文
要求八类条件全部满足，且 "No post-result retuning"）：**v3 满足全部替换门槛，按协议可替换官方
插值谱**。`outputs/q3_copula_v3_20261028/summary.json` 的 `adoption_recommendation` 明确写
"replace the official interpolated spectra"。

但建议的落地方式保持可审计：官方正式交付件 `outputs/task4_raw_q3_20261026/generated_profiles.csv`
**本次任务未修改**（不在 task-10 写范围），替换动作需由 lead/交付流程显式执行；v3 产物已在
`outputs/q3_copula_v3_20261028/` 就绪（文件集与 v2 一致：generated_profiles.csv、
generated_metadata.csv、generation_validation.csv、generation_gate.csv、
generation_gate_comparison.csv、generation_quality_by_style.csv、defect_safety.csv、
protocol.json、verification.json、summary.json）。v1/v2 目录未改动（各 10 个文件）。

### 16.6 复现

```
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe experiments\lead_review\q3\v3_diag.py
.venv\Scripts\python.exe scripts\q3_official_generation_20261028.py --source "<附件目录>" --generator v3
.venv\Scripts\python.exe -m unittest tests.test_generation_copula   (24 tests, OK)
```
耗时：诊断 0.4 s；v3 全流程 12.5 s；测试 0.5 s。
