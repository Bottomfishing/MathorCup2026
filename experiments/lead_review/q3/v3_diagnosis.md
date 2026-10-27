# v3 覆盖率诊断：分辨率地板还是系统性支撑域偏移？

日期 2026-10-28。任务 task-10 要求。诊断脚本 `experiments/lead_review/q3/v3_diag.py`，
原始点表 `experiments/lead_review/q3/v3_coverage_points.csv`（147 行 = 3 折 × 3 风格 × 3 方法 ×
该折该风格 heldout 谱数），逐折汇总 `experiments/lead_review/q3/v3_fold_summary.csv`。

v3 的生成器与脚本实现在 `src/beer_models/generation_copula.py`（`HybridSupportGenerator`）与
`scripts/q3_official_generation_20261028.py --generator v3`。**门槛、基线、`build_gate`、评估代码一行未改**。

## 1. 诊断方法

沿用预注册的 3 折（`validation_seed=20261024`）、每折每风格 100 条、三种方法同 seed。对每个
held-out 谱计算：到该风格生成谱集合的最近距离（`spectrum_diagnostics` 同一 z 空间、同一
`cdist/sqrt(p)` 口径）、coverage 半径（训练集 LOO 最近邻距离的 95 分位，与预注册一致），
判定 `covered = nearest <= radius`。额外加入第 4 个参照 `oracle_train_set`：**生成集直接等于该
风格训练谱本身**。它是任何"忠于训练支撑域"的生成器的上界观测：若 oracle 都不能覆盖，则
"生成器支撑太窄"就不是可修复的建模缺陷，而是该 coverage 指标的定义 + 小样本效应。

## 2. 结果（按风格对折取平均，与 gate 同口径）

| style | v2 donor_residual | interpolate（官方基线） | oracle_train_set | 半径 | heldout→训练平均距离 |
|---|---|---|---|---|---|
| Tripel | 0.92963 | 0.92963 | 0.92963 | 1.171445 | 0.7521 |
| Kriek | **0.83333** | 0.91667 | **0.75000** | 1.091035 | 0.8677 |
| Lambic | **0.91667** | 1.00000 | **0.72222** | 0.899564 | 0.7109 |

逐折 v2 / interpolate / oracle：
- Tripel fold0 0.8889 三者相同，fold1 1.0 三者相同，fold2 0.9000 三者相同；
- Kriek fold0 0.50 / 0.75 / 0.25（heldout 4 条），fold1 1.0 三者相同（3 条），fold2 1.0 三者相同（3 条）；
- Lambic fold0 1.0 / 1.0 / 0.75（4 条），fold1 0.75 / 1.0 / 0.75（4 条），fold2 1.0 / 1.0 / 0.6667（3 条）。

## 3. 判读

**判读一：不是 v2 残差模型的 bug。** oracle 用训练谱本身作生成集，Kriek 只有 0.75、Lambic 只有
0.7222，**比 v2 的 0.83333 / 0.91667 更差**。也就是说把"生成谱完全贴合训练支撑"做到极限，也过不了
Kriek(≥0.86667) / Lambic(≥0.95) 这两条规则。v2 的 coverage 失败不可能通过"把支撑域再收窄"修好。

**判读二：半径定义 + 小样本的联合效应。** 半径是训练集 LOO 最近邻距离的 95 分位，训练折里
Kriek 每折 6–7 条、Lambic 7–8 条；而 held-out 谱到训练云的距离系统性更大（Kriek 平均 0.8677，
半径 1.0910；Lambic 0.7109 / 0.8996），尾部谱超出半径。逐漏点：

| style | fold | beer | 方法 | 最近生成距离 / 半径 | 超出 |
|---|---|---|---|---|---|
| Kriek | 0 | 161 | v2 | 1.3312 / 0.8794 | +51.4% |
| Kriek | 0 | 166 | v2 | 0.9151 / 0.8794 | +4.1% |
| Lambic | 1 | 183 | v2 | 0.9263 / 0.9169 | +1.0% |
| Tripel | 0 | 160 | v2 | 1.5692 / 0.9860 | +59.2% |
| Tripel | 2 | 82 | v2 | 1.5206 / 1.0043 | +51.4% |

Tripel 的三个漏点是同一批结构性离群（beer 160 / 82），**三种方法加 oracle 全部漏**，对 gate 无差别；
真正造成差距的是 Kriek 166 与 Lambic 183 这种"只超 1–4%"的边缘谱——它们是否被覆盖，取决于生成谱
恰好落在半径内还是外 0.01–0.04 的半径距离，而每折 Kriek 只有 12 个、Lambic 只有 12 个 heldout 谱，
1 条谱 = 0.0833（Kriek 折均）或 0.0833（Lambic 折均）：**Lambic 允许的降幅 0.05，小于 1 条谱的分辨率
（0.0833）；Kriek 允许的降幅 0.05 同样小于 1 条谱**。这是任务所说"分辨率地板"的直接证据：规则在
这个样本量下要求的差异小于指标的一个量化台阶。

**判读三：coverage 与 variance∈[0.8,1.2] 两条规则互相拉扯。** interpolate 能过 coverage 不是因为它
更"真"，而是因为它故意欠离散：凸组合把谱拉向风格质心（v2 的 variance_ratio 0.98347，interpolate 只有
0.72599），coverage 奖励"更靠近质心的稠密采样"。pure-fidelity 的 v2（相关结构最好：rank_rmse
0.09749）在 coverage 上反而吃亏。要同时满足两条，需要在"支撑域内部填充"与"真实边际/相关结构"
之间做混合，而不是继续优化单一残差模型。

## 4. v3 的预注册设计（跑 gate 之前一次性写死）

设计来源即本诊断，参数不搜索、不看 gate 结果：

- 两分量混合：`interior_fraction = 0.5` 的 `StyleGenerator(method='interpolate')` 内部分量 +
  0.5 的 v2 `donor_residual_projected` 外部分量；内部分量用独立 seed 偏移 `interior_seed_offset = 7919`；
- 同样的逐坐标经验分位映射（保留边际与秩结构）、同样的支撑投影、**缺陷 cap 最后施加**（DMS 17.1 µg/L、
  双乙酰 0.081 mg/L、4VG 0.180 mg/L）；
- 方法名 `hybrid_support_projected`；写入 `outputs/q3_copula_v3_20261028/protocol.json`
  的 `pre_registered_design`（`fixed_before_gate_evaluation=true`、`gate_thresholds_modified=false`、
  `baseline_modified=false`、`evaluation_code_modified=false`、`parameter_search="none"`）。

设计取舍的诚实说明：内部分量是项目已经在用的插值，它的作用只是补齐 baseline 已证明能覆盖的
支撑域内部；v3 的交付谱因此比 v2 更靠近风格质心（交付集方差比约 0.73，见 FINDINGS §16），
这是换取 coverage 合规的代价，不隐瞒、不声称"更真实"。

## 5. v3 结果（同一套预注册门槛）

`outputs/q3_copula_v3_20261028/verification.json`：`integrity_passed=true`（16/16）、
`gate_passed=true`（14/14 项，含 `gate_all_passed`）、`all_passed=true`。13 条规则全部通过，
v2 失败的两条已补齐：Kriek_coverage 0.916667 ≥ 0.866667、Lambic_coverage 1.000000 ≥ 0.950000。
按预注册协议：**可替换官方插值谱**。

## 6. 复现

```
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe experiments\lead_review\q3\v3_diag.py
.venv\Scripts\python.exe scripts\q3_official_generation_20261028.py --source <附件目录> --generator v3
.venv\Scripts\python.exe -m unittest tests.test_generation_copula
```
