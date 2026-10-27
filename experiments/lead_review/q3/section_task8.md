
---

## 15. Q3 生成器 v2（task-8）：donor 残差 bootstrap + 支撑集投影，同一预注册门槛

### 15.1 为什么做 v2
v1（`copula_quantile_projected`）的门禁失败点是 `rank_correlation_rmse = 0.38996`（门槛 ≤ interp+0.05 = 0.19729），根因是 v1 在 8 个主成分之外**逐坐标独立采样残差**，破坏了 216 维残差的相关结构。v2 把残差采样改成**整条残差向量从同风格 donor 啤酒 bootstrap 复用**，并在分位映射后增加**支撑集投影**（投影到该风格 leave-one-out 近邻半径内），缺陷 cap 规则与 v1 完全相同且**最后施加**（投影不能撤销 cap）。

### 15.2 v1 / v2 / 基线 三方对比（3 折 × 3 风格 × 100 条，同 seed，`outputs/q3_copula_v2_20261028/generation_validation.csv`）

| 指标 | interpolate（基线，官方） | copula_quantile_projected（v1） | **donor_residual_projected（v2）** | v2 门禁 |
|---|---|---|---|---|
| marginal_w1 | 0.40107 | 0.38541 | **0.38554** | PASS（≤0.40107） |
| energy | 0.36665 | 0.30950 | **0.33300** | PASS（≤0.36665） |
| variance_ratio | 0.72599 | 0.98240 | **0.98347** | PASS（∈[0.8,1.2]） |
| rank_correlation_rmse | 0.14729 | 0.38996 | **0.09749** | PASS（≤0.19729） |
| heldout_coverage | 0.94877 | 0.86543 | **0.89321** | 见下 |
| exact / near training copy | 0 / 0.00222 | 0 / 0 | **0 / 0** | PASS |
| internal_near_duplicate_rate | 0.26889 | 0 | **0** | PASS（非门禁项） |
| model_median_w1 | 0.12813 | 0.12129 | **0.14411** | PASS（≤0.14813） |

逐风格门禁：Tripel_energy 0.17781 ≤ 0.21421 PASS；Kriek_energy 0.50361 ≤ 0.61384 PASS；Lambic_energy 0.31758 ≤ 0.38189 PASS；Tripel_coverage 0.92963 ≥ 0.87963 PASS；**Kriek_coverage 0.83333 < 0.86667 FAIL**；**Lambic_coverage 0.91667 < 0.95 FAIL**。

- **v2 通过 11/13 条预注册规则（v1 通过 10/13）**；rank 结构问题被彻底修复：0.38996 → 0.09749，不仅通过门槛，而且**优于插值基线 0.14729**。
- 仍然失败的只有两条小样本覆盖率规则。按折看：Kriek 每折 12 个 heldout 谱、Lambic 每折 4 个，失败差额分别约 1 个与 0.4 个谱，落在该指标的分辨率地板。v2 的 Lambic 覆盖率 0.91667 比 v1 的 0.83333 有改善，Kriek 未变。
- **支撑集投影在交付谱上没有触发**（`n_support_projected = 0`，三个风格的半径 1.15–1.42）：donor 残差复用 + 分位映射后的样本本来就落在该风格的 LOO 近邻半径内，所以投影是"已就位的安全网"而不是有效改善手段。这一点必须如实写，不能宣称"投影提升了 coverage"。

### 15.3 交付谱质量（`outputs/q3_copula_v2_20261028/generation_quality_by_style.csv`，全局标准化空间）

| 风格 | variance_ratio | near_duplicate | pair_distance_ratio | maha_generated / maha_real | coverage_95 | marginal_w1 |
|---|---|---|---|---|---|---|
| Tripel | 0.9303 | 0.0000 | 0.9583 | 2.5853 / 2.6423 | 1.0000 | 0.0385 |
| Kriek | 0.9118 | 0.0000 | 0.8894 | 2.6596 / 2.6814 | 1.0000 | 0.0547 |
| Lambic | 0.9131 | 0.0000 | 0.8984 | 2.5869 / 2.6695 | 1.0000 | 0.0436 |

对照团队正式交付的 interpolate：variance_ratio 0.55–0.86、Kriek/Lambic 内部近重复 34–42%、marginal_w1 0.136–0.139。v2 的方差比 0.91–0.93、近重复 0、马氏距离与真实样本几乎重合（2.59–2.66 vs 2.64–2.68），分布上与真实风格样本基本不可区分。

### 15.4 完整性与缺陷安全
`verification.json`：`integrity_passed = true`（16/16），300 行、每风格 100、method 标签 `donor_residual_projected`、同 seed 逐位相同、不同 seed 不同、列 227 与官方同序、无 NaN、每风格每缺陷 cap 全过；`gate_passed = false`（仅上述两条 coverage）。缺陷最大值：DMS 17.1 µg/L（cap 0.9×19）、双乙酰 0.081 mg/L（cap 0.9×0.09）、4VG 0.180 mg/L（Tripel）/0.009702（Kriek）/0.03542（Lambic）（cap 0.9×0.20），**n_above_threshold 全 0**。trans-2-nonenal 与 stale hops 无阈值/无坐标，仅报告。
v1 未被改动：用 `--generator v1` 重跑到临时目录，13 条门禁与全部指标和 task-6 交付**逐位一致**（rank_rmse 0.3899643441942832 相同）。

### 15.5 结论与建议
1. **今晚仍不替换**官方 interpolate：预注册规则 13 条中仍有 2 条 coverage 未过，协议明确 "otherwise retain interpolation and its diagnosed weaknesses"，且无 post-result retuning。
2. 论文可写的诚实结论：**残差结构是 v1 的主要缺陷，v2 已把它修好（rank_rmse 0.098，优于插值基线），方差/近重复/W1/energy 全面优于插值；剩下的仅是 Kriek/Lambic 小样本覆盖率的 1 个谱级差距**。这比"生成器全面失败"或"全面成功"都更可信、更有信息量。
3. 若要真正过门禁，正确方向不是继续调参，而是把覆盖率规则的样本分辨率问题写进协议修订（例如 coverage 以绝对 heldout 谱数表述，或最小样本门 10→20）；这属于下一轮预注册，不能今晚事后改。

### 15.6 复现命令与产物
```
.venv\Scripts\python.exe scripts\q3_official_generation_20261028.py --source "<附件目录>" --generator v2
.venv\Scripts\python.exe scripts\q3_official_generation_20261028.py --source "<附件目录>" --generator v1 --output <临时目录>   # v1 回归
.venv\Scripts\python.exe -m unittest tests.test_generation_copula -v    # 19 项，全过
```
产物：`outputs/q3_copula_v2_20261028/`（generated_profiles.csv 300×227、generated_metadata.csv、generation_validation.csv、generation_gate.csv、generation_gate_comparison.csv、generation_quality_by_style.csv、defect_safety.csv、protocol.json、verification.json、summary.json）。代码：`src/beer_models/generation_copula.py::DonorResidualGenerator`（v1 类未改）、`scripts/q3_official_generation_20261028.py`（新增 `--generator v1|v2`，默认 v1）、`tests/test_generation_copula.py`（19 项）。
