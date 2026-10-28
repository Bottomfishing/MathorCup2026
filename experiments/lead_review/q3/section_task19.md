## 19. task-19（T4/T6）：风格证据、Tripel 冲突诊断、生成器消融（2026-10-28）

本节只产生**证据**，不改变迁移判据、λ 网格、生成器协议或预注册门禁；历史目录
`outputs/q3_migration_20261028/`、`outputs/q3_copula_v3_20261028/` 一律只读。脚本
`scripts/q3_evidence_20261028.py`（模块 `src/beer_models/q3_evidence.py`），产物
`outputs/q3_evidence_20261028/`（15 个文件，`verification.json` 52 项检查全 True）。

### 19.1 风格分类器折外评估（T4 首要目标）

真实 250 款啤酒、3 折分层折外（与冻结验证折同 seed 20261024；Faro 只有 3 款，5 折不可能），
每折留出分母（Tripel/Kriek/Lambic）：fold0 9/4/4、fold1 9/3/4、fold2 10/3/3。
分类器管线复用 `src/beer_models/style_models.py` 与 `configs/q3_style_protocol.json`，
imputer/scaler 均在训练折内拟合；输入 223 列（官方 224 列去掉缺失 9 个值的双乙酰，与
`outputs/current_models/q3.joblib` 的 `features` 逐列一致）。

| 方法 | 总体准确率 | 宏平均召回 | Tripel 召回 | Kriek 召回 | Lambic 召回 |
|---|---|---|---|---|---|
| logistic（22 类，class_weight=balanced） | 0.512 | 0.478 | 17/28 = 0.607 | 6/10 = 0.600 | 8/11 = 0.727 |
| extra_trees（300 棵） | **0.544** | **0.523** | **19/28 = 0.679** | **9/10 = 0.900** | **10/11 = 0.909** |
| 最近质心折外（几何参照） | 0.224 | 0.241 | 2/28 = 0.071 | 0/10 = 0.000 | 7/11 = 0.636 |

- 随机基线 1/22 = 0.045，多数类基线 31/250 = 0.124（Blond）。分类器显著高于这两个基线，
  也显著高于此前引用的**全库最近质心 LOO 0.132**——即 "0.132 vs 0.124 证明风格不可辨识"
  的说法过强：分类器可以部分辨识，但总体 0.54 仍远不足以支撑"迁移成功"。
- 混淆细节（extra_trees）：Tripel 28 条中 19 条判对，误判为 Blond 4、Amber 2、Brett 1；
  Kriek 9/10 正确、1 条判 Lambic；Lambic 10/11 正确、1 条判 West Flanders ale。
  最近质心几何参照对 Tripel 几乎全错（主要判 Strong ale 11、Dubbel 3），说明本数据上
  "欧氏质心"不是可靠的风格证据，分类器才是更公平的折外证据。
- 结论：**风格支持是"部分可验证、总体偏弱"**。论文不得用 33/33 化学约束通过替代风格证据，
  也不得把分类器 0.544 写成"风格生成正确率"；它是真实啤酒上的折外风格可辨识度上限参照。

### 19.2 Tripel 冲突诊断（目标距离 vs 缺陷安全能否兼得）

11 条 Tripel 案例（λ*：9 条 0.55、2 条 0.50，判据不改、不扩网格）：

| 量 | 值 |
|---|---|
| 平均目标距离缩短（到原始 Tripel 质心） | 12.50%（剩余目标差 87.50%） |
| 目标质心 DMS | 38.62 µg/L = **2.03 ×** 中位阈值 19.0 |
| 目标质心双乙酰 | 0.565 mg/L = **6.27 ×** 中位阈值 0.09 |
| 满足三项缺陷 cap（0.9×阈值）的 Tripel 训练样本 | **0 / 28**（median 与 mean 两情景都是 0） |
| 缺陷安全参照点 | 无真实样本 → 改用投影到 cap 的质心（DMS 17.10 µg/L、双乙酰 0.081 mg/L、4VG 不变） |
| 迁移谱到该安全点平均缩短 | 12.44% |
| 安全点到源谱距离 / 原始质心到源谱距离 | **1.0005 倍**（平均） |
| 硬投影代价 | 平均夹住 5.8 个坐标，占"源→目标"总改动的 6.47% |
| 每案例 DMS 达标 cap | 11/11（median 与 mean 情景都 11/11） |

结论（可直接写进论文）：

1. **冲突的本质是数据支持，不是几何不可达**。Tripel 训练集的"典型谱"本身就不合缺陷 cap
   （0/28 三条同时达标；质心 DMS 2.03×、双乙酰 6.27× 阈值）。因此任何"采样一个典型 Tripel"
   的做法必然违反缺陷约束；而把目标点投影到 cap 上只多走 0.05% 的距离，说明"目标距离"与
   "缺陷安全"在几何上几乎可以同时满足。
2. 真正没被解决的是**风格支持**：见 19.3——迁移后的 Tripel 谱在冻结分类器与几何上都不被
   承认为 Tripel。所以 19.2 的"距离缩短 12.5% 且 11/11 缺陷达标"只能证明**缺陷约束满足**，
   不能证明**风格迁移成功**。
3. 因此 Tripel 的正确表述是："目标风格区域本身超标，我们只能给出一个贴 cap 的**条件性**
   迁移（λ*=0.5–0.55，距目标质心仍差 87.5%），并用分类器折外证据说明其风格支持有限"。

### 19.3 迁移谱的冻结分类器应用（新增证据）

将 33 条迁移后谱（官方 224 维）喂给 `outputs/current_models/q3.joblib`（全训练冻结分类器）：

| 请求风格 | 一致判为目标（logistic ∧ extra_trees） | logistic 命中 | extra_trees 命中 | 最近质心命中 |
|---|---|---|---|---|
| Tripel | **1/11** | 11/11 | 1/11（10 条判 Blond） | **0/11**（11 条判 Brett/cofermented） |
| Kriek | 11/11 | 11/11 | 11/11 | 11/11 |
| Lambic | 11/11 | 11/11 | 11/11 | 2/11（其余判 Brown） |

诚实解读：Kriek/Lambic 的 11/11 部分来自 **λ*=1.0**（迁移谱就是目标质心附近），不能当作
迁移能力的独立证据；Tripel 的 λ*=0.5–0.55 混合谱不被 extra_trees（更可靠的折外模型）
承认为 Tripel，logistic 却全部判 Tripel——**两个分类器结论相反，说明 Tripel 风格支持
不确定**，只能报"证据冲突/不足"，不能报"迁移成功"。

### 19.4 生成器消融（T6：同折、同生成量、同 seed）

直接拼接三个冻结目录的 `generation_validation.csv`（3 折 × 3 风格 × 方法），并合并每
(fold, style) 的留出分母；**没有重跑生成器，没有改任何门槛**：

| 生成器 | marginal_w1 | energy | 方差比 | 秩相关误差 | 覆盖率 | 近重复率 | model_median_w1 |
|---|---|---|---|---|---|---|---|
| interpolate（官方基线） | 0.4011 | 0.3666 | 0.7260 | 0.1473 | 0.9488 | 0.2689 | 0.1281 |
| copula_quantile_projected（v1） | 0.3854 | 0.3095 | 0.9824 | 0.3900 | 0.8654 | 0.0000 | 0.1213 |
| interp_quantile_projected（v1.5） | 0.3854 | 0.3261 | 0.9824 | 0.1469 | 0.9210 | 0.0711 | 0.1281 |
| donor_residual_projected（v2） | 0.3855 | 0.3330 | 0.9835 | **0.0975** | 0.8932 | 0.0000 | 0.1441 |
| hybrid_support_projected（v3） | 0.3851 | 0.3332 | 0.8401 | 0.1361 | **0.9488** | 0.0722 | **0.1222** |

- 每折留出分母（Tripel/Kriek/Lambic）：fold0 9/4/4、fold1 9/3/4、fold2 10/3/3；覆盖率
  **一条谱 = 0.083（Kriek 12）/0.25（Lambic 4）/0.10–0.11（Tripel 9–10）**，所以覆盖率
  0.05 的容差在 Kriek/Lambic 上不到一条谱，必须同时报分母与逐折值。
- 近重复定义（来自 `q3_raw.spectrum_diagnostics`）：生成谱两两最近距离 ≤
  `near_copy_radius = 0.05 × 训练集 LOO 最近距离中位数`；精确复制是最近真实谱距离 < 1e-10。
  这不是"精确复制率"，不能混用。
- v3 gate 的官方口径是 **13 条实质规则 + 1 条总汇总 = 14 项**（`gate_all_passed` 是汇总，
  不是第 14 条独立规则）。v3 的 13 条实质规则全过（`generator_gate_rules.csv`），但——

**披露（必须写进论文）**：v3 的设计在**同一批固定验证折**上使用了 v2 的覆盖诊断（诊断发现
oracle 也过不了 Kriek/Lambic 的覆盖率规则，于是预注册了"内部插值分量 + 外部 donor 残差分
量"的混合方案）。因此 v3 的 13/13 **不是独立泛化证明**，而是"在已知固定折上、预注册方案
后一次通过"；正式替换建议仍应保留"内部验证"字样，并同时给出 v2（11/13）与插值基线
（13/13 但方差比 0.726、近重复 26.9%）的对照。

### 19.5 文件与复现

产物（`outputs/q3_evidence_20261028/`）：`style_classifier_oof.json/.csv`、
`tripel_conflict.csv`、`tripel_conflict_summary.json`、`tripel_defect_safe_samples.csv`、
`migrated_style_classifier.csv`、`generator_ablation.csv`、`generator_ablation_summary.csv`、
`generator_gate_rules.csv`、`generator_gate_breakdown.json`、`dev_use_disclosure.md`、
`summary.json`、`summary.md`、`verification.json`、`protocol.json`（含全部输入 sha256）。

```powershell
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe scripts\q3_evidence_20261028.py --source '<附件目录>'      # 6.1 s，52 checks 全 True
.venv\Scripts\python.exe -m unittest tests.test_q3_evidence -v                     # 21 tests OK
```
