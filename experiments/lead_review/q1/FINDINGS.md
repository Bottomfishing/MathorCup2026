# Q1 复核与口径纠偏（lead review）

日期：2026-10-27　分析：q1-analyst　工作目录：`C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件`
范围：只读正式产物（`outputs/`、`deliverables/`、`configs/`、`src/` 未改动），全部新代码/结果在 `experiments/lead_review/q1/`。

## TL;DR（先看这 6 条）

1. **题面 Q1 的交付物是"250 款 × 评分卡 50 个可评维度"的 0–6 整数分级表**（Aroma 15 + Flavour 15 + 其他 20；卡里另有 A/F 两个乙醛槽，但 4.csv/原始记录都没有这两项，只能标 NA）。不是只对 10 款酒，也不是只交连续值。证据：3.xlsx summary 共 52 列（50 个感官维度 + comments），4.csv 恰有 50 个数值感官列。
2. **3.xlsx 是空白模板，不能当标定锚**：实测 12 张卡（`beer_1`…`beer_12`，题面写 10 张是概述误差）+ summary；卡内所有评分格都是公式默认 `0`，summary 12 款评分也全为 0，全表没有任何人工 0–6 实测分。团队的 0–6 监督标签实际来自外部 Zenodo raw（DOI 10.5281/zenodo.23624915），**论文必须披露**。
3. **"数据缺失"落点**：4.csv 缺失 = 0；1.csv 共 73 个缺失格（0.125%），其中 226 个模型输入列内 60 格（`amyl.acetate` 27 格=10.8% 最多）；7.csv 缺失 750 格（1.60%），全部集中在 `palate_acetaldehyde`/`palate_acetate`/`palate_dentist` 三列各 250/250（100%）。题面所说"缺失"应由 1.csv 与 7.csv 共同解释，**不在 4.csv**。
4. **按题面口径（每款酒一个面板分）现有模型并不差**：OOF 面板中位等级 MAE **0.3928**、完全命中 **74.1%**、±1 级 **91.7%**；逐目标面板 Spearman 均值 0.328。个人级 MAE 0.7952 的信息论地板是 **0.6636**（面板加权中位 oracle），模型只比地板高 0.132。
5. **"高分精确召回 = 0"是取整决策造成的伪失败**：ET 输出"整数条件中位数"，永远不吐 5/6（面板 5 分仅 179/12500、6 分仅 19/12500 格）；但 task6 OOF 概率列的面板级逐目标 AUC(≥5) 均值 **0.807**、中位 0.897，precision@5 = 7.2%（基准率 1.58%，约 4.6× lift），≥6 的 precision@5 = 1.6%（基准率 0.15%，约 10.5× lift）。**排序信息存在，只是没被交付表用上。**
6. **真正可改进的点是"面板均值标定"**：ET 整数预测对面板均值的 MAE 0.6123，竟差于两个朴素基线（目标均值 0.5493、风格×目标均值 0.4668）。逐目标折外线性标定后 **0.5027（−17.9%）**，再加风格先验收缩 **0.4622（−24.5%）**，同时 ±1 命中率从 91.7% 升到 93.1%。这是 1 天内收益/成本最高的改动。

一句话结论：**团队此前确实在错误的评估单位（品评员个体逐条）上用错误的指标（精确 5/6 召回）自证失败；题面口径下的等级预测是可用且可辩护的，但"面板均值/强度"这一列需要做标定，且论文必须把标签来源与 3.xlsx 空白的事实写清楚。**

---

## Q1 题面到底要求交付什么

题面原文（`experiments/lead_review/problem_text.txt:25-27`）：

> 问题1：数据缺失条件下的感官评分。以 1.csv 化合物数据为基础，给出 3.xlsx 中定义的 Aroma 与 Flavour 两套、共约 40 个感官维度的 0–6 分制的分级评分结果。

实测核对：

| 证据 | 内容 |
|---|---|
| `3.xlsx` sheets | `beer_1`…`beer_12`, `summary`（12 张卡 + 汇总，题面"10 张"为概述） |
| `3.xlsx` summary | 13 行 × 52 列：52 个列名 = 50 个感官维度 + `comments`；12 行 = Beer#1–12 |
| `3.xlsx` 卡内评分 | 每卡 101 行 × 33 列、316 个非空单元格；非空数字 1–6 全部是打印的 0–6 刻度标签（表头行 row8/21/34 的 col5,7,9,11,13,15 与 col21…31）；评分输入格为空，得分格为公式默认 `0`（如 beer_1 row10/12/14/16 的 col17/33） |
| `3.xlsx` summary 评分 | 除 52 个表头名与 12 个 Beer 编号外，所有评分格 = `0`（默认公式结果），**无人工实测分** |
| `4.csv` | 250 行 × 53 列：`beer`,`beer_id`,`tasting_category_fine` + **50 个数值感官列**，缺失 0 |
| `2.csv` | 231×232 相关阵；实测为 **Spearman** 值（题面写 Pearson，团队已核实最大误差 1.72e-15），论文一句话说明即可 |

**结论**：交付 = 250 款 × 50 维（52 槽中 2 个乙醛槽无数据，标 NA）× 0–6 整数分级，共 12,500 个可评单元格。这与 `outputs/task6_final_20261026/paper/tables/q1__oof_scores_all52.csv` 的形状完全一致（13,000 行 = 250×52，其中 12,500 个非空 `estimated_grade_0_6`）。"数据缺失条件"指建模输入侧的缺失（1.csv 化学 + 7.csv palate），不是评分标签缺失。

补充：题面"约 40 个"与卡内 50 个的差异，建议论文写成"评分卡实际定义 50 个可评维度，其中 Aroma 15、Flavour 15、其余 20；题面'约 40'为概述"。

## Q2 评估单位与指标：个人级 vs 啤酒级

### 2.1 团队口径错在哪

团队把 2,197 条品评记录展开成 109,850 个"人×酒×维度"个体评分，用个体级 MAE 与"5/6 分精确召回"当主指标。但题面要求的是**每款酒每个维度一个 0–6 分**，即评估单位应是 `(beer, target)` 的 12,500 个单元格；品评员之间的分歧是标签噪声，不是模型要逐条拟合的目标。个体级"5/6 精确召回"更会系统性归零：条件中位数为 0/1 的格子占绝大多数，中位数预测天然不吐 5/6。

### 2.2 现有 OOF 结果的重算（不重跑训练）

数据：`outputs/r2_raw_q1_20261026/oof_predictions.csv.gz`（方法含 `extra_trees`、`training_median` 等；已核对 `extra_trees` 的 `predicted_grade` 与 `outputs/task6_final_20261026/paper/tables/q1__oof_scores_all52.csv` 的 `estimated_grade_0_6` 在 12,500 格上逐格一致）；真值：`observed_panel_distribution.csv` 的 `panel_weighted_median`/`panel_weighted_mean`；权重：`rating_weights_and_folds.csv`。

**个体级（加权，分母 12,500）**

| 方法 | 加权 MAE | 完全命中 | ±1 内 |
|---|---|---|---|
| ExtraTrees（正式模型） | 0.795214 | 0.604919 | 0.783035 |
| 训练先验中位（基线） | 0.829079 | 0.593806 | 0.772551 |
| 面板中位 oracle（逐格真值中位） | **0.663567** | 0.658689 | 0.818681 |

`0.663567` 不是"另一个模型"，而是**任何只利用啤酒身份的点预测在加权 L1 下的下界**（逐格加权中位最小化 L1）。模型 0.7952 距下界 0.1317；从基线到 oracle 的可改进空间仅 0.1655，模型已吃掉 20.5%。

**啤酒级（12,500 个 `(beer,target)`，与题面一致）**

| 方法 | 面板均值 MAE | 面板中位 MAE | 完全命中 | ±1 内 | 面板 Spearman(逐目标均值) |
|---|---|---|---|---|---|
| ExtraTrees 整数中位（现交付） | 0.6123 | **0.3928** | **0.7413** | **0.9169** | 0.3283 |
| 训练目标均值（折外） | 0.5493 | 0.7725 | 0.3794 | 0.9091 | −0.0563 |
| 风格×目标均值（折外） | **0.4668** | 0.6848 | 0.4916 | 0.8985 | 0.3665 |
| 训练先验中位（折外） | 0.6533 | 0.4480 | 0.7143 | 0.8986 | −0.0100 |
| 面板中位 oracle | 0.4437 | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

参考基线：全预测 0 的 MAE = 0.8994、完全命中 = 64.1%（面板中位为 0 的占 8,013/12,500 = 64.1%）、±1 = 71.7%。面板中位分布：0:8013, 1:954, 2:1347, 3:1367, 4:621, 5:179, 6:19。

**怎么读这张表**（重要，写论文时别混用）：
- 若交付"0–6 整数等级"，现模型是四条里最好的（MAE 0.393 / 命中 74.1% / ±1 91.7%），显著优于先验中位与均值基线。
- 若考察"面板强度均值"（连续量），现模型（0.612）反而差于目标均值（0.549）与风格均值（0.467）——模型有信息但**输出标定错位**（见 Q4）。这也是"MAE 0.79 上不去"体感的真正来源之一：在个体级它已接近地板，在均值级它被基线反超。
- 面板中位 oracle 对面板均值的 MAE = 0.4437，说明**即使交出完美的整数等级表，它和连续面板均值也天然差 0.44**；用连续均值去卡整数表是拿错了尺子。

**混淆矩阵**（行=观测面板中位，列=预测，完整 CSV 见 `confusion_panel_extra_trees.csv` / `confusion_panel_training_median.csv`）：

ExtraTrees（列 0..6）：
```
0: [7548, 227, 209, 179, 107, 47,  7]
1: [ 311, 309, 153,  67,  17,  5,  1]
2: [ 138, 346, 537, 269,  71, 17,  0]
3: [  13,  70, 437, 778, 332, 64,  7]
4: [   3,   2,  11,  74,  94, 46,  4]
```
（观测 5/6 行在混淆矩阵中不存在，因为预测列不含 5/6；观测 5 分 179 格中 47 格被预测为 5？注意上表 0 行第 5 列 47 为"观测 0 预测 5"，而观测 5 行整行缺失——即观测 5/6 全部被压到 0–4。这正是召回 0 的直接机制。）

### 2.3 高分识别：换指标后完全不是"0"

| 指标 | ≥5 分 | =6 分 |
|---|---|---|
| 正例面板格数 | 198 / 12,500（1.58%） | 19 / 12,500（0.15%） |
| 面板级逐目标 AUC 均值 | **0.807**（34 个有正例目标，中位 0.897） | **0.829**（12 个目标，中位 0.880） |
| 面板级 pooled AUC | 0.906 | 0.936 |
| precision@5（每目标取概率 top5） | **7.2%**（≈4.6× 基准率） | **1.6%**（≈10.5× 基准率） |
| 个人级加权 AUC | 0.813 | 0.794 |
| 现交付整数表的精确召回/精确率 | 0 / 0（预测 5/6 的格子数 = 0） | 0 / 0 |

注意：此处 AUC 用 `task6` 的 `probability_rating_ge5`/`probability_rating_6` 计算，与 `docs/Q1实际预测改进_20261026.md` 里的旧值 0.694/0.677 不一致，需团队核对概率列口径（是否 panel-level 后验、是否用了 target 基准率）；但两套数字方向一致：**排序信号明显强于"召回 0"给人的印象**。

**建议论文写法**：高分识别应报 AUC / precision@k / lift，并说明 5–6 分在 12,500 格中仅占 1.58%/0.15%，任何"精确召回"指标在如此稀疏的正例上都不稳定；若评委要求混淆矩阵，就报 7×7 全表并注明 5/6 列为空的原因（中位决策规则）。

## Q3 3.xlsx 能否作为"标准化值→0–6 原始分"的标定锚

**不能直接用**：3.xlsx 没有任何人工分（见 Q1 证据），12 张卡与 summary 全是空模板/默认 0。团队"无法唯一恢复"的结论在"只用附件"的前提下成立。

但在竞赛里只需要**可辩护的单调标定 + 标定误差报告**，这一点是成立的，且误差很小：

1. `4.csv` 的 50 个标准化值与面板原始均值几乎仿射：逐目标 Pearson 均值 **0.9668**、中位 0.9772；50/50 个目标 >0.9，6 个 >0.99。
2. 用 k=10 个"锚点啤酒"逐目标拟合 `z → 面板均值` 线性映射，在其余 240 款上评估：MAE **0.1362**（k=12: 0.1323）；保序回归（isotonic）MAE 0.1798（k=12: 0.1703）。常数基线 MAE 0.5472。
3. 把映射值四舍五入回 0–6 与观测面板中位比：k=10 完全命中 **61.0%**、±1 **95.2%**（k=12: 61.2% / 95.3%）。
4. 锚点从哪来？竞赛内只能用可披露的来源：要么用外部公开原始记录（我们用的 Zenodo raw，需在论文披露并说明检索/匹配方法），要么在论文中把"分位数映射规则"写成显式定义（如每维按训练集面板分布的分位数切 0–6），并报告该规则在留出酒上的标定误差。**不要声称 3.xlsx 提供了锚点。**

论文可用公式（逐目标 t、酒 b）：

```
μ̂_{b,t} = a_t + b_t · z_{b,t}            (线性标定, 锚点集 A_t 上最小二乘, |A_t|≈10)
ĝ_{b,t} = round(clip(μ̂_{b,t}, 0, 6))     (整数等级)
标定误差: MAE_t = mean_{b∉A_t} |μ̂_{b,t} − ȳ_{b,t}|
```

结论：把"标准化连续值→0–6"写成"单调仿射标定 + 留出误差 0.13–0.18 MAE + ±1 级 95%"是**可辩护的**；但必须承认 3.xlsx 本身不提供锚点。

## Q4 1 天内最值得做的改进（按 预期收益/成本/风险 排序）

| 优先级 | 动作 | 预期收益（已实测/估计） | 成本 | 风险 |
|---|---|---|---|---|
| **P0** | 换评估口径：主表改为啤酒级，报 MAE/命中/±1/Spearman/混淆矩阵；个人级只作辅助 | 不改进模型，但把"0.79/召回 0"从"失败"改写成"个体噪声地板附近的正常结果" | 0（本报告已给表） | 低；需与论文组统一口径 |
| **P0** | 给交付表加"面板均值标定列"：折外逐目标线性 `μ̂ = a_t + b_t·ĝ_ET`，再四舍五入出等级 | 面板均值 MAE 0.6123 → **0.5027（−17.9%）**；±1 命中 91.7%→93.1% | 半天（本报告已跑出 OOF 版） | 低；标定参数需折外拟合，避免泄漏 |
| **P1** | 再加风格先验收缩：`μ̂ = a_t + b_t·ĝ_ET + c_t·style_mean`（tasting_category_fine 来自 1.csv） | 面板均值 MAE → **0.4622（−24.5%）**，已优于风格基线 0.4668 | 半天 | 中；需说明 style 是输入元数据、非标签泄漏；论文写"分层收缩" |
| **P1** | 高分改报排序指标；如需交付 5/6 预测，用分位数标定把预测等级边际分布对齐观测（每目标每折按训练分位映射） | 可产出 80 个 5 分预测：召回 5.1%、精确率 12.5%（基准率 1.58% → 7.9× lift）；但面板中位 MAE 0.3928→0.4195，±1 91.7%→90.5% | 半天 | 中；属于"边际校准"不是"识别力提升"，须如实写 trade-off |
| **P2** | 直接把学习目标改成面板均值（逐目标 ET/GBM 回归，而非个体条件中位） | 估计可到 0.45 左右（本报告 Ridge 尝试失败：alpha=100 仍 0.79，线性模型不适合；需树模型） | 1 天，需重跑训练管线（本次限制内未做） | 中；改动正式管线需重验证 |
| **P3** | 尺度标定说明（Q3）：4.csv 标准化↔0–6 单调仿射，10 锚点留出 MAE 0.136、±1 95.2% | 让"0–6 从哪来"可辩护 | 2 小时（已跑） | 低；必须披露 3.xlsx 空白与外部 raw |

**已直接跑出的新结果**（全部在 `experiments/lead_review/q1/`）：
- `panel_level_metrics.csv`：Q2 全部数字。
- `improvement_experiments.csv`：P0/P1 的 MAE 对比。
- `calibration_z_to_raw.csv`：Q3 逐目标 Pearson/Spearman/锚点标定误差。
- `ridge_retarget.csv`：直接回归面板均值的负结果（Ridge 明显差于 ET/基线，说明该走树模型或标定，不是线性重定向）。
- `candidate_panel_mean_and_grade_250x50.csv`：250×50 的"标定后面板均值 + 四舍五入等级"候选交付表（研究用，未替换正式交付）。

## Q5 诚实结论

**题面口径下的成绩是可接受的**，理由三条：

1. 交付物是整数等级表，现模型面板中位 MAE 0.3928（0–6 量程的 6.5%）、完全命中 74.1%、±1 级 91.7%，且优于目标均值/先验中位基线；全零基线 MAE 0.8994 被大幅超越。
2. 个体级 MAE 0.7952 与"只利用啤酒身份"的 L1 下界 0.6636 只差 0.13；剩余误差主要来自品评员分歧（啤酒内方差占逐目标 69.5%，ICC 均值 0.305），不是模型没学好。
3. "高分召回 0"是决策规则伪失败：AUC(≥5) 0.807、precision@5 4.6× lift；整数中位预测按定义不吐长尾等级。

**所以明确写：团队此前在错误的评估单位（品评员个体逐条）和错误的指标（5/6 精确召回）上自证失败。** 正确的陈述方式是：

> 在题面要求的啤酒×维度 0–6 分级口径下，Q1 的等级预测 OOF MAE 为 0.393、完全命中 74.1%、±1 级 91.7%；高分维度因 5–6 分仅占 1.6% 且采用中位决策，精确召回不作为主指标，改报逐目标 AUC（≥5 为 0.81，≥6 为 0.83）与 precision@5 lift（4.6×/10.5×）。个体级 MAE 0.795 已接近仅利用啤酒身份的信息下界 0.664，个体分歧（ICC≈0.30）是主要误差来源。

同时必须诚实写出的**真实短板与风险**（不粉饰）：

- 面板均值这一列现模型不如朴素基线（0.612 vs 0.467），交付表若含"连续强度"必须用标定后的版本，否则会被评委抓 MAE。
- 0–6 标签来自外部 Zenodo raw，3.xlsx 是空模板；这是本方案最大的可攻击点，必须在论文"数据与标签"节交代匹配方法与可复现性（`docs/R1_原始感官评分审计_20261025.md` 已有审计：双向唯一匹配、最大绝对差 2.83e-8）。
- 若评委只认"3.xlsx 是评分卡、必须用卡上 0–6 标注"的严格口径，则本题在"标签不存在"这一点上对所有队伍都开放；我们的应对是"公开原始记录 + 可辩护标定 + 误差报告"，不是假装卡里有分。
- 高分识别是"排序可用、阈值不稳"：分位数标定后精确率 12.5% 仍不高，不要把它包装成强识别能力。
- 所有"改进"均为预测关联，不构成因果结论（例如"化合物 X 导致 5 分"这类表述不能用）。

## 复现

```powershell
cd "C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件"
.venv\Scripts\python.exe experiments\lead_review\q1\panel_analysis.py   # Q1/Q2/Q3/Q4 主表
.venv\Scripts\python.exe experiments\lead_review\q1\extras.py           # 4.csv 极值核查、逐目标 AUC、Ridge 重定向负结果
```

产物（同在 `experiments/lead_review/q1/`）：
`missingness_1csv.csv`、`panel_level_metrics.csv`、`panel_median_distribution.csv`、`confusion_panel_extra_trees.csv`、`confusion_panel_training_median.csv`、`variance_decomposition_icc.csv`、`calibration_z_to_raw.csv`、`improvement_experiments.csv`、`ridge_retarget.csv`、`candidate_panel_mean_and_grade_250x50.csv`、`analysis_report.json`、`extra_report.json`、`panel_analysis.log`。

## 附：可直接进论文的表述（按段改）

1. **数据与缺失**：题面称"数据缺失条件下"，实测 4.csv 无缺失；1.csv 250×234 中 73 个缺失格（0.125%），剔除 7 个派生特征后的 226 个输入列含 60 个缺失格，最多为 amyl.acetate（27 格，10.8%）；7.csv 的 750 个缺失格全部来自 3 个 palate 列（各 100% 缺失）。缺失以逐列中位数填补并保留缺失指示，缺失率敏感性在 MCAR 5/10/20% 下个体 MAE 为 0.7964/0.7980/0.8018。
2. **评分卡与交付**：3.xlsx 定义 50 个可评感官维度（Aroma 15、Flavour 15、其他 20），另 2 个乙醛槽无数据来源，交付表以 NA 标注；共 12,500 个可评单元格。附件 3.xlsx 为空白模板，0–6 监督信号来自公开原始品评记录（DOI 10.5281/zenodo.23624915），经 38 维 z 分数双向唯一匹配到 250 款酒。
3. **评估口径**：以每款酒每个维度的面板加权中位数为等级真值、加权均值为强度真值；主指标为等级 MAE/完全命中/±1 命中/逐目标 Spearman，辅助指标为个体级加权 MAE 与逐目标 AUC/precision@k。个体级标签用于诊断品评员分歧，不作为主评分单位。
4. **尺度标定**：4.csv 标准化值与面板均值逐目标 Pearson 均值 0.967；10 个锚点/目标的单调线性标定在留出 240 款上 MAE 0.136、四舍五入后 ±1 级 95.2%，可作为"标准化值→0–6"的可辩护映射；3.xlsx 本身不含实测分，不能直接充当锚点。
5. **结果表述**：等级 MAE 0.393、完全命中 74.1%、±1 级 91.7%；面板均值经折外线性标定 + 风格收缩后 MAE 0.462；高分识别以 AUC(≥5)=0.81、precision@5 4.6× lift 报告，不报精确召回。

---

# 追加：决策层新表（响应 lead m00208）

## 1. 新交付表：累积概率 top-k 决策（模型不变，只改决策）

规则（逐目标 t、折外折 f）：训练折上取观测面板中位 ≥5 的比例 r5 与 =6 的比例 r6（只用训练折标签）；折外 250 款按 r5/r6 取整数 k5、k6；每折测试酒中按 `probability_rating_6` 取 top-k6 记为 6，其余按 `probability_rating_ge5` 取 top-k5 记为 5，其余保留 ET 整数中位但在 0–4 内封顶。等价于"逐维累积概率阈值化 + 边际分布匹配"，不重训模型。产物：`q1_decision_topk_table_250x50.csv`（含旧等级/概率/新等级）、`q1_decision_topk_grades_wide_250x50.csv`（250×50 宽表）。

**等级分布（12,500 格）**

| 表 | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 旧表（整数条件中位） | 8324 | 863 | 1378 | 1701 | 234 | **0** | **0** |
| 新表（top-k 决策） | 8279 | 856 | 1361 | 1648 | 152 | **188** | **16** |

**OOF 指标对比（啤酒级，真值 = 面板加权中位/均值）**

| 指标 | 旧表 | 新表（top-k） | 变化 |
|---|---|---|---|
| 面板中位 MAE | 0.3928 | 0.4080 | +0.0152 |
| 面板均值 MAE | 0.6123 | 0.6231 | +0.0108 |
| 完全命中 | 74.128% | 73.880% | −0.25pt |
| ±1 命中 | 91.688% | 91.312% | −0.38pt |
| 逐目标宏平均 Spearman | 0.3283 | 0.2903 | −0.038 |
| 预测 5 分格数 / 精确率 / 召回 | 0 / — / 0 | 204 / **25.49%** / **26.26%** | F1 0.2587（基准率 1.58%，≈16× lift） |
| 预测 6 分格数 / 精确率 / 召回 | 0 / — / 0 | 16 / **6.25%** / **5.26%** | 基准率 0.15%，≈41× lift（仅 16 格，不稳） |

口径说明：高分精确率/召回按 `(beer,target)` 单元格、真值 `panel_weighted_median ≥5`（或 `=6`）计算，不是品评员个体逐条；基准率为 198/12500 与 19/12500。

**高分覆盖（"每维至少能达到 5/6 吗"）**：不能强求。观测面板中位 ≥5 的目标只有 34/50 个（共 198 格），=6 的只有 12/50 个（共 19 格）；其余 16 个维度在 250 款里从未出现 5 分、38 个从未出现 6 分（如 metallic/dms/diacetyl/X4vg/t2n 等；完整清单见 `decision_topk_report.json` 的 `observed_prevalence`）。新表在可达维度上 100% 覆盖（34/50 有 5、12/50 有 6），最大 20 个 5/目标、4 个 6/目标；若强行"每维都给 5/6"（force≥1 变体），MAE 恶化到 0.469、5 分精确率跌到 16.3%、6 分 0.66%，**不可取**。

**结论**：决策层能把"高分召回 0"变成"5 分精确率 25%、召回 26%，且主指标几乎不掉（MAE +0.015、命中 −0.25pt）"。但这不是识别力提升，只是把已有排序信息（面板级 AUC 0.807）转成显式标签。若竞赛评分只看等级 MAE/命中率，旧表更优；若要交付"能报出 5/6 的表"或展示高分筛查能力，用 top-k 新表更合适。**建议交付两列**：主列用新 top-k 等级（可解释、覆盖高分），同时给出 `probability_rating_ge5/6` 作为"高分潜力"辅助列；论文如实写 trade-off。

## 2. 概率列 AUC 与 docs 旧值 0.694 的差异来源（已核对）

同一 task6 OOF 概率列，按不同评估单位/汇总方式算出的 AUC(≥5) 差异很大：

| 口径 | ≥5 | ≥6 | 说明 |
|---|---|---|---|
| 面板级·逐目标宏平均 | **0.807** | **0.829** | 本文推荐作为 Q1 主指标（34 / 12 个有正例目标） |
| 面板级·pooled（全局混池） | 0.906 | 0.936 | 被各目标基准率差异抬高，**不宜作头条** |
| 个人级·逐目标宏平均（加权） | 0.659 | 0.576 | 品评员个体噪声仍在，接近 docs 旧值 |
| 个人级·pooled（加权） | 0.813 | 0.794 | 同样被基准率差异抬高 |
| 用 ET 连续分 `unconstrained_grade` 排序·个人级 pooled | 0.682 | 0.651 | 与 docs 0.694/0.677 最接近 |

即：**0.694/0.677 是个人级逐目标（或 pooled）口径的旧数，0.807/0.829 是面板级逐目标宏平均口径；差异主要来自评估单位（个体 vs 面板）与汇总方式（pooled vs macro），不是模型或数据变了。** 论文应明确写"面板级、逐目标宏平均、概率列 `probability_rating_ge5/6`"，不要混用 0.694 与 0.807；若评委追问，可同时给出个人级 0.659 作为保守下界。

## 3. 论文一句话写法（Q1 不被"召回 0"自伤）

> Q1 的交付口径是每款酒每个感官维度的 0–6 分级，因此主指标为啤酒级面板中位 MAE 0.393、完全命中 74.1%、±1 级 91.7%；5–6 分在 12,500 个单元格中仅占 1.6%/0.15%，故高分识别不以精确召回为主指标，而报面板级逐目标 AUC（≥5 为 0.81、≥6 为 0.83）及累积概率 top-k 决策的高分精确率 25.5%（≥5，基准率 1.58%，约 16× lift）。旧版"5/6 精确召回为 0"源于以整数条件中位数直接输出等级，属决策规则导致的截断，不是排序信息缺失。

复现：`.venv\Scripts\python.exe experiments\lead_review\q1\decision_layer.py`（阈值/边际匹配对照）与 `decision_topk.py`（推荐 top-k 表与 AUC 分解）。

## 4. 决策层正式化（2026-10-28，已进入正式产物链）

原型结论已落成正式代码，不再依赖 `experiments/` 脚本：

- 规则库：`src/beer_models/q1_decision.py`（纯函数、无 IO；`topk_decision`、`calibrate_panel_mean`、`isotonic_calibrate`、`quantile_calibrate`、`fold_training_rate`、`fold_style_mean`、`linear_to_grade`、`high_grade_stats`）。
- 协议：`configs/q1_decision_20261028.json`（先冻结规则，再出数；非 force top-k，`k5/k6 = round_half_even(训练折观测率 × 组内格数)`，平票按行位置）。
- 官方脚本：`scripts/q1_official_20261028.py --source <附件目录>`，输出 `outputs/q1_decision_20261028/` 下 6 张表 + `protocol.json`/`verification.json`/`summary.json`。
- 单测：`tests/test_q1_decision.py`（21 个合成数据用例，`PYTHONPATH=src` 下 21/21 OK）。

**正式产物数字（OOF，12500 格；与冻结 handoff 逐表 1e-9 对齐，`verification.json` 15/15 True）**

| 口径 | 面板中位 MAE | 面板均值 MAE | 完全命中 | ±1 命中 | ≥5 精确率/召回 | =6 精确率/召回 |
|---|---|---|---|---|---|---|
| 旧整数条件中位数（对照） | 0.3928 | 0.6123 | 74.13% | 91.69% | 0 / 0 | 0 / 0 |
| **非 force top-k（新交付主表）** | **0.4080** | 0.6231 | 73.88% | 91.31% | **25.49% / 26.26%**（204 格，F1 0.2587） | **6.25% / 5.26%**（16 格，F1 0.0571） |
| 个体级加权（信息下界对照） | 0.7952 | — | 60.49% | 78.30% | — | — |
| 面板中位 oracle 在个体级（下界） | 0.6636 | — | 65.87% | 81.87% | — | — |

标定组（只改面板均值的连续输出，不改整数等级表）：

| 标定方式 | 面板均值 MAE | 面板中位 MAE | 完全命中 | ±1 命中 |
|---|---|---|---|---|
| ET 整数中位数（当前） | 0.6123 | 0.3928 | 74.13% | 91.69% |
| 折外逐目标线性 | 0.5027 | 0.7172 | 41.79% | 93.06% |
| 折外逐目标 isotonic | 0.5026 | 0.7176 | 41.81% | 92.95% |
| **折外逐目标线性 + 风格均值** | **0.4622** | 0.6895 | 47.54% | 91.11% |
| 折外分位数（边际）标定 | 0.6503 | 0.4195 | 73.69% | 90.48% |

要点：
- 新主表把"高分召回 0"救成 ≥5 精确率 25.5%/召回 26.3%（基准率 1.58%，约 16× lift），代价是面板中位 MAE +0.0152、命中率 −0.25pt，属可接受 trade-off；若评分只看等级 MAE，保留旧表作对照列即可（两张表同时交付，口径写清）。
- 高分覆盖率受真值限制：仅 34/50 个维度出现过面板中位 ≥5（共 198 格）、12/50 出现过 6（19 格）；其余维度本就不该强行给 5/6。force≥1 变体（每维至少 1 个高分）已证实劣化到 MAE 0.469、6 分精确率 0.66%，不可取。
- 标定后的面板均值单独成表（`q1__calibrated_panel_mean_12500.csv`）：**折外线性+风格均值把面板均值 MAE 从 0.6123 降到 0.4622（−24.5%）**，但整数等级仍以 top-k 表为主，避免两套数字互相污染。
- 更正一条此前记录：style 标定在原型 `panel_analysis.py` 里并非"秩亏即回落均值"；模块最初对"某折内某特征恒为 0 常数"误判为秩亏。已改为仅在秩 1（全部特征无信息）时回落训练均值，其余用最小范数解，与 sklearn/参考实现一致后，style 行与 handoff 完全对齐（0.689469/0.462234/0.47536/0.91112）。

复现（正式链，约 40 秒）：

```powershell
cd "C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件"
$env:PYTHONPATH='src'
.venv\Scripts\python.exe -m unittest tests.test_q1_decision -v
.venv\Scripts\python.exe scripts\q1_official_20261028.py --source "C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件"
```

产物：`outputs/q1_decision_20261028/q1__submission_long_12500.csv`、`q1__submission_wide_250x50.csv`、`q1__evaluation_long_12500.csv`、`q1__metrics_summary.csv`、`q1__calibrated_panel_mean_12500.csv`、`q1__high_grade_coverage_by_target.csv`、`protocol.json`、`verification.json`（15/15 True）、`summary.json`。

## 5. 审查修复（46ca932 四项，2026-10-28 晚）

**本节使 §4 与 §1 中以下数字作废，不得再引用**：面板均值折外线性 `0.502721`、风格+线性 `0.462234`、isotonic `0.502598`、quantile `0.650288`；“5 分 204 格 / 精确 25.49% / 召回 26.26%”的“5 分”叫法；以及据此写的“标定后 0.4622”。替代数字见下表。

### 5.1 修复 1：标定泄漏（审查 #1）

旧实现 `fold_style_mean` + `calibrate_panel_mean` 在同一个外层折外矩阵上互相嵌套：风格均值用全部数据折外计算，标定器又在外层验证折标签参与的特征上拟合，导致外层验证折标签经“其他折特征”间接进入参数。新实现为**严格嵌套**：对每个外层折 f，只用外层训练集（其余两折）内部再做 2 折内层切分，重训官方 ET（同 configs/final_protocol.json 的 ET 参数）得内层预测，风格统计与四类标定器都只在外层训练集上拟合，再预测外层验证折。代码：`beer_models.q1_decision.fit_nested_linear/fit_nested_isotonic/fit_nested_quantile/fold_style_mean` + `scripts/q1_official_20261028.py` 的 `build_inner_predictions/nested_calibrations`。

**扰动泄漏测试**（`verification.json` 的 `checks.q1_leakage_perturbation_invariant`）：把外层验证折 beer '2' 的面板加权均值 +0.25，重新走完整标定流程，四种方法的最终预测 `max|Δ| = 0.0`（要求 ≤1e-12）。每外层折校准器只用 8350 个外层训练单元格拟合、对 4150 个验证单元格预测（`checks.q1_calibrator_fit_on_outer_training_only=True`）。

| 面板均值标定 | 修复前（泄漏，作废） | 修复后（严格嵌套） | 差值 |
|---|---|---|---|
| 折外逐目标线性 | 0.5027214 | **0.5054899** | +0.00277 |
| 折外 isotonic | 0.5025976 | **0.5056423** | +0.00304 |
| 风格均值 + 线性 | 0.4622338 | **0.4660719** | +0.00384 |
| 分位（quantile） | 0.6502882 | **0.6542616** | +0.00397 |
| ET 原始连续分直接当均值 | 0.6123350 | 0.6123350 | 0（无标定拟合，不变） |

结论：泄漏把“风格+线性标定”抬高了约 0.0038，不改变量级，但**旧 0.4622 不能再当正式数**，正式口径改为 0.466072。个体级指标不受标定影响：加权 MAE 0.795214（legacy）/0.801403（top-k+cap），exact 0.604919/0.604061，±1 0.783035/0.782723（加权口径，分母权重和 12500）。

### 5.2 修复 2：父子项约束（审查 #3）

24 对 `template_rule=checkbox_zero_or_parent`（`outputs/task1_acceptance_20261025/scorecard_scope.csv`）：A/F × {malt,hops,esters}_all → 4 个子项。top-k 决策后加确定性后处理 `child_grade = min(child_grade, parent_grade)`（`beer_models.q1_decision.enforce_parent_child`，多级时先父后子）。

- 修复前 top-k 表违规 **12 处**，修复后 **0 处**（`checks.q1_precap_parent_violations_match_audit` / `q1_parent_violations_zero_after_cap`），且只下调、不改其他格（`q1_cap_only_lowers_violating_cells`，changed_cells=12）。
- 分布：pre-cap `{0:8279,1:856,2:1361,3:1648,4:152,5:188,6:16}` → 官方（cap 后）`{0:8279,1:856,2:1363,3:1655,4:153,5:180,6:14}`；旧整数中位 `{0:8324,1:863,2:1378,3:1701,4:234}`。

| 指标（啤酒级） | 旧整数中位 | top-k（cap 前） | **官方 top-k（cap 后）** |
|---|---|---|---|
| 面板中位 MAE | 0.3928 | 0.4080 | **0.40648** |
| 面板均值 MAE | 0.612335 | 0.623064 | **0.621288** |
| 完全命中 | 0.74128 | 0.73880 | **0.73880** |
| ±1 准确率 | 0.91688 | 0.91312 | **0.91336** |
| 逐目标宏平均 Spearman | 0.328348 | 0.290312 | **0.290356** |
| ≥5 预测格/精确/召回 | 0 / 0 / 0 | 204 / 0.254902 / 0.262626 | **194 / 0.262887 / 0.257576** |
| =6 预测格/精确/召回 | 0 / 0 / 0 | 16 / 0.062500 / 0.052632 | **14 / 0.071429 / 0.052632** |

说明：`metrics_summary` 里 `q1_topk_nonforced` 行是 **cap 前**阶段（0.408 / 204 / 16，冻结交付契约仍断言这一行），`q1_topk_nonforced_parent_capped` 行才是**官方提交**（0.40648 / 194 / 14）；两者不得互相置换或混引。`q1__high_grade_coverage_by_target.csv` 因此同时给出 `n_predicted_ge5/6`（cap 前）与 `n_predicted_ge5/6_parent_capped`（官方）。

### 5.3 修复 3：指标口径（审查 #4）

“5 分 204 格、精确 25.49%、召回 26.26%”实为 **≥5 口径**（188 个 5 分 + 16 个 6 分 = 204）；真正的 exact5 是 188 个预测格。权威四口径表：`outputs/q1_decision_20261028/q1__metric_calibers.csv`（列：scope, prediction_set, caliber, n_predicted, true_positive, precision, n_observed, recall）。

啤酒级（cap 后官方表）：

| caliber | n_predicted | true_positive | precision | n_observed | recall |
|---|---|---|---|---|---|
| exact5 | 180 | 38 | 0.211111 | 179 | 0.212291 |
| exact6 | 14 | 1 | 0.071429 | 19 | 0.052632 |
| ge5 | 194 | 51 | 0.262887 | 198 | 0.257576 |

cap 前同口径：exact5 188/39/0.207447/179/0.217877；exact6 16/1/0.0625/19/0.052632；ge5 204/52/0.254902/198/0.262626。个体级（加权品评员口径，分母 12500）：cap 后 exact5 180/41.6829/0.231572/524.9898/0.079398；exact6 14/2.1959/0.156849/124.8476/0.017589；ge5 194/70.2107/0.361911/649.8375/0.108044；legacy 三口径预测/命中均为 0（信息下界：面板中位 oracle 个体加权 MAE 0.663567）。

### 5.4 修复 4：决策规则接入运行时（审查 #2）

- `current_runtime.train` 现在把正式决策参数写进 bundle：`q1_decision = beer_models.current_runtime.q1_decision_bundle(...)`，含 rule、forced=False、middle_cap=4、`rate_ge5/rate_6`（逐目标观测高分比例，来自固定全量训练面板加权中位）、24 对 parent/child 索引、实现名与 `implementation_sha256`；`training.json` 同步记录。
- `inference.run` 检测到 `q1_decision` 后调用 **同一个** `beer_models.q1_decision.runtime_decision` 生成官方等级，写入 `estimated_grade_0_6`，同时保留 `grade_legacy_integer_median` 与 `probability_rating_ge5/6`；`prediction_info.json` 记录 `grade_method` 与 `decision_rule`。`scripts/current.py predict q1` 无需改动即走此路径（`scripts/current.py:66` 调 `inference.run`）。
- 一致性证据（`experiments/lead_review/q1/runtime_check.py` + `.log`，真实执行 train→predict）：bundle / 当前模块 / prediction_info 三处实现哈希全部 `e972cfe42043b35aa65c7e1808e0745cb0ad0cec923b3e6bd3fcfcea9c6f1e99`；`official_vs_runtime_decision_max_abs_diff=0.0`；`official_vs_topk_pieces_max_abs_diff=0.0`；`legacy_column_equals_model_median=0.0`；`parent_violations_official=0`；行数 long=13000/primary=250/wide=250。全量拟合（in-sample，非 OOF）等级分布 `{0:8132,1:885,2:1415,3:1549,4:305,5:195,6:19}`，不得与 OOF 的 194/14 混用。
- 回归测试：`tests/test_q1_decision.py` 33 项、`tests/test_current_runtime_q1_decision.py` 3 项全过；全量 `discover -s tests` **329 项 OK（24 skipped）**。

### 5.5 复现命令

```
$env:PYTHONPATH='src'
.venv\Scripts\python.exe scripts\q1_official_20261028.py --source "<附件目录>"
.venv\Scripts\python.exe experiments\lead_review\q1\runtime_check.py   # 约 30s，重训一次并验证 train->predict 一致
.venv\Scripts\python.exe -m unittest discover -s tests
```

产物：`outputs/q1_decision_20261028/`（10 文件，含新增 `q1__metric_calibers.csv`；pre-fix 目录备份在 `outputs/q1_decision_20261028_pre_leakfix/`，只作追溯）。协议：`configs/q1_decision_20261028.json`（revision 2，含 `verification.metric_row_semantics` 与 `delivery_contract` 说明）。

### 5.6 剩余风险与需 Lead 处理

- 交付层 `src/beer_models/delivery_current.py` 与其测试断言的是 cap 前那一行（0.408/204/16）；当前全量测试通过，但 README/docs 若改写成官方 cap 后数字（0.40648/194/14），需要交付层负责人同步更新其断言；本任务未改 `delivery*.py`。
- 逐目标宏平均 Spearman 从旧整数中位的 0.328348 降到 0.290356（top-k 把大量格改成 5/6 后的排序代价），论文里应说明这是决策层的排序-等级权衡，不是模型退化。

## 6. 面板均值直接回归：泄漏修复后的重算（2026-10-28 晚，task-15）

### 6.1 泄漏定位与修复

`src/beer_models/q1_panel_mean.py` 本身只有指标函数（MAE/RMSE/±1/宏 Spearman/配对 bootstrap/四门判定），没有 OOF 或标定逻辑；同类泄漏在 `scripts/q1_panelmean_20261028.py` 的旧 `baseline_predictions()`：`fold_style_mean(panel_mean_reference, style, target, fold)` + `calibrate_panel_mean(...)` 直接在整张 OOF 表上逐折拟合、互相嵌套，与审查 #1 是同一路径。

已改为严格嵌套（task-12 口径）：

- 每个外层折只用**外层训练啤酒**重训官方 raw-panel 模型（2 个 inner 折），得到 inner OOF 整数等级；风格均值只用 inner 训练啤酒的 `panel_weighted_mean` 计算；
- 逐目标标定器（style+linear 与 linear-only）只用上述 inner OOF 单元拟合，再预测该外层验证折；
- 实现：`outer_style_base` / `build_inner_predictions` / `nested_baseline`（脚本内）+ `beer_models.q1_decision.fit_nested_linear/apply_nested_linear`。

复现校验（`verification.json` 全 True，18 项）：

| 校验 | 值 |
|---|---|
| nested style+linear 基线 panel-mean MAE | 0.46607189992522735（命中 task-12 官方数 1e-9） |
| nested linear-only 基线 | 0.5054899408175908（命中 1e-9） |
| round 后 ±1 | 0.91416（命中 1e-12） |
| truth 对齐官方分布 | max abs diff 1.33e-15，中位错格 0 |
| 每折标定拟合单元 / 验证单元 | 8300/4200、8350/4150、8350/4150，合计 12500 |
| 每折拟合啤酒 | 166/167/167，均不含该折验证啤酒（脚本内 SystemExit 断言） |

候选表未受影响：两次运行的 `q1__panel_mean_pred_12500.csv`、`q1__panel_mean_pred_hgb_12500.csv` 与修复前逐字节相等，候选逐重复 MAE 也完全一致——修复只改 baseline 与 gate，不改候选。

### 6.2 基线：泄漏版 vs 严格嵌套版

| 基线 | panel-mean MAE | exact（round vs 中位） | ±1（round） | 宏 Spearman | 与 nested-linear 列 Pearson |
|---|---|---|---|---|---|
| 旧泄漏版（已作废） | 0.4622338 | 0.47536 | 0.91112 | 0.357544 | 0.961920 |
| **严格嵌套 style+linear（新基线）** | **0.4660719** | 0.46344 | 0.91416 | 0.353977 | 0.969966 |
| 严格嵌套 linear-only | 0.5054899 | — | — | — | — |

即旧基线偏乐观 0.00384（MAE）与 0.00304（±1），作废；新基线本身在 exact/±1 上反而略好（0.46344/0.91416）。

### 6.3 候选比较与四门判定（ET 直接回归 / HGB 直接回归）

| 方法 | panel-mean MAE（3 重复均值，gate 用） | 相对新基线改善 | 配对 bootstrap 95% | 三重复 MAE |
|---|---|---|---|---|
| extra_trees_direct | 0.4518585 | **+0.0142134** | **[0.0082081, 0.0197454]** | 0.4519723 / 0.4512179 / 0.4523854 |
| hist_gradient_boosting_direct | 0.4691090 | −0.0030371 | [−0.0111281, 0.0042249] | 0.4691090 ×3（HistGBR 无采样随机性） |

四门（预注册，门槛写死在 `configs/q1_panel_mean_20261028.json`；gate4 为绝对下限 0.9119 = 官方整数等级 ±1 0.9169 − 0.5pt，不随基线改正而重锚）：

| 方法 | g1 MAE | g2 bootstrap | g3 每重复 | g4 ±1 | all |
|---|---|---|---|---|---|
| extra_trees_direct | ✓（0.4518585 ≤ 0.4560719） | ✓（下限 0.0082081 > 0） | ✓（三重复均 < 0.4660719） | ✗（0.91136 < 0.9119，差 0.00054 ≈ 0.05pt） | ✗ |
| hist_gradient_boosting_direct | ✗ | ✗ | ✗ | ✗（0.89976） | ✗ |

panel-mean MAE 上 ET-direct 确实更好（pred 平均后 0.4515784，相对改善 3.05%），泄漏修复让这个优势数字变大（改善 0.0103752→0.0142134，bootstrap 下限 0.0044690→0.0082081），但 round 后等级 ±1（0.91136）仍低于预注册绝对下限 0.9119，因此 **verdict 不变：`do_not_adopt_keep_baseline`**。

关于 gate4 的两种读法（需 Lead 明确，不自行采用）：相对口径“不低于修复后基线 −0.5pt”= 0.90916 时 ET-direct 会四门全过（`tests/test_q1_panel_mean.py::test_gate4_keeps_absolute_floor` 两种读法都已编码）。把下限从 0.9119 改为 0.90916 属于改动预注册门槛，本任务按原门槛执行。

### 6.4 结论与论文表述

- 正式交付不变：整数等级（top-k+cap）+ 严格嵌套标定面板均值（0.4660719）；ET 直接面板均值模型按预注册门槛不采纳，`adoption_recommendation=do_not_adopt_keep_baseline`。
- 可写进论文：“在严格嵌套的折外标定下，直接回归面板加权均值的 Extra Trees 把 panel-mean MAE 从 0.4661 降到 0.4519（改善 0.0142，配对 beer-bootstrap 95% [0.0082, 0.0197]，三个重复全部改善）；但其 round 后等级 ±1 为 0.9114，低于预注册下限 0.9119（0.05pt），故按协议不采纳，保留现有交付。该差异不是因果结论，只是固定折上的预测比较。”
- 同时必须说明：旧数字 0.4622338 / ±1 0.9119 是泄漏标定产物，已作废；任何文档不得再引用旧基线或旧改善 0.010375。

### 6.5 产物、复现与备份

产物 `outputs/q1_panelmean_20261028/`（7 文件）：`q1__panel_mean_pred_12500.csv`、`q1__panel_mean_pred_hgb_12500.csv`、`q1__panel_mean_metrics.csv`（baseline 行 0.466072/0.46344/0.91416）、`q1__panel_mean_comparison.json`（baseline 块含 nested_ci_style_linear / nested_ci_linear / calibration_rows）、`protocol.json`、`verification.json`（18 checks 全 True）、`summary.json`（`do_not_adopt_keep_baseline`、逐门、bootstrap）。旧文件备份在 `outputs/q1_panelmean_20261028_pre_leakfix/`。

```
$env:PYTHONPATH='src'
.venv\Scripts\python.exe scripts\q1_panelmean_20261028.py --source "<附件目录>"   # 约 6 分钟（含 6 次 inner 重训）
.venv\Scripts\python.exe -m unittest tests.test_q1_panel_mean -v   # 22 项全过
.venv\Scripts\python.exe -m unittest discover -s tests              # 全量 OK（24 skipped）
```

协议 `configs/q1_panel_mean_20261028.json` revision 2：`leak_fix` 段写死旧/新基线路径与 1e-9 复现要求；`reported_baseline` = 0.46607189992522735 / linear 0.5054899408175908 / ±1 0.91416；gate1/3 以新基线为参照，gate4 保持绝对下限 0.9119 并记录“不重锚”的理由。

## 7. 去批次依赖的正式评分（T1，task-17，2026-10-28）

### 7.1 问题与规则

旧正式决策用 `topk_decision`：每个 `(target, outer_fold)` 组内 `k = round(rate * n_rows)`，再把本批
概率最高的 k 个格子提为 5/6。`n_rows` 是**本次请求的批大小**，因此同一款酒的等级会随批次变化：
把 12500 格拆成 12500 次单条请求，≥5 的格子从 204 变成 0（整批与逐条差 217 格）。这使"部署后的
单条预测/新批次评分"无法复现交付表，属于实现层面的正确性缺陷（不是精度问题）。

T1 改为逐样本固定阈值规则（`configs/q1_decision_det_20261028.json` 先冻结）：

- 每个 `(target, outer_fold)` 用**该折以外的格**估 `tau_ge5`、`tau_6`：`r = 训练格中面板中位≥5/≥6 的比例`，
  `k = ceil(r * n_train)`（clip 到 `[1, n_train]`），`tau = 训练概率第 k 大`；`r=0` 或 `n_train=0` 时
  `tau=+inf`，该 target 本折不输出 5/6（已列出回退清单）。
- 逐格判定：`p6 >= tau_6 -> 6`；否则 `p_ge5 >= tau_ge5 -> 5`；否则 `min(legacy_integer_median, 4)`。
  与批大小、批内排名、行位置无关。
- 24 对父子项 cap 作为确定性后处理，父子违规 0。
- 运行时：`current_runtime.train` 用 `risk_full_training.csv.gz` 的 `source=extra_trees` 内层 OOF 概率
  （250×50，训练面板内）估每 target 的 `tau`，写进 bundle 的 `q1_decision`（`rule=fixed_threshold_quantile`、
  `tau_ge5`/`tau_6`、`implementation=beer_models.q1_decision.runtime_threshold_decision`、
  `implementation_sha256`）；`inference.run`/`official_scores` 调同一函数生成官方等级，并保留
  `grade_legacy_integer_median`、`probability_rating_ge5/6` 列；旧 top-k 参数只作历史记录放在
  `legacy_topk`，不再作为官方规则。

### 7.2 结果：新旧等级与指标（真跑，outputs/q1_decision_det_20261028/）

| 方案 | 等级分布 0/1/2/3/4/5/6 | 面板中位 MAE | 完全命中 | ±1 | 宏 Spearman |
|---|---|---|---|---|---|
| 旧整数条件中位（参考） | 8324/863/1378/1701/234/0/0 | 0.392800 | 0.741280 | 0.916880 | 0.328348 |
| 旧 top-k+cap（历史基线） | 8279/856/1363/1655/153/180/14 | 0.406480 | 0.738800 | 0.913360 | 0.290356 |
| **T1 固定阈值（新正式）** | **8096/856/1355/1667/146/321/59** | **0.479680** | **0.721680** | **0.896240** | **0.286810** |

| 方案 | ≥5 预测数 | 精确率 | 召回率 | 6 预测数 | 精确率 | 召回率 |
|---|---|---|---|---|---|---|
| 旧整数中位 | 0 | 0 | 0 | 0 | 0 | 0 |
| 旧 top-k+cap | 194 | 0.262887 | 0.257576 | 14 | 0.071429 | 0.052632 |
| **T1 固定阈值** | **380** | **0.094737** | **0.181818** | **59** | **0.050847** | **0.157895** |

四口径（beer 级，详见 `q1__metric_calibers.csv`）：T1 exact5 321 格/命中 23（精确率 0.0717、召回 0.1285），
exact6 59/3（0.0508/0.1579），≥5 380/36（0.0947/0.1818）；旧 top-k 为 exact5 180/38、exact6 14/1、≥5 194/51。
个体级加权（分母 12500）：T1 MAE 0.862902 / 完全命中 0.591018 / ±1 0.768867；旧整数中位 0.795214/0.604919/0.783035。

### 7.3 批次一致性与阈值纯度（verification 全 True）

- 逐格等级一致性，全部 `mismatch=0`：OOF 网格的逐条请求、每 10 条分块、不等长分块 `[1,7,13,199]`、
  打乱、追加无关行；以及运行时 `current_runtime.train → joblib → inference.run` 链路上的逐条、每 10 条、
  打乱、追加行（见 `q1__batch_invariance.csv`，9 个场景全 pass）。
- 阈值纯度：把外层验证折的真实面板中位加噪（N(0.5,0.1) 后 clip）重算阈值与等级，
  `tau max|Δ|=0.0`、`grade max|Δ|=0`（验证折标签不进入任何阈值或拟合）。
- 运行时与画像入口：`inference.run` 的 `scores_all52_long.csv` 与 `official_scores` 逐格一致
  （`entry_mismatch=0`），bundle 内 `implementation_sha256` 与源码 sha256 一致（脚本内断言）。
- 历史目录 `outputs/q1_decision_20261028/` 前后 sha256 全等（未改动旧交付）。
- 回退清单：无 ≥5 的 target 16 个（A_esters_flower, A_hops_citrus, A_hops_woody, A_malt_grain,
  F_esters_ethac, F_esters_flower, F_hops_woody, F_malt_grain, X4vg, barnyard, coriander, diacetyl, dms,
  metallic, stale_hops, t2n）；无 6 的 target 38 个（见 summary.json）。

### 7.4 代价与"过度提升"诊断（必须如实写进论文/汇报）

T1 换来了批次无关性，但**精度变差**：面板中位 MAE 0.40648→0.47968（+0.0732），完全命中 0.7388→0.72168
（−0.0171），±1 0.91336→0.89624（−0.0171），个体级 MAE 0.795214→0.862902。高分侧：≥5 预测数 194→380，
精确率 0.2629→0.0947，召回 0.2576→0.1818；6 分 14→59，精确率 0.0714→0.0508。

原因可定位到"固定阈值在 OOF 分布漂移下无法约束提升数"：`clove` 在 fold 0 的训练折里只有 1/166 个
面板中位≥5（0.6%，`tau=0.005356`），而该折 84 款酒的 `p_ge5` 全为同一个值 0.006574（模型基本输出先验），
84 格全部越过阈值 → 单 target 就贡献 84 个 5 分（真值只有 1 个 ≥5）。提升数最多的 target：
clove 84、A_malt_all 20、bitternes 19、A_esters_all 17、alcohol 17、F_malt_all 17、A_esters_isoaa 16、
F_hops_noble 15、F_hops_all 15、A_malt_burn 13。旧 top-k 的"精确配额"把这类 target 的 5 分压到 1-2 个，
所以 MAE 更低，但代价是批大小一变结果就变（逐条时 5/6=0）。

这是设计取舍而非调参问题：按任务要求，**没有为接近旧分布去调 `k`/`tau`**（预注册规则原样执行）。
若 Lead 认为部署不需要逐条打分，可以保留旧 top-k 作为交付表；若需要（竞赛评分/后续接口）逐条可复现，
则 T1 是可辩护的正式规则，但必须同时报告上表的精度代价。可选的后续硬化（需另立预注册、非本次结果）：
对常量/并列概率增加"最小支持 + 并列按比例提升"的确定性规则，或用全训练分位与 OOF 分位做交叉校验。

### 7.5 产物与复现

产物 `outputs/q1_decision_det_20261028/`（10 文件）：`q1__evaluation_long_12500.csv`（含 `grade_topk_precap`、
`tau_ge5`、`tau_6`）、`q1__submission_long_12500.csv`、`q1__submission_wide_250x50.csv`、
`q1__metrics_summary.csv`、`q1__metric_calibers.csv`、`q1__batch_invariance.csv`、`q1__thresholds.csv`
（3 折×50 + 运行时 50 行，含 r/k/tau/是否回退）、`protocol.json`（config 全文 + 输入/模块/脚本 sha256）、
`summary.json`、`verification.json`（16 项全 True）。

```
$env:PYTHONPATH='src'
.venv\Scripts\python.exe scripts\q1_official_det_20261028.py --source "<附件目录>"   # ~2 分钟（含一次运行时 bundle 重训）
.venv\Scripts\python.exe -m unittest tests.test_q1_decision_det -v                    # 14 项
.venv\Scripts\python.exe -m unittest discover -s tests                               # 全量
```

协议 `configs/q1_decision_det_20261028.json` 冻结于计算之前（`frozen_before_results=true`、
`post_result_threshold_retuning=false`），历史目录 `outputs/q1_decision_20261028/` 未被触碰。
