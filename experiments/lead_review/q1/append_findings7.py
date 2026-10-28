# -*- coding: utf-8 -*-
"""Append section 7 (T1 batch-independent decision) to FINDINGS.md."""
from pathlib import Path

path = Path(r'C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件\experiments\lead_review\q1\FINDINGS.md')
text = path.read_text(encoding='utf-8')
section = '''
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
.venv\\Scripts\\python.exe scripts\\q1_official_det_20261028.py --source "<附件目录>"   # ~2 分钟（含一次运行时 bundle 重训）
.venv\\Scripts\\python.exe -m unittest tests.test_q1_decision_det -v                    # 14 项
.venv\\Scripts\\python.exe -m unittest discover -s tests                               # 全量
```

协议 `configs/q1_decision_det_20261028.json` 冻结于计算之前（`frozen_before_results=true`、
`post_result_threshold_retuning=false`），历史目录 `outputs/q1_decision_20261028/` 未被触碰。
'''
assert '## 7.' not in text
path.write_text(text.rstrip('\n') + '\n' + section, encoding='utf-8')
print('appended', len(section), 'chars; total lines', len(path.read_text(encoding='utf-8').splitlines()))
