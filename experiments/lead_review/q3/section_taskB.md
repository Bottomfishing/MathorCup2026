## 20. 决策规则切换后的重打分审计（task B，2026-10-28）

### 20.1 背景与口径
Q1 正式决策从 `non_forced_cumulative_probability_topk_with_parent_child_cap` 切到 T1 修复后的
`fixed_threshold_quantile`，并重训 `outputs/current_models/q1.joblib`（sha256
`82cf74966320e82d3a72a6585dcaa3c2c573a112c9c7f5e5e5fcae2fe80436ad`；旧模型
`3312c314bf26ca8f5f08b995ba0a1a8b701db23ed3b3359433890e95a5b9c3ae` 备份在
`outputs/current_models_pre_det_20261028/`）。task B 要求用新模型重跑 66 条前后谱 × 52 项感官画像，
旧产物整体备份后替换，不保留两套并存。

- 旧产物备份：`outputs/q3_migration_20261028_pre_det/`（13 文件，与交付目录同集）
- 交付目录重跑：`outputs/q3_migration_20261028/`（现为新模型产物）
- 复现命令：`PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/q3_official_migration_20261028.py --source "<附件目录>"`

### 20.2 模型与可复现性
- `protocol.json::sensory_pipeline.q1_model_sha256` = 新 sha；`replay_sensory_table()` 在新模型上
  重放 3432 行（66 谱 × 52 target × 19 列）**mismatch_cells=0，max|Δ|=1.3e-15**（容差 1e-12），
  模型 sha 与文件 sha 一致；交付表 sha256 `395e51fc44e5bba3…`。
- `verification.json` 共 **90 项 checks 全部 True**（原 78 项 + 12 项 det 审计），含模型 sha 断言、
  `det_rerun_cells_compared=3300`、五缺陷项变差 0、变化格数在 100 格上限内、证据计数未变。

### 20.3 新旧 grade 方向变化
可比格 = 3432 − 132（两个无原始标签的卡目标）= **3300 格**；变化 **65 格（升 19 / 降 46 / 不变 3235）**，
涉及 **30/33 条 case**。

| 角色 | 升 | 降 | 不变 |
|---|---|---|---|
| before | 2 | 19 | 1629 |
| after | 17 | 27 | 1606 |

- 变化幅度：降侧 −5 格 7、−4 格 3、−3 格 13、−2 格 20、−1 格 3；升侧 +3 格 11、+2 格 7、+1 格 1。
  即新规则对部分子目标平均降 2–3 格，并非“±1 格微调”，这一点在论文里必须如实写。
- 五缺陷项（dms / diacetyl / X4vg / t2n / stale_hops）：**up 0、down 0、unchanged 66/66**，
  `defect_worse_count=0`、`defect_worse_cells=[]` → 满足 Lead 的“worse 必须为 0”。
- 变化集中在感官子目标：lactic 11、A_esters_all 7、F_malt_all 5、A_malt_all 4、F_malt_burn 4、
  alcohol 4、aftertaste 3、A_malt_burn 3 等；单条 case 最多 7 格（S028_TO_Tripel before）。
- 方向解读：新规则总体“降多于升”（46 vs 19），说明逐样本固定阈值比旧 top-k 父子 cap 在该批谱上
  更保守；after 侧升格 17 格是新规则对迁移后谱更宽松的子目标，但没有任何缺陷项升格。

### 20.4 四口径计数与旧版对比（均未变）
| 口径 | 旧 | 新 |
|---|---|---|
| 化学判据 A/B/C（33 条） | 33/33/33 | 33/33/33 |
| claim_level | model_supported 9 / conditional 24 | model_supported 9 / conditional 24 |
| style nearest centroid | target 13 / other 20 | target 13 / other 20 |
| 缺陷状态（DMS improved/unchanged） | 9 / 24 | 9 / 24 |

`claim_level_changed_cases=[]`、`style_nearest_centroid_changed_cases=[]`、
`defect_status_changed_cases=[]`、`criteria_unchanged=true` —— 因为 A/B/C、claim_level 与缺陷状态
来自迁移数学与几何，不读 Q1；Q1 只影响 52 项感官画像数值。

### 20.5 新增交付文件
- `s2c_det_grade_changes.csv`：65 行 × 10 列（case_id/source_beer/target_style/role/target_52/
  grade_old/grade_new/delta/direction/defect_target）。
- `s2c_det_grade_change_matrix.csv`：66 行（33 case × before/after）× 11 列（cells/up/down/
  unchanged/defect_up/defect_down/defect_worse 等）。
- `summary.json` 新增 `det_rerun` 段（模型 sha 旧/新、3300/65/19+46、五缺陷逐项 up/down/unchanged、
  四口径旧/新计数与变化 case 列表、100 格上限判定）；`summary.md` 增一行审计结论。
- `verification.json` 新增 12 项 det checks 与 `det_rerun` 段。

### 20.6 测试
`tests/test_q3_migration.py` 新增 `DetRerunTests` 5 项（表行数与 evidence 一致、缺陷项不升格、
证据计数未变、矩阵 66 行与合计一致、模型 sha 匹配当前 bundle）；全套 **41 项 OK**（带 `Q3_SOURCE`
6.5 s，不设时 skip 路径同样 exit 0）。

### 20.7 诚实说明
- 52 项画像仍是“全训练模型预测”，不是样本外精度；迁移前后的差异（65/3300 = 2.0%）只反映
  Q1 决策规则变化，不等于迁移效果变化。
- 新旧 grade 都是 0–6 序数尺度上的**同一种谱的不同打分**；升/降只在缺陷项上有“好/坏”方向，
  其余子目标不能写成“感官改善/恶化”。
- 五缺陷项等级完全不变，与 20.4 的缺陷状态未变一致；这不表示缺陷谱没变，而是迁移谱本身的
  缺陷坐标与新规则下的等级判定都稳定。
