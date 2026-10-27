
## 18. 任务 16 修复：Q3 迁移 52 项感官画像表不可复现（N2）

**问题（复核方 `experiments/lead_review/audit2/REPORT.md` 第 6 项）**：交付的
`outputs/q3_migration_20261028/s2c_sensory_before_after_52.csv` 中 3300 个有标签格有 57 格与
"当前 `q1.joblib` + 交付谱"重放不一致（after 31 / before 26、18 个 case、max|Δ|=5）；
`protocol.json::sensory_pipeline` 只记了模型路径、未记模型指纹；且 Q1 正式口径已升级为
决策层 top-k + 24 对父子 cap（`estimated_grade_0_6` 不再是旧的整数条件中位数）。

**根因**：旧画像表是用旧 Q1 口径 / 旧谱跑出来的产物；生成脚本的映射仍按旧输出 schema
（`probability_rating_ge5` 无 `_x/_y` 后缀）拼列，在升级后的 `scores_all52_long.csv` 上直接
`KeyError: 'prob_ge5'`，说明该表无法从当前模型与交付谱重建。

### 18.1 修复内容（`scripts/q3_official_migration_20261028.py`）
1. **全表重生成**：`build_sensory_profiles` 走正式入口 `beer_models.inference.run`（与
   `scripts/current.py predict --task q1` 同一调用），未自拼旧列、未绕过决策层；
   `grade_raw = estimated_grade_0_6`（决策层）；`prob_ge5/prob_6` 取**决策层概率** `_y`，
   旧风险头概率另存 `prob_ge5_risk_head/prob_6_risk_head`（`_x`），
   `grade_legacy_integer_median` 一并保留，列结构在原表基础上只增不减（3432 行 = 33 case × 2 role × 52 target）。
2. **模型指纹与口径**：`protocol.json::sensory_pipeline` 增 `q1_model_path`（POSIX 相对路径）、
   `q1_model_sha256`、`q1_model_grade_semantics`（official_column / grade_method / point_semantics /
   decision_rule / legacy_column / parent_child_pairs=24 / implementation_sha256 / note）、`run_at`。
   实测 `outputs/current_models/q1.joblib` sha256 =
   `3312c314bf26ca8f5f08b995ba0a1a8b701db23ed3b3359433890e95a5b9c3ae`（51,450,843 B），与复核方给出值一致。
3. **可复现自查**：新增 `replay_sensory_table()`——从交付的
   `s2c_migration_spectra_224_long.csv` 读回 before/after 谱，用**当前** q1.joblib 重跑正式入口，
   与落盘表逐格比对（NaN 成对算一致，其余容差 1e-12），并把
   行数 / 比对列数 / 容差 / 不一致格数 / 逐列不一致 / max|Δ| / 模型 sha / 表 sha / 重放时间写入
   `verification.json::sensory_replay`；`summary.json::sensory_replay` 同步。

### 18.2 自查结果（真跑，2026-10-28，脚本 exit 0）
- `verification.json`：**78 项 checks 全 True**（原 72 项 + 6 项新检查），`failed_checks=[]`。
- 新增检查名：`sensory_replay_rows`、`sensory_replay_mismatch_cells`、
  `sensory_replay_max_abs_delta`、`sensory_replay_model_sha256`、
  `sensory_replay_delivered_table_sha256`、`sensory_decision_layer_grade_column`。
- 重放：**3432 行 × 19 列，mismatch_cells = 0，max|Δ| = 1.1e-15**（容差 1e-12），
  模型 sha 与 protocol 记录一致；交付表 sha256 =
  `a36621b787e18f72db67a80d0d752c65ccbfcc4f88340269ee32d2efedeb5e6d`（1,226,990 B）。
- 与修复前旧表逐列对比（`experiments/lead_review/q3/s2c_sensory_before_after_52_pre_N2fix.csv`）：
  `grade_raw` 有 **57 格 / 18 case 不同（after 31、before 26、max|Δ|=5）**——与复核方 N2 证据逐数吻合，
  即旧表正是这 57 格来自旧口径；`prob_ge1`、`raw_median_estimate` 0 格不同；
  `prob_ge5`/`prob_6` 因改为决策层概率分别有 3300/3234 格不同（旧表存的是风险头概率，
  现已完整保留在 `prob_ge5_risk_head`/`prob_6_risk_head` 两列），这是口径升级而非随机漂移。
- 复核方举例 `S209_TO_Tripel` after `F_malt_all`：旧表 2 → 新表 **5**（与当前模型一致），
  该格 prob_ge5=0.05995、prob_6=0.00580。
- 计数未变（如实核对）：claim_level 仍 `{conditional: 24, model_supported: 9}`、
  style_diagnostic `{Tripel: 0, Kriek: 11, Lambic: 2}`、
  defect_status `{DMS: improved 9/unchanged 24, Diacetyl: improved 9/unchanged 18/unknown 6,
  4VG: unchanged 33, t2n: unknown 33, Stale_hops: unknown 33}`。

### 18.3 测试
`tests/test_q3_migration.py` 增至 **36 项**（原 32 + 4）：
`SensoryProvenanceTests`（protocol 记录模型 sha/口径、交付表保留决策层与 legacy/risk-head 列、
verification 记录重放）、`SensoryReplayTests::test_sensory_table_replays_cell_for_cell`
（用存储谱 + 当前模型真重放，要求 0 不一致）。带 `Q3_SOURCE` 时 **36 tests OK**（5.8 s），
不设时 36 run / 25 skip、exit 0。重放测试自带 `experiments/lead_review/q3/tmp_q1_test_replay` 临时目录。

### 18.4 诚实说明
- 该表是**全训练集模型推断**，不是新啤酒实测、也不是 OOS 精度估计；模型指纹只证明"表来自这个模型"，
  不证明模型本身准。
- 逐格重放在 1e-12 内一致；两次独立运行的浮点末位可能相差 ~1e-15，因此表的 sha256 以
  `verification.json::sensory_replay.delivered_table_sha256` 记录为准。
- 本次只重生成画像表与其指纹/自查，**迁移谱、A/B/C 判据、claim_level 均未改动**，协议与门槛未动。

### 18.5 复现命令
```powershell
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe scripts\q3_official_migration_20261028.py --source '<附件目录>'   # 5.2 s，78 checks 全 True，含重放自查
$env:Q3_SOURCE='<附件目录>'
.venv\Scripts\python.exe -m unittest tests.test_q3_migration                                # 36 tests OK
```
