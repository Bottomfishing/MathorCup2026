# MathorCup A：啤酒风味程序工作区

**Q2候选与证据已交付**：固定50目标150条Top-3，82项名称全部核对，保留bootstrap、方法不一致和化学证据不足项。见[Q2因素交付说明](docs/Q2因素交付说明.md)，结果`outputs/q2_delivery_20261024/`。下一阶段为Q3生成与迁移收口。

**Q1交付已补齐**：新样本预测命令及字段说明见[Q1评分交付说明](docs/Q1评分交付说明.md)，准确率/缺失汇总位于`outputs/q1_delivery_20261024/`。可用`complete_modeling.py --stage q1`仅重建Q1。下一阶段为Q2因素与化学解释。

**最新接手进度（2026-10-24）**：新增Q1嵌套模型选择、相对等级及缺失检验，以及Q3联合分布检验和缺陷保留迁移。结果入口`outputs/final_20261024/`，独立核验通过；该目录名不表示三问最终完成。后续按阶段完成后立即推送main。**先读[分阶段进度](docs/分阶段进度.md)**。论文由其他队员负责，不在本任务范围内。下文保留此前实验与历史交接说明。

2026-10-24追加：用户确认按明确假设继续。第一问已生成临时0～6**相对等级**，入口`outputs/q1_relative_grades_20261024_v1/`，规则与边界见`docs/相对分级假设.md`。不等于恢复原始品评分，也不代表官方认可。

当前实现阶段：P0数据审计、P1标准化目标基线、Q2因素解释与稳定性、Q3生成与迁移首轮实验。尚未交付正式0～6分评分或物理约束认证的迁移方案。

P0有效输出为 `outputs/p0_audit_20261023_final/`。阶段结论见 `docs/P0阶段交接.md`，验证记录见 `docs/P0验证记录.json`。P0的v1/v2目录仅是开发中间结果。

P1结果入口：`outputs/p1_review_20261023/baseline_report.md`；交接说明：`docs/P1阶段交接.md`。这里的P1 v1/v2分别是首轮与稳健追加实验，均保留，不等同于P0的开发中间目录。

最新进度（2026-10-24）：Q2新增样本外单因素/相关组置换、同风格对照及证据索引；Q3完成三种生成方法、900条谱、45,000个感官预测、有限候选迁移搜索及参考阈值敏感性。39项测试及独立结果复算通过。**下一次先读 `docs/续接记录.md`**；详细结果见 `docs/Q2阶段交接.md`、`docs/Q3阶段交接.md`、`docs/结果索引.md`。

当前有效汇总目录为 `outputs/q2_review_20261024_v3/`、`outputs/q3_review_20261024/`；Q3数值结果为 `outputs/q3_exploratory_20261024_v2/`，阈值情景为 `outputs/q3_threshold_scenarios_20261024_v2/`。这里的完成仅指探索程序运行完成，不能当作竞赛三问正式完成。

## 第一问基线运行

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install --index-url https://pypi.org/simple -r requirements-lock.txt
./run_baseline.ps1 -Source '原始赛道A/附件目录' -Output 'outputs/my_baseline'
./run_baseline.ps1 -Source '原始赛道A/附件目录' -Config 'configs/baseline_robust_v2.json' -Output 'outputs/my_robust'
.venv/Scripts/python.exe scripts/verify_baseline.py outputs/my_baseline
.venv/Scripts/python.exe scripts/verify_baseline.py outputs/my_robust
.venv/Scripts/python.exe scripts/report_baseline.py outputs/my_baseline --additional outputs/my_robust --report-output outputs/my_review
```

先完成P0审计；非当前机器的路径通过 `-Audit` 指定审计输出目录。当前默认指向已保存P0结果，程序会核对源文件哈希。

输出包括固定划分、内部选参、逐目标和逐风格指标、全部OOF预测、各折模型、额外缺失实验、配置及环境记录。只加载本项目生成且来源可信的joblib模型。追加稳健实验属观察首轮结果后的探索改进，不能冒充独立验证。

训练比较使用全部50个目标，主要汇总45个Aroma/Taste维度。原始评分尺度未识别，不提供伪装为原始评分的0～6逆变换。

## 运行

Python 3.12。可使用已经提供 pandas/numpy/openpyxl 的环境；独立环境安装：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-audit.txt
./run_audit.ps1 -Python .venv/Scripts/python.exe -Source '原始赛道A/附件目录'
```

当前机器已验证的 Python 路径：

```text
C:/Users/wb198/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe
```

可将上述路径作为 `-Python` 参数，无需修改共享依赖目录。源目录应包含 1.csv、2.csv、3.xlsx、4.csv、5.csv、6.csv、7.csv、8.xlsx 和数据说明.txt。

默认输出到时间戳目录。使用 `-Output` 指定一个新目录；程序拒绝覆盖现有目录，也拒绝向原始附件目录内写入。仅同时存在 run_manifest.json、summary.json 和 data_quality_report.md 的成功运行可用于交接，失败目录不得当作最终结果。

## 测试

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -m unittest discover -s tests -v
```

测试覆盖主键错配、重复样本、条码精度、非法数值、尺度识别、零参考、相关矩阵和缺失数据下的 Spearman 排名等风险。测试执行器须使用安装了 requirements-audit.txt 的 Python。

## 输出及使用边界

- `data_inventory.csv` / `run_manifest.json`：原始文件、代码哈希、运行环境。
- `sample_mapping.csv` / `numeric_profile.csv`：样本对应、缺失、数值范围；异常值仅标记。
- `feature_dictionary.csv` / `transform_checks.csv` / `calibrated_conflicts.csv`：别名、参考单位、尺度证据和冲突。
- `sensory_mapping.csv` / `scorecard_audit.json`：目标范围和模板评分检查。
- `matrix_checks.json` / `correlation_method_comparison.csv`：矩阵完整性与重算对照。
- `model_readiness.json`：原始评分尺度是否可用，防止无依据逆变换。
- `data_quality_report.md`：可交接给队员的审计结果。

运行通过不代表评分尺度已解决。不得把模板缓存零值作标签，不得把任意拉伸到 0～6 的值称为原始评分，不得自动用校准表覆盖冲突值。7.csv 是论坛评论相关数据，不能默认为专家评分。

完整开发安排见 `docs/A题程序开发总体规划.md`。AI 辅助范围及尚待队员复核事项见 `docs/AI使用记录.md`。
