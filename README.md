# MathorCup 2026 A：当前代码与交付

当前版本 **20261027，阶段三收敛版**。队友取数与转发统一使用 [deliverables/current_20261027](deliverables/current_20261027/README.md)，本机生成同名ZIP供直接转发。方法与入口以本页为准。Q1高分、Stale、完整五缺陷等问题按用户指示记录，后续优化；见[待优化问题](docs/待优化问题.md)。整理不宣称模型质量提高。

## 查看结果与论文取数

无需安装或重训。先读交付包 `conclusions.csv`、`unresolved.csv`，再按 `file_index.csv` 找 `tables/`、`figures/`、`evidence/`。CSV可直接在Excel打开，PNG/SVG用于插图，不另造易过期的Excel副本。

| 内容 | 包内tables/文件与方法 | 使用边界 |
|---|---|---|
| Q1正式评分 | q1__estimated_grades_all52.csv；ET整数条件中位数 | 250款第一轮折外评分；full_fit另列，不用于报准确率 |
| Q1性能 | q1__metrics_by_repeat/target/observed_grade.csv、confusion.csv、comparison_summary.csv | 三次重复MAE约0.794561，个人5/6精确召回均0；高分候选未采用 |
| Q1独立实验 | q1_independent__* | 局部高分收益与误报同时列出，未接入 |
| Q2历史标准化响应 | q2_standardized__top3_panel_all50.csv | 150条关联，非因果 |
| Q2原始分解释 | q2_raw__top3_raw_all50.csv、top3_raw_all52_slots.csv | 原始分GBR仅17目标51条非零SHAP；均值Ridge分列，零归因/无标签不补位 |
| Q3生成 | q3_generation__generated_profiles.csv、current__generated_official_Q1_scores52.csv.gz | Tripel/Kriek/Lambic各100条插值谱；画像是模型估计，有方差收缩/近重复 |
| Q3迁移 | q3_transfer__current_plan_status.csv、两端谱、chemical_changes.csv | 119号223输入四风险代理条件案例；18/20及9/12共同通过，附加扰动22/32仍有限；其余10款受限 |
| Q3两种画像 | current__transfer_official_Q1_scores52.csv、current__transfer_auxiliary_Q3_risks.csv | 正式Q1的224输入评分与Q3辅助223输入概率分开；双乙酰NA，要求保持实际未知值 |

## 数据与环境

Python **3.12**；`requirements-r3-lock.txt`含核心与SHAP锁定依赖，无新增依赖。原附件保留仓库外，本机路径为 `D:/mathor cup2026赛题/2026年MathorCup大数据挑战赛-赛道A初赛/附件`。固定训练需要附件1.csv、4.csv、8.xlsx，校验原审计哈希；原始评分可从固定公开版本获取或用 `--raw` 指定本地文件。

实际250款啤酒、226种测量属性加5个派生字段共231列；正式Q1排除7个汇总字段使用224列。2197条原始记录覆盖50目标，评分卡52项的两项乙醛无标签，分数及概率始终留空。原始附件、逐品评员记录、下载参考工作簿、模型二进制均不上传。

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-r3-lock.txt
.venv/Scripts/python.exe scripts/fetch_raw_sensory.py --no-proxy
.venv/Scripts/python.exe scripts/current.py status
```

## 训练、预测与刷新

从仓库根目录运行。每个输出目录须不存在，默认模型目录 `outputs/current_models/`。固定全量部署训练不重跑历史筛选、旧网格、SHAP研究或21步链。

```powershell
.venv/Scripts/python.exe scripts/current.py train --source '原始附件目录'
.venv/Scripts/python.exe scripts/current.py predict --input '新化学谱.csv' --output outputs/my_prediction
.venv/Scripts/python.exe scripts/current.py render --output outputs/my_render
```

输入列名/坐标与附件1一致，可含唯一beer列；缺失用训练填补，不能恢复测量，也不自动取指数。Q1输出52项评分、校准前后风险及缺失情况。Q3辅助预测显式加 `--task q3`，输出单列辅助风险与风格，不当正式Q1或安全认证。自选模型目录用 `--models`。固定谱刷新无需原始记录或重训。

训练复用**历史训练内折外概率**恢复已采用风险来源与校准，位于 `data/current/risk_full_training.csv.gz`，附原来源哈希；不读取旧模型缓存，不是新增验证。模型本机重建，不随包分享。

## 核验与打包

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe -m unittest discover -s tests
.venv/Scripts/python.exe scripts/current.py verify --models outputs/current_models --source '原始附件目录' --output outputs/my_check
.venv/Scripts/python.exe scripts/current.py package --rendered outputs/my_render --output outputs/my_teammate_delivery
```

verify核对当前口径、缺标签、NA保持、Q2排名与部署预测。加 `--package 交付目录` 检查包内文件哈希与排除项。不传models仅核验已提交表，不能说成重训验证。package不重训；不传rendered复用同正式Q1已核验画像。包内code/可独立训练、预测、刷新、核验；完整历史研究与重新打包使用Git仓库。

已从零模型缓存重建固定模型，对照250款13000行、生成谱15600行、迁移104行和Q3辅助风险/风格，最大差约1.91e-14。172项单元测试通过。实现检查不替代模型质量；详见[阶段三说明](docs/阶段三交付说明.md)。

## 采用理由与限制

保留Q1中位评分，因为候选的高分收益与误报/稳定性未同时达到预设采用标准；没有用MAE掩盖高分问题。Q2三种统计口径分别引用。Q3保留插值生成与119号固定删改谱；辅助模型不使用缺失双乙酰，化学仍要求保持实际未知值，尚非工艺已实现。Stale专用比较未通过采用依据，标未知。9/12与22/32均来自同250款开发数据的敏感性，非独立测试。

题目要求、实现条件与额外保守情景见包内current_constraints.csv约束台账；旧30概率条件是附加情景，不冒充题面。没有迁移后酿造/品评实测，模型内通过、物理可行与真实效果分开；同源原论文不算外部验证。

[历史依赖与复现](docs/历史依赖与复现.md)说明必须保留的旧目录。旧task6_final_20261026、用户原“交付”目录及旧日期文档是历史快照，不再作当前取数入口；原“交付”目录未修改。
