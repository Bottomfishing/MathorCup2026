# 论文交接包（paper_handoff_20261028，r2）

给论文执笔者的取数包。**先读 `论文取数与口径.md`**，再用 `paper_claims.csv` 逐条引用。
本目录 r2 已按提交 `0a70627` 的正式口径重算，r1 数字全部作废。

## 内容
- `论文取数与口径.md`：三问主张、口径披露、三套 Q1 结果并排、Q2/Q3 数字、论文结构与 8 条答辩、复现命令。
- `paper_claims.csv`：33 条可写主张，每条带 `value` 与 `evidence` 文件。
- `tables/`：论文要引用的官方表（q1__*、q1_history__*、q2__*、q2_ablation__*、q3_generation__*、q3_migration__*、q3_evidence__*）。
- `figures/`：交付包 4 组图（PNG+SVG）。
- `verification.json`：本包的数字一致性检查（读取官方 outputs，断言关键值）。

## 三条必须遵守的写法
1. 0–6 标签来自外部 Zenodo 原始品评记录（2197 条），3.xlsx 是空白模板。
2. Q1 头条用"正式：逐样本固定阈值"那一行（0.4797/72.17%/89.62%），批次 top-k（0.4065）只作对照。
3. Q3 生成谱不是可酿造配方；Tripel 迁移后风格判定 1/11，属"风格支持不确定"。

## 重生成
```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe experiments\lead_review\build_paper_facts_20261028.py
```
