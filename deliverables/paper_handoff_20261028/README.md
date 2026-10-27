# paper_handoff_20261028 — 论文取数包

由 Lead 汇总 `experiments/lead_review/{q1,q2,q3}` 三份复核产出打包而成，**未修改任何正式交付、模型或协议**。
用途：论文执笔人直接取数写 Q1/Q2/Q3，不需要再跑实验；所有数字可追溯到脚本（见 `论文取数与口径.md §7`）。

## 先读

- **[论文取数与口径.md](论文取数与口径.md)** ← 主文档：每题该怎么写、数字表、图表清单、评委反驳应对、不要写的话
- `verification.json` ← 打包时自动校验的断言（Q1 12500 行与 top-k 分布、Q2 150 行/50 维、Q3 33/33）

## 产物

### Q1 感官评分（12,500 格 = 250 款 × 50 维）
| 文件 | 内容 |
|---|---|
| `tables/q1__submission_long_12500.csv` | **主结果表**（beer, target, family, grade_0_6, p_ge5, p6） |
| `tables/q1__submission_wide_250x50.csv` | 宽表 250×50，可直接放附录/提交 |
| `tables/q1__evaluation_long_12500.csv` | 主表 + 旧整数中位 + 面板均值/中位参考（评估用，不进正文） |
| `tables/q1__metrics_summary.csv` | 新旧口径指标 + 个体级 + 信息下界 + 标定实验 |
| `tables/q1__calibrated_panel_mean_12500.csv` | 折外逐目标标定后面板均值（0.6123→0.5027→0.4622） |
| `tables/q1__high_grade_coverage_by_target.csv` | 每维 5/6 观测覆盖率与预测数（说明"34/50 维才有 5"） |

### Q2 化合物—感官 Top-3
| 文件 | 内容 |
|---|---|
| `tables/q2__top3_all50_150.csv` | **主表** 50 维 × Top-3 = 150 条（含系数、图重要度、bootstrap 频率、机制、证据等级） |
| `tables/q2__top3_wide_50.csv` / `_fields.csv` | 宽表（只化合物名 / 带字段） |
| `tables/q2__stability_bootstrap.csv` | 200× bootstrap 稳定性（入选频率、Jaccard） |
| `tables/q2__method_jaccard.csv` | 4 方法 6 对 Top-3 Jaccard（逐维） |
| `tables/q2__network_edges_0_5.csv` / `_communities.csv` | 共表达图（231 节点/645 边/60 社区） |
| `tables/q2__evidence_tier_counts.csv` | 化学证据 A/B/C 分布 |
| `tables/q2__summary.json` | 全部 Q2 汇总数字 |
| `figures/fig_q2_*.png` | 稳定性 / Jaccard / 证据分布 三张图 |

### Q3 风格生成与迁移
| 文件 | 内容 |
|---|---|
| `tables/q3__migration_33cases.csv` | **主表** 3 目标 × 11 源酒迁移（λ*、质心距离降幅、缺陷前后值） |
| `tables/q3__migration_candidate_counts.csv` | 每款候选数 + A/B/C 三口径通过数（33/33 的证据） |
| `tables/q3__generation_quality.csv` | 7 种生成器 × 3 风格：方差比/近重复/边缘 W1 |
| `tables/q3__source_defects_11beers.csv` | 11 款源酒各缺陷浓度、阈值、超阈程度 |
| `tables/q3__style_defect_exceedance.csv` | 22 风格缺陷超阈分布（Tripel DMS 2.24× 阈值的证据） |

## 生成方式

```powershell
.\.venv\Scripts\python.exe experiments\lead_review\build_paper_handoff.py
```

脚本只读取、只复制/整形，不重训、不改正式产物；重跑会覆盖本目录 `tables/ figures/ verification.json`。
