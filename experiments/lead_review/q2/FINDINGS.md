# Q2 化合物–风味分析：交付重建、稳定性与化学合理性（FINDINGS）

> 角色：Q2 负责人（q2-analyst）。日期 2026-10-27。仅写 experiments/lead_review/q2/，未改动 src/、configs/、deliverables/、outputs/。
> 全部数字由 `experiments/lead_review/q2/02_rebuild_q2.py` 用真实附件数据跑出（seed 固定，200 次 bootstrap），可在 2–3 分钟内复现（命令见文末）。

---

## 0. 一句话结论

- **原 17/50 覆盖不是题面要求的结果，是"GBR 原始分表"这一个子口径造成的**；团队 Ridge 主表本身已是 50/50 全覆盖。题面要求的是"每个感官维度 Top-3"，因此**"零归因不补排名"只对 2 个无面板标签的 Acetaldehyde 成立，对 50 个有标签目标属于自我加码**，必须改。
- 本报告重建了一张 **50/50 维 × Top-3 = 150 条、无空位** 的交付表（每维 3 个化合物 + bootstrap 入选频率 + 证据级 + 置信度 + 机理论述），并给出题面要求的两种稳定性指标：
  - **bootstrap Top-3 入选频率**：均值 **0.474**（中位 0.403），**57/150 个排名位频率 ≥ 0.5**，**20/50 个维度 Top-3 平均入选频率 ≥ 0.5**；
  - **多方法 Jaccard**：核心两法（图邻域加权 vs 普通 Ridge）均值 **0.600**，**45/50 维 ≥ 0.5**；6 对方法全体均值 **0.331**，**48/50 维至少一对方法 ≥ 0.5**，5/50 维六对均值 ≥ 0.5。
- **化学合理性**：150 条中 **A 级（身份标记物/6.csv 经验同号支持）8 条、B 级（机理同族或 6.csv 支持）70 条、C 级（探索性，须标注）72 条**；A+B 占 **52%**。弱信号维度（diacetyl、dms、t2n、clove 等）在 4.csv 目标与对应化合物之间本就近乎无关（|ρ| ≤ 0.15），**这不是分析没做，而是目标测量层面的信息上限**，已在表中如实标注为探索性并给出机理论述。

---

## 1. 问题 1：题面口径核实与"零归因不补排名"判断

### 1.1 数据 NA 结构（实测）

| 文件 | 形状 | NA 情况 | 结论 |
|---|---|---|---|
| 1.csv | 250 × 234（3 meta + 231 特征） | 9 列共 73 个缺失（单列 1–9 个） | 建模需中位数填补（bootstrap 时重做填补） |
| 2.csv | 231 × 231 | 无 NA | 对称相关矩阵 |
| 4.csv | 250 × 53（3 meta + **50 个标准化目标**） | **目标列 0 缺失** | **50 个目标全部可排名** |
| 5.csv | 50 × 51 | 无 NA | 感官–感官相关，**不提供化合物证据** |
| 6.csv | 44 × 51 | 无 NA | 44 化合物 × 50 感官交叉相关，可作经验佐证 |
| 7.csv | 250 × 189 | 3 整列全 NA（palate_acetaldehyde / palate_acetate / palate_dentist，共 750 格） | 184 列可用；与 4.csv 的 50 目标不冲突 |
| 8.xlsx | 综合参考 | — | 身份/描述符/阈值对照 |

**52 项口径**：52 = 47 主内容 + 5 补充（alcohol / aftertaste / body / co2 / overall）。
47 = 45 个有 4.csv 面板标签 + 2 个 Acetaldehyde（A_acetaldehyde、F_acetaldehyde：评分卡有、4.csv 与 summary 均无标签）。
→ **可排名 = 50 维（=150 条）**；仅 2 个 Acetaldehyde 的 **6 个排名位**无标签可留空。团队 `q2_standardized__top3_all52_slots.csv` 的 156 行 = 150 已填 + 6 空位，与上述一致。

### 1.2 17/50 的真实来源

- GBR 原始分表（`q2_raw__top3_raw_all50.csv`）中 `gbr_selected` 只勾选了 **17 个目标 × 3 = 51 条**，其余 99 个槽位是"零归因"、6 个是无标签空位（`validation__q2_rank_coverage.csv` 明确写有 `nonzero_rank_targets=17`）。
- 团队 Ridge 主表（`q2_standardized__top3_panel_all50.csv`）本身是 **50 目标 × 3 = 150 条全覆盖**。
- 因此 **17/50 是"GBR 子口径"的覆盖率，被误当成 Q2 总覆盖率**。题面要求"给出每个感官维度的 Top-3"，用"该模型归因为零所以不排名"来留空，属于**方法缺陷被表述成了交付规则**，评委自然会读成"Q2 没做完"。
- **正确表述**：50 个面板目标每维给 3 个候选，按统一规则从 219 个候选化合物中选出；零归因只能说明该模型下该目标解释力弱，不能作为"无关键因素"的证据（题面也从未要求显著性/因果）。

### 1.3 可写进论文的附加事实（已逐元素验证）

- **2.csv 实际是 1.csv 的 Spearman 矩阵，不是 Pearson**：把 2.csv 第 k 行/列与 1.csv 第 k 个数值列按**位置映射**（2.csv 行序 = 1.csv 去掉 3 个 meta 列后的列序）重算，逐元素平均绝对误差 **1.60e-16**、最大 **6.66e-16**；若按 Pearson 重算，平均误差 0.08768。题面写 Pearson，需在论文数据预处理一节注明"实测为 Spearman，据此按秩相关使用"。
- 1.csv 化合物列名与 `feature_dictionary.csv` 的 **field_name_R** 完全一致（231/231）；不能用 `field_name_python`（只有 47/231 能匹配）。
- 6.csv 44 行全部能映射回 1.csv 字段，50 列 = 4.csv 前 45 目标按位次 + 后 5 列（Alcohol / Aftertaste / Body / Carbonation / Overall appreciation）。

---

## 2. 问题 2：重建"全 50 维 × Top-3"交付表（方法）

### 2.1 共表达图（2.csv）

- 阈值 |ρ| ≥ 0.5：**231 节点、645 边、平均度 5.58**；Louvain 社区发现得 **60 个社区，最大社区 29 个节点**（0.6 阈值 → 266 边，作敏感性；0.7 → 109 边）。
- 图结构写入 `q2_network_communities.csv`（231 行：field、chemical_name、degree、community_id、community_size、top3_neighbors）与 `q2_network_edges_0_5.csv`（645 条边）。**目录中还残留早期 `q2_network_edges_0_6.csv`，论文与后续引用一律用 `_0_5` 版本。**

### 2.2 重要性排序：Ridge 系数 + 图邻域加权（题面允许的"图上的邻域加权"）

- 候选池 **219** = 231 − 12 个整体性质/派生量（沿用 `configs/q2_exploratory_v1.json` 的 `exclude_from_compound_top3`：pH、color、bitterness、GFS、unfermentables、kcalperc、esters_sum、aroma_hops_sum、acids_sum、sulfur_sum、protein、betaglucan）。
- 预处理：按列中位数填补 → asinh 压缩 → 标准化；y 中心化。
- 重要性：Ridge（α=1000）系数 b（219×50），
  **score_j = |b_j| + 0.5 · mean_{k∈N(j)} |b_k|**（N(j) 为 2.csv 图上邻居集合）。
  自项保证直接身份标记物不被图平滑掉，邻域项体现共表达模块的富集——即"图上的邻域加权"。按 score 每维取 Top-3。
- **为什么不用纯图正则（Laplacian 惩罚）**：实测 λ=1000 的图正则 Ridge 稳定性更高（频率均值 0.506、70/150 槽 ≥0.5），但化学上更差（diacetyl→β-phellandrene、co2→ethyl octanoate，且把 alcohol 的 Ethanol 挤掉）。本版选择了**化学可辩护性优先**的邻域加权；λ 的稳定性–可解释性权衡在 §5 量化给出，供论文"方法选择"一节引用。

### 2.3 稳定性量化（题面两种指标都做了）

**（a）bootstrap（按啤酒，B=200）**：每次重采样 250 瓶中 250 次有放回抽样，**在重采样数据上重新中位数填补/asinh/标准化**再拟合，统计每个排名位入选频率、每次重采样 Top-3 与全数据 Top-3 的 Jaccard。

- 150 个排名位入选频率：**均值 0.474、中位 0.403、最小 0.105、最大 1.00**；**57/150（38%）≥ 0.5**；
- 每维 Top-3 平均频率：**min 0.253、中位 0.442、max 0.887**；**20/50 维 ≥ 0.5**；
- bootstrap Jaccard（每维每次重采样 vs 全数据）：**均值 0.361、中位 0.324、最小 0.167**；**9/50 维均值 ≥ 0.5**。

**（b）多方法 Jaccard（4 种方法 = 6 对）**：图邻域加权（graphridge）、普通 Ridge（|b|）、ExtraTrees（300 树）、边际 Spearman（|ρ|）。

| 方法对 | Jaccard 均值 | 平均重合个数/3 | 维度数 Jaccard≥0.5 |
|---|---|---|---|
| graphridge vs Ridge | **0.600** | 2.16 | **45/50** |
| graphridge vs ExtraTrees | 0.380 | 1.56 | 28/50 |
| graphridge vs 边际 Spearman | 0.206 | 0.90 | 10/50 |
| Ridge vs ExtraTrees | 0.382 | 1.56 | 29/50 |
| Ridge vs 边际 Spearman | 0.198 | 0.86 | 10/50 |
| ExtraTrees vs 边际 Spearman | 0.222 | 0.96 | 12/50 |
| **全体（每维 6 对均值）** | **0.331** | **1.33** | **5/50 维均值≥0.5；3/50 维六对全≥0.5；48/50 维至少一对≥0.5** |

**（c）与团队现有 Ridge 主表的逐维 Jaccard**：均值 **0.354**、中位 **0.5**、**28/50 维 ≥0.5** → 两条独立实现（团队管线 vs 本重建）在 28 个维度至少重合 2/3，可作为"管线稳健性"的旁证。

**（d）解读（论文可直接写）**：方法族之间 Top-3 的一致率天然偏低（面板均值目标、219 个高度共线候选、仅有 250 样本），因此把"多方法全一致"当作唯一稳定性标准会过低。更合理的判据是：**同一建模范式内的扰动稳定性（graphridge↔Ridge 45/50 维 ≥0.5）+ 跨范式至少一对一致（48/50 维）**；报告时同时给出，避免选择性汇报。

### 2.4 化学合理性：证据分级与机理论述（题面只要求"论证"）

**分级规则**（写进论文方法即可复现）：
- **A 级**：入选化合物是该维度的身份标记物（如 X4vg→4-vinylguaiacol、metallic→Iron）；或机理同族且 6.csv 交叉相关 |r|≥0.2 且同号；或直接机理类别且 bootstrap 频率 ≥0.5；
- **B 级**：入选化合物与目标语义族存在明确机理关联（机理同族），或 6.csv |r|≥0.2 同号；
- **C 级**：机理关联弱/间接，仅作为探索性候选（必须标注，不宣称因果）。
- 置信度：high = A 且频率≥0.5；medium = A/B 且频率≥0.4；low = 其余。

**结果**：A=8、B=70、C=72（A+B=52%）；high=8、medium=35、low=107；6.csv 可对照的 Top-3 行 84/150；身份标记物进入 Top-3 的维度 5 个。

**A 级条目（可直接进论文表）**：
| 维度 | Top-3 化合物（频率, 6.csv r） | 机理 |
|---|---|---|
| acidity / lactic / acetic | Lactic acid (1.00, 0.32–0.45)；Acetic acid (0.63–0.83, 0.35–0.42) | 乳酸/醋酸是酸感与"乳酸味/醋酸味"的直接来源，6.csv 交叉相关同号支持 |
| X4vg | 4-vinylguaiacol (1.00, 0.473) | 4-乙烯基愈创木酚即"丁香/烟熏"酚类，身份标记物 |
| metallic | Iron (0.68, 0.244) | 铁离子是金属味的标准来源（阈值极低） |

**B 级典型（机理+6.csv 支持）**：
- 酒花萜类/倍半萜 31 条：`A_hops_citrus` → Citronellol (0.66) / Geraniol (0.53) / Linalool（柑橘–花香，酒花萜类教科书路径）；`A_hops_all`、`A_hops_noble`、`A_hops_tropical` 全部为萜类；
- 酯类 17 条：`A_esters_*` → 乙酸酯/乙酯类（香蕉/苹果/花香，酵母酯化路径）；`alcohol` → Ethyl hexanoate / Ethyl octanoate（脂肪酸乙酯随乙醇升高）；
- 呋喃/美拉德 6 条：`A_malt_bread` → Furfural (0.69) / Ethyl furoate / 5-methylfurfural（烘焙美拉德产物 → 面包/焦糖）；
- 醛类 6 条：`F_malt_all` → Acetaldehyde (0.83)（青苹果/生酒味）；吡嗪 2 条（2,6-dimethylpyrazine → 烘焙坚果）；甘油 2 条（body/口感）；硫化物 1 条（dms 维 → Sulfur dioxide，仅 B 级）。

**必须诚实标注的弱信号维度**（也是"因果说不清"的真实来源）：
- diacetyl 目标 vs X2.3.butanedione：中位数填补后 Spearman **−0.098**；dms vs DMS **+0.088**；t2n vs X.E..2.nonenal **−0.151**；clove vs eugenol **+0.023**。
- 强对应的维度：bitternes vs bitterness **+0.682**、alcohol vs ethanol **+0.817**、co2 vs CO2.PSI. **+0.461**、X4vg vs X4VG **+0.473**、metallic vs iron **+0.244**。
- 团队 Ridge 主表在弱信号维度上同样给出机制上说不通的候选（diacetyl→Eugenol、dms→Sulfur dioxide、co2→Ethyl octanoate 第 1/CO2 第 2、bitternes→Iron、clove→Eucalyptol，逐维 OOF R² 仅 0.01–0.64），说明**这是 4.csv 面板均值目标的信息上限，不是某个模型的实现缺陷**。论文写法：把这些维度作为"数据可解释性边界"讨论，并给出机理上应关注的化合物（双乙酰→奶油/黄油；DMS→熟玉米/煮蔬菜；t2n→纸板/氧化；丁香酚→丁香），同时明确"当前面板均值目标不足以统计识别，故标为探索性"。

---

## 3. 问题 5：1 天内最值得做的改进（排序 + 量化预期，按投入/收益）

| 优先级 | 改进 | 工作量 | 量化预期（本报告已实测） |
|---|---|---|---|
| **P0（必做，0.5h）** | 把 Q2 主表口径统一为 **50 维 × 3 = 150 条**；删除"零归因不补排名"表述，改为"仅 2 个无面板标签的 Acetaldehyde 的 6 个槽位保留空位（另有说明）" | 0.5 h（改表头/正文，无需重跑） | 覆盖 17/50 → **50/50**（+33 维、+99 条）；评委会看到的"未完成"信号直接消除 |
| **P1（推荐，半天）** | 主表换成"Ridge + 2.csv 图邻域加权"（本交付），并加三列：`bootstrap_freq`、`chemical_evidence(A/B/C)`、`confidence(high/medium/low)` | 0.5 天（脚本已就绪） | 与团队 Ridge 主表 **28/50 维 Jaccard≥0.5**（可主张互相印证）；方法内部一致性 **45/50 维 Jaccard≥0.5（均值 0.60）**；A+B 证据占 **52%** |
| **P2（半天）** | 论文新增"稳定性"小节：bootstrap 频率分布 + 6 对方法 Jaccard + 与团队主表重合度 | 0.5 天 | 能直接回答题面"稳定性可通过…量化"，且不夸大：均值 0.474、57/150 槽 ≥0.5、48/50 维至少一对方法一致 |
| **P3（可选，2h）** | 网络图升级：645 边/60 社区绘图 + 每维 Top-3 在社区中的位置（模块富集） | 2 h | 让"共表达图"从背景图变成证据图（题面明确要求"构建共表达图"） |
| **P4（不建议现在做）** | 追加强图正则（λ=1000）版本 | 1 h | 频率均值可到 0.506、70/150 槽 ≥0.5（+0.03），但化学首选变差（diacetyl→β-phellandrene 等），**稳定性提升不值这个代价**，只把 λ 敏感性作为附录 |

**不建议做的自我否定**：不要在正文写"关联不代表因果"式的免责大段；题面 Q2 明确只要求"稳定性与化学合理性论证"。写成"证据分级（A/B/C）+ 探索性标注 + 机理路径"即可，既严谨又不自伤。

---

## 4. 论文可用的表 / 图清单与写法

| 编号 | 文件（experiments/lead_review/q2/） | 行数 | 论文用途 | 写法建议 |
|---|---|---|---|---|
| 表 Q2-1 | `q2_top3_full50_graphridge.csv` | 150 | **主表**：每维 Top-3 + 频率 + 证据级 + 置信度 + 机理 | 正文只放 50 维精简版（target、3 个化合物、freq、证据级）；完整 25 列进附录/补充材料；表注写明"规则：score=|b|+0.5·邻居均值|b|，α=1000，219 候选" |
| 表 Q2-2 | `q2_top3_wide.csv` | 50 | 每维 Top-3 宽表（rank1–3 + 频率） | 作为 Q2-1 的紧凑替代，便于排版 |
| 表 Q2-3 | `q2_stability_bootstrap.csv` | 50 | 每维 3 槽入选频率 + 平均频率 + bootstrap Jaccard | 正文给"20/50 维平均频率≥0.5、9/50 维 bootstrap Jaccard≥0.5"；分布图配套 |
| 表 Q2-4 | `q2_method_jaccard.csv` | 50 | 6 对方法 Jaccard + 重合个数 | 正文给"核心两法 0.60（45/50 维≥0.5）；六对均值 0.331；48/50 维至少一对一致" |
| 图 Q2-1 | `q2_network_communities.csv` / `q2_network_edges_0_5.csv` | 231 / 645 | 共表达图（按社区着色、标注每维 Top-3） | 图注：|ρ|≥0.5，231 节点 645 边，Louvain 60 社区（最大 29） |
| 图 Q2-2 | `fig_q2_bootstrap_stability.png` | — | 150 个排名位频率分布 + 每维均值 | 图注给均值 0.474/中位 0.403/57 个槽≥0.5 |
| 图 Q2-3 | `fig_q2_method_jaccard.png` | — | 6 对方法 Jaccard 箱线/条形 | 图注给 0.60 vs 0.33 的分层解释（同范式 vs 跨范式） |
| 图 Q2-4 | `fig_q2_evidence_confidence.png` | — | A/B/C 与 high/medium/low 分布 | 用来说明"证据分级，而非全域因果" |
| 附 | `q2_summary.json` | — | 全部汇总数字的机器可读记录 | 便于复核 |

**正文段落骨架（建议）**：
1. 数据：2.csv 为 1.csv 的 Spearman（位置上界误差 6.66e-16），构建 |ρ|≥0.5 共表达图（645 边/60 社区）。
2. 方法：219 候选，Ridge α=1000 + 图邻域加权，逐维 Top-3；固定规则，可复现。
3. 稳定性：bootstrap（200×按啤酒）频率分布 + 4 方法 Jaccard；给两张图表与关键数字。
4. 化学合理性：A 级逐条给机理+6.csv 支持；B 级按机制类别计数（萜 31/酯 17/美拉德呋喃 6/醛 6/醇 5）；C 级在表内标"探索性"，正文一句"面板均值目标对 diacetyl/DMS/t2n 等维度分辨力有限（|ρ|≤0.15），相关维度仅作假设生成"。
5. 结论：覆盖 50/50、150 条；同范式稳定（45/50 维 Jaccard≥0.5）；52% 条目有直接机理/经验支持。

---

## 5. 问题 6：诚实评估

- **17/50 会被视为"没做完 Q2"吗？会。** 题面白纸黑字"给出每个感官维度的 Top-3"，表格缺 33 行、且缺失原因写成"零归因不补排名"，评委会读成"模型失败被写成交付规则"。这是本次最该改的一点，且**0.5 小时就能改完**（Ridge 主表本已 150 行）。
- **新方案覆盖**：**50/50 维、150 条无空位**；仅 2 个无面板标签的 Acetaldehyde 在 52 项口径下另有 6 个说明性空位（不影响 50 维主表）。
- **稳定性预期（实测值，不是估计）**：bootstrap 频率均值 **0.474**、57/150 槽 ≥0.5、20/50 维均值 ≥0.5；方法内部 Jaccard **0.60（45/50 维 ≥0.5）**、跨方法全体均值 0.331、48/50 维至少一对 ≥0.5；与团队主表重合中位 0.5（28/50 维 ≥0.5）。这套数字属于"**中等稳定、核心范式稳定**"，比旧表的"只报 51 条、无稳定性分布"强，但也别写成"高度稳定"。
- **化学合理性预期**：A+B 52%、A 级 8 条可逐条辩护；弱信号维度如实标探索性，比"强行给 17 维、其余空白"更容易被接受。
- **尚未覆盖的风险**：(1) 4.csv 面板目标是品评均值，单化合物解释力上限低（团队 OOF R² 0.01–0.64），任何方法都难在 diacetyl/DMS/t2n 上给出统计可信的身份标记物；(2) 5.csv 与 8.xlsx 未用于化合物级证据（5.csv 是感官–感官），论文不要把它们写成"化合物佐证"，只用 6.csv 与文献；(3) 本方案与团队主表有 22/50 维 Jaccard<0.5，换主表时要在论文里说明"以本表为 Q2 主表、团队旧表作为稳健性对照"，不要两张表并列给评委挑刺。

---

## 6. 复现命令

```powershell
cd "C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件"
$env:PYTHONIOENCODING = "utf-8"
# 主脚本（约 2-3 分钟）：建图、Ridge+邻域加权、200 次 bootstrap、4 方法 Jaccard、证据分级、出表出图
.venv\Scripts\python.exe experiments\lead_review\q2\02_rebuild_q2.py
# 数字核对
.venv\Scripts\python.exe experiments\lead_review\q2\07_numbers.py
# 化学合理性/团队表对照
.venv\Scripts\python.exe experiments\lead_review\q2\06_diag.py
```

**参数与随机性**：候选池 219、α=1000、图阈值 0.5、邻居权重 0.5、bootstrap B=200、随机种子固定（脚本内 RNG 初始化）。所有中间量（网社区、边、稳定性、Jaccard、summary）均落盘于 `experiments/lead_review/q2/`。

**产物清单**（`experiments/lead_review/q2/`）：
`q2_top3_full50_graphridge.csv`(150×25) · `q2_top3_wide.csv`(50) · `q2_stability_bootstrap.csv`(50) · `q2_method_jaccard.csv`(50) · `q2_network_communities.csv`(231) · `q2_network_edges_0_5.csv`(645) · `q2_summary.json` · `fig_q2_bootstrap_stability.png` · `fig_q2_method_jaccard.png` · `fig_q2_evidence_confidence.png` · 脚本 `00_inspect.py`/`01_probe_mapping.py`/`02_rebuild_q2.py`/`04_inspect.py`/`05_inspect_final.py`/`06_diag.py`/`07_numbers.py`。

> 注：目录中的 `SUPERSEDED_q2_network_edges_0_6.csv` 是早期敏感性版本残留（已改名标注）；论文与引用一律用 `_0_5`（0.6 阈值仅作敏感性附录，266 边）。
