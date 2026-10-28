# MathorCup A 当前交付 20261028

本目录是 20261028 版本：先读 conclusions.csv、unresolved.csv，再按 file_index.csv 查表。
与 20261027 的差别（历史组仍以 history_ 前缀保留）：
Q1 决策层改为逐样本固定阈值（批次无关，整批/逐条/分块同值），历史批次 top-k 表降为对照；
另含面板均值折外标定参考列与面板均值直接回归（未过预注册门槛，不采纳）证据；
Q2 换为 50/50 维全覆盖的图 Top-3（150 条）与 bootstrap/Jaccard 稳定性，并含图邻域消融（只提升稳定性、不提升预测）；
Q3 增加 copula 生成器的预注册门槛判定：v3 混合谱 14/14 通过并替换官方插值谱（v1/v2 作诊断保留）；
迁移 11 款×3 风格=33 条题面口径案例，含完整谱、前后感官画像与证据分级；
另有风格分类器折外证据（ET 0.544）、Tripel 缺陷安全冲突诊断与生成器消融。

数据披露：3.xlsx 为空白评分模板；0-6 标签来自外部 Zenodo 23624915 原始记录；2.csv 为 Spearman 矩阵。
本包不含原始附件、个人评分记录与模型二进制。

命令（在 code/ 内）：
`python scripts/current.py status`
`python scripts/current.py train --source "原附件目录"`
`python scripts/current.py predict --input 化学谱.csv --output outputs/my_prediction`
`python scripts/current.py render --output outputs/my_render`
`python scripts/current.py verify --output outputs/my_check`
`python scripts/current.py package --output outputs/current_20261028`
