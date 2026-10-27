# MathorCup A 当前交付 20261028

本目录是 20261028 版本：先读 conclusions.csv、unresolved.csv，再按 file_index.csv 查表。
与 20261027 的差别（历史组仍以 history_ 前缀保留）：
Q1 增加决策层 top-k（同一 ET 模型，输出 5/6，代价与 lift 如实记录）+ 面板均值折外标定参考列；
Q2 换为 50/50 维全覆盖的图 Top-3（150 条）与 bootstrap/Jaccard 稳定性；
Q3 增加 copula 生成器的预注册门槛判定：v3 混合谱 14/14 通过并替换官方插值谱（v1/v2 作诊断保留）；迁移 11 款×3 风格=33 条题面口径案例。

数据披露：3.xlsx 为空白评分模板；0-6 标签来自外部 Zenodo 23624915 原始记录；2.csv 为 Spearman 矩阵。
本包不含原始附件、个人评分记录与模型二进制。

命令（在 code/ 内）：
`python scripts/current.py status`
`python scripts/current.py train --source "原附件目录"`
`python scripts/current.py predict --input 化学谱.csv --output outputs/my_prediction`
`python scripts/current.py render --output outputs/my_render`
`python scripts/current.py verify --output outputs/my_check`
`python scripts/current.py package --output outputs/current_20261028`
