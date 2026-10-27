# MathorCup A 当前交付 20261027

本目录是当前取数版本。先读 conclusions.csv、unresolved.csv，再按 file_index.csv 找 tables、figures 和 evidence。
完整代码在 code/；不包含原始附件、个人评分记录、下载参考工作簿或模型二进制。

Q1正式中位评分不变，个人5/6召回仍0。Q2三种口径分开。Q3仅119号四代理条件案例；Stale、完整五缺陷及真实工艺未解决。9/12共同通过与22/32附加扰动同时报告。

队友运行：进入 code，Python3.12创建.venv并安装 requirements-r3-lock.txt。
查看结果不需训练；`python scripts/current.py status`。
本地取回固定原始评分：`python scripts/fetch_raw_sensory.py --no-proxy`。
训练：`python scripts/current.py train --source "原附件目录"`。
预测：`python scripts/current.py predict --input "化学谱.csv" --output outputs/my_prediction`。
刷新固定谱画像：`python scripts/current.py render --output outputs/my_render`。
核验：`python scripts/current.py verify --models outputs/current_models --source "原附件目录" --output outputs/my_check`。
每个输出目录须新建。Q3辅助预测需显式 --task q3，不能当正式Q1。

历史实验复现需完整GitHub仓库，见 code/docs/历史依赖与复现.md。本目录没有重新包装旧任务6的结论。
