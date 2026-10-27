当前日常代码快照（20261028）。表与图见上一级 tables/figures，证据见 evidence/。
code/outputs/ 内嵌了 status/verify 所需的正式产物，可在本目录离线执行：
  ..\..\.venv\Scripts\python.exe -m unittest discover -s tests  # 若带 tests
  python scripts/current.py verify --output outputs/my_check  # 需 PYTHONPATH=src
训练与刷新仍需原始附件（--source）与已训练模型（train 生成）。完整历史实验见 Git 仓库。
