# MathorCup 2026 A：啤酒风味建模

本仓库包含数据审计、感官预测、化合物因素分析、风格条件生成及受约束迁移。当前结果均已提交main。论文由其他队员负责，本任务只交付代码、数据分析、实验结果和使用说明。

## 先读这些文件

- [统一结果索引](docs/结果索引.md)：当前结果及历史实验的区别。
- [复现与交付说明](docs/复现与交付说明.md)：从新克隆、安装依赖到重建结果的顺序。
- [分阶段进度](docs/分阶段进度.md)：完成情况和下一步。
- [Q1评分说明](docs/Q1评分交付说明.md)、[Q2因素说明](docs/Q2因素交付说明.md)、[Q3生成迁移说明](docs/Q3生成迁移交付说明.md)：字段、模型假设和结果边界。

## 从新克隆开始

需要Python 3.12及原始附件目录（1.csv至8.xlsx、数据说明.txt）。源数据保留在仓库外。PowerShell：

```powershell
git clone https://github.com/Bottomfishing/MathorCup2026.git
cd MathorCup2026
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe scripts/reproduce_delivery.py --source '原始附件目录' --output outputs/my_reproduction
```

输出目录必须不存在。流程会获取已固定校验和的公开参考工作簿，重建当前模型和交付表，检查科学数值是否与已提交结果一致；日志和最终状态见输出目录中的`reproduction.json`。源数据、模型二进制和下载的参考工作簿不随Git上传，运行时自行提供或重建。

仅需单阶段运行、调用新样本预测或重画图时，使用对应交付说明里的命令。旧脚本可能记录原作者机器路径；上述入口和当前交付脚本使用显式路径，不依赖旧joblib。

## 当前结果

| 问题 | 主要入口 | 当前结论边界 |
|---|---|---|
| Q1 | outputs/final_20261024/q1、outputs/q1_delivery_20261024 | 连续预测及相对0～6等级；原始人工评分未恢复 |
| Q2 | outputs/q2_delivery_20261024 | 50目标150条关联候选及稳定性/化学证据；部分直接证据不足 |
| Q3 | outputs/final_20261024/q3、outputs/q3_delivery_20261024 | 300条谱、画像及9款模型迁移候选；未认证真实风格和物理可行性 |

57项单元测试及从GitHub新克隆后的完整交付复现已通过：独立Python 3.12环境、11个步骤、8组关键表在1e-10容差内一致。记录见[复现记录](outputs/reproduction_20261024/reproduction.json)。这些结果不代表原始评分、因果作用或酿造约束已解决。

交付后新增Q1研究：[直接分级与尺度判定](docs/Q1分级改进与尺度判定.md)。相对等级MAE从1.617降至1.472，但高等级识别变差，作为显式加载的可选方案，保留旧默认模型。本轮61项测试通过；新候选有单独实验与验证，不冒用上述基点复现记录。

## 参考与历史

公开原研究数据：[Zenodo 10653704](https://zenodo.org/records/10653704)。名称和阈值来源保留校验和与工作表行号；不加载公开外部预训练模型。AI方法及代码贡献见阶段交付说明与[AI使用记录](docs/AI使用记录.md)。

旧基线与探索结果全部保留，入口见统一结果索引的历史部分。文件夹`final_20261024`是已有运行名，不表示三问科学问题全部解决。
