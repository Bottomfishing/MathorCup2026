# -*- coding: utf-8 -*-
"""S0c: 8.xlsx sheets + threshold-ish content."""
import numpy as np, pandas as pd
ATT = r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件"
xl = pd.ExcelFile(ATT + r"\8.xlsx")
print("sheets:", xl.sheet_names)
for s in xl.sheet_names:
    d = xl.parse(s, header=None, nrows=12)
    print("=" * 70)
    print("SHEET:", s, "shape-probe", d.shape)
    print(d.to_string(max_colwidth=42)[:3000])
