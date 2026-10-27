# -*- coding: utf-8 -*-
"""S0: survey 1.csv / 8.xlsx for Q3 restart. Read-only on attachments."""
import json, sys
import numpy as np
import pandas as pd

ATT = r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件"
OUT = r"C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件\experiments\lead_review\q3"

df = pd.read_csv(ATT + r"\1.csv", low_memory=False)
print("1.csv shape:", df.shape)
print("first cols:", list(df.columns[:14]))
# style column
cand = [c for c in df.columns if "tasting" in c.lower() or "category" in c.lower() or "style" in c.lower()]
print("style-ish cols:", cand)
for c in cand:
    print("---", c, df[c].nunique(dropna=False))
    print(df[c].value_counts(dropna=False).head(40).to_string())
