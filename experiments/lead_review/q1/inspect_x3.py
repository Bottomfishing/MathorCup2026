# -*- coding: utf-8 -*-
"""Dump 3.xlsx structure: cards + summary, show formulas/values."""
import os, json
import numpy as np
import pandas as pd

SRC = r'C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件'
P = os.path.join(SRC, '3.xlsx')
xl = pd.ExcelFile(P)

# 1) summary sheet raw
s = pd.read_excel(P, sheet_name='summary', header=None)
print('== summary (header=None) shape', s.shape)
print(s.to_string(max_rows=30, max_cols=60))

# 2) beer_1 raw non-empty cells with coordinates
b = pd.read_excel(P, sheet_name='beer_1', header=None)
print('\n== beer_1 non-null cells (row,col,val):')
rows = []
for i in range(b.shape[0]):
    for j in range(b.shape[1]):
        v = b.iat[i, j]
        if pd.notna(v):
            rows.append((i + 1, j + 1, v))
for r in rows:
    print(r)
print('total nonnull', len(rows))
