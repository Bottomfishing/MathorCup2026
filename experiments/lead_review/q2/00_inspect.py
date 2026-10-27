# -*- coding: utf-8 -*-
"""Inspect attachments 1/2/4/5/6/7 and existing Q2 tables (read-only)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件")
ROOT = Path(r"C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件")
OUT = ROOT / "experiments" / "lead_review" / "q2"
OUT.mkdir(parents=True, exist_ok=True)


def hr(t):
    print("\n" + "=" * 30 + " " + t + " " + "=" * 30)


for name in ["1.csv", "2.csv", "4.csv", "5.csv", "6.csv", "7.csv"]:
    df = pd.read_csv(SRC / name, encoding="utf-8-sig")
    hr(name + f" shape={df.shape}")
    print("columns[:12]:", list(df.columns[:12]))
    print("columns[-8:]:", list(df.columns[-8:]))
    print("dtypes counts:", df.dtypes.value_counts().to_dict())
    print("NA total:", int(df.isna().sum().sum()), "| cols with NA:", int((df.isna().sum() > 0).sum()))
    nan_cols = df.columns[df.isna().sum() > 0].tolist()
    print("NA cols:", nan_cols[:20], "..." if len(nan_cols) > 20 else "")
    print("first row idx:", df.iloc[:2, :3].to_dict())
    if "2.csv" == name:
        num = df.select_dtypes("number")
        print("matrix numeric shape:", num.shape, "symmetric err:", float(np.nanmax(np.abs(num.values - num.values.T))))
    if "4.csv" == name:
        print("4.csv columns full:", list(df.columns))
        num = df.select_dtypes("number")
        print("4 numeric shape:", num.shape, "NA by col:", num.isna().sum().to_dict())
    if "7.csv" == name:
        print("7 columns full:")
        for c in df.columns:
            print(repr(c), int(df[c].isna().sum()))
    if "5.csv" == name:
        print("5 columns full:", list(df.columns))
        print("5 index head:", df.iloc[:5, :3].to_dict())
    if "6.csv" == name:
        print("6 first col:", df.iloc[:50, 0].tolist()[:50])
        print("6 columns full:", list(df.columns))

hr("existing Q2 table heads")
for rel in [
    "outputs/task6_final_20261026/paper/tables/q2_standardized__top3_all52_slots.csv",
    "outputs/task6_final_20261026/paper/tables/q2_raw__top3_raw_all50.csv",
    "outputs/task6_final_20261026/paper/tables/q2_standardized__top3_panel_all50.csv",
    "outputs/task6_final_20261026/paper/tables/validation__q2_rank_coverage.csv",
    "outputs/task6_final_20261026/paper/tables/q2_raw__rank_stability.csv",
]:
    df = pd.read_csv(ROOT / rel, encoding="utf-8-sig")
    print("\n--", rel, df.shape)
    print("columns:", list(df.columns))
    print(df.head(4).to_string()[:3000])

hr("scorecard scope 52 slots")
df = pd.read_csv(ROOT / "outputs/task6_final_20261026/paper/tables/scope__scorecard_scope.csv", encoding="utf-8-sig")
print(df.shape, list(df.columns))
print(df.to_string()[:5000])
