# -*- coding: utf-8 -*-
"""Probe name mappings before the full rebuild."""
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件")
ROOT = Path(r"C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件")

fd = pd.read_csv(ROOT / "outputs/p0_audit_20261023_final/feature_dictionary.csv")
fd["key"] = fd["chemical_name"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
print("feature_dictionary rows:", len(fd), "unique keys:", fd["key"].nunique())
print("origins:", fd["origin_paper"].value_counts().to_dict())

m2 = pd.read_csv(SRC / "2.csv", encoding="utf-8-sig", index_col=0)
m2.index = m2.index.astype(str); m2.columns = m2.columns.astype(str)
print("\n2.csv shape:", m2.shape, "index==columns:", list(m2.index) == list(m2.columns))
rows2 = [(n, n.lower().replace(" ", "").replace("-", "").replace(",", "").replace("(", "").replace(")", "").replace(".", "")) for n in m2.index]
# match by compact key
m2keys = {r[1]: r[0] for r in rows2}
fdkeys = dict(zip(fd["key"], fd["field_name_python"]))
unmatched2 = [n for n, k in rows2 if k not in fdkeys]
print("2.csv unmatched by compact key:", len(unmatched2), unmatched2[:20])
print("2.csv matched:", len(rows2) - len(unmatched2))

# 6.csv
m6 = pd.read_csv(SRC / "6.csv", encoding="utf-8-sig")
print("\n6.csv shape:", m6.shape, "first col:", m6.columns[0])
names6 = m6.iloc[:, 0].astype(str).tolist()
un6 = []
for n in names6:
    k = n.lower().replace(" ", "").replace("-", "").replace(",", "").replace("(", "").replace(")", "").replace(".", "").replace("/", "")
    if k not in fdkeys:
        un6.append(n)
print("6.csv unmatched:", len(un6), un6)
print("6.csv matches:")
for n in names6:
    k = n.lower().replace(" ", "").replace("-", "").replace(",", "").replace("(", "").replace(")", "").replace(".", "").replace("/", "")
    print(f"  {n!r:40s} -> {fdkeys.get(k)}")

# 5/6 column order vs 4 targets
tgt = pd.read_csv(SRC / "4.csv", encoding="utf-8-sig")
targets = [c for c in tgt.columns if c not in ("beer", "beer_id", "tasting_category_fine")]
print("\n4 targets:", len(targets))
cols5 = [c for c in m6.columns[1:]]
print("6 col count:", len(cols5))
for a, b in zip(targets, cols5):
    print(f"  {a:20s} | {b!r}")
