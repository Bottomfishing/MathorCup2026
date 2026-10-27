# -*- coding: utf-8 -*-
"""S0b: compound columns + 8.xlsx sheet survey."""
import numpy as np, pandas as pd
ATT = r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件"
df = pd.read_csv(ATT + r"\1.csv", low_memory=False)
comps = [c for c in df.columns if c not in ("beer", "beer_id", "tasting_category_fine")]
print("n compound cols:", len(comps))
print(json_ := "\n".join(comps) if False else "")
for c in comps:
    print(c, end=" | ")
print()
# target search
import re
for pat in ["dms", "diacetyl", "2.3", "butanedione", "4.vg", "vinyl", "guaiacol", "nonenal", "stale", "hop", "sulfur", "dmso", "acetaldehyde", "ethanol", "ph", "color", "bitter", "co2", "iso"]:
    hits = [c for c in comps if pat in c.lower()]
    print(f"[{pat}] ->", hits)
