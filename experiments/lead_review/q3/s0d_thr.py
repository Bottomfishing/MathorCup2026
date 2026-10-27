# -*- coding: utf-8 -*-
"""S0d: full Sensory thresholds sheet -> csv for reuse."""
import pandas as pd
ATT = r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件"
OUT = r"C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件\experiments\lead_review\q3"
st = pd.read_excel(ATT + r"\8.xlsx", sheet_name="Sensory thresholds")
print("shape", st.shape, list(st.columns))
st.to_csv(OUT + r"\x8_sensory_thresholds.csv", index=False, encoding="utf-8-sig")
# rows with threshold info
has = st[st["threshold_mean"].notna() | st["threshold_median"].notna()]
print("rows with threshold:", len(has))
cols = ["field_name_python", "chemical_name", "off_flavors", "aging", "unit", "threshold_mean", "threshold_median", "origin_paper"]
pd.set_option("display.width", 250)
print(has[cols].to_string())
