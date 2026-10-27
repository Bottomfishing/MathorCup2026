# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SRC = r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件"
ROOT = r"C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件"
x = pd.read_csv(SRC + r"\1.csv", encoding="utf-8-sig")
t4 = pd.read_csv(SRC + r"\4.csv", encoding="utf-8-sig")
cols = [c for c in x.columns if c not in ("beer", "beer_id", "tasting_category_fine")]
for t, comp in [("diacetyl", "X2.3.butanedione"), ("dms", "DMS"), ("co2", "CO2.PSI."),
                ("alcohol", "ethanol..v.v."), ("X4vg", "X4VG"), ("metallic", "iron")]:
    yv = t4[t].to_numpy()
    xv = x[comp].to_numpy(dtype=float)
    r = spearmanr(xv, yv).statistic
    print(f"{t:10s} vs {comp:18s} rho={r:+.3f}  target_std={yv.std():.3f}  comp_nan={np.isnan(xv).sum()}")

print("\n== team table picks for these targets ==")
tt = pd.read_csv(ROOT + r"\outputs\task6_final_20261026\paper\tables\q2_standardized__top3_panel_all50.csv")
for t in ["diacetyl", "dms", "co2", "alcohol", "X4vg", "metallic", "t2n", "bitternes", "clove"]:
    g = tt[tt.target == t]
    print(t, "|", "; ".join(f"{r.chemical_name}(r2={r.ridge_oof_target_r2:.2f},f={r.bootstrap_frequency:.2f})" for r in g.itertuples()))
