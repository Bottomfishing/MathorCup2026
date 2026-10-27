# -*- coding: utf-8 -*-
"""S1: Pils/Lager defect status under 1.csv + 8.xlsx thresholds."""
import numpy as np, pandas as pd, json
ATT = r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件"
OUT = r"C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件\experiments\lead_review\q3"
pd.set_option("display.width", 260)

df = pd.read_csv(ATT + r"\1.csv", low_memory=False)
st = pd.read_excel(ATT + r"\8.xlsx", sheet_name="Sensory thresholds")
cal = pd.read_excel(ATT + r"\8.xlsx", sheet_name="Calibrated chemical dataset")

DEF = {"DMS": "DMS", "Diacetyl": "X2.3.butanedione", "4VG": "X4VG", "t2n": "X.E..2.nonenal"}
THR = {}
for k, col in [("DMS", "DMS"), ("Diacetyl", "2,3-butanedione"), ("4VG", "X4VG")]:
    row = st[st["field_name_python"].astype(str).str.strip() == col]
    THR[k] = (float(row["threshold_mean"].iloc[0]), float(row["threshold_median"].iloc[0]), row["unit"].iloc[0])

# --- verify log10 transform against calibrated sheet ---
print("== transform check: corr(10**x1c, cal) vs corr(x1c, cal) ==")
m = df.merge(cal[["beer_id"] + [c for c in cal.columns if c not in ("beer", "beer_id", "tasting_category_fine")]],
             on="beer_id", how="inner", suffixes=("", "_cal"))
print("merged rows:", len(m))
rows = []
for c in [c for c in cal.columns if c not in ("beer", "beer_id", "tasting_category_fine")]:
    if c not in df.columns:
        continue
    sub = m[[c, c + "_cal"]].dropna()
    if len(sub) < 40:
        continue
    r_log = np.corrcoef(10 ** sub[c], sub[c + "_cal"])[0, 1]
    r_raw = np.corrcoef(sub[c], sub[c + "_cal"])[0, 1]
    med = float(np.median(sub[c]))
    rows.append(dict(col=c, n=len(sub), r_log10=r_log, r_raw=r_raw, med=med,
                     verdict="log10" if abs(r_log) > abs(r_raw) else "raw"))
tc = pd.DataFrame(rows)
print(tc["verdict"].value_counts())
print(tc[tc["verdict"] == "raw"].to_string())
print("---- defect cols transform verdict ----")
print(tc[tc["col"].isin(DEF.values())].to_string())
tc.to_csv(OUT + r"\s1_transform_check.csv", index=False, encoding="utf-8-sig")

# --- Pils/Lager subset defect concentrations ---
pl = df[df["tasting_category_fine"] == "Pils/Lager"].copy()
print("\n== Pils/Lager:", len(pl), "beers, ids:", pl["beer_id"].tolist())
print(pl[["beer", "beer_id"]].to_string())
for k, c in DEF.items():
    v = pl[c]
    print(f"\n[{k}] col={c} n_missing={v.isna().sum()}")
    print("  coord stats:", v.describe().round(4).to_dict())
    if k in THR:
        mean, med, unit = THR[k]
        conc = 10 ** v
        print(f"  threshold mean={mean} median={med} {unit}; log10 mean/thr={np.log10(mean):.4f} median/thr={np.log10(med):.4f}")
        print("  conc:", np.round(conc.dropna().values, 5).tolist())
        print(f"  exceed(mean-thr): {(conc > mean).sum()}/{len(pl)}  exceed(median-thr): {(conc > med).sum()}/{len(pl)}")
pl.to_csv(OUT + r"\s1_pils_lager.csv", index=False, encoding="utf-8-sig")

# --- all-style defect exceedance for context ---
print("\n== exceedance by style (median-threshold) ==")
recs = []
for sty, g in df.groupby("tasting_category_fine"):
    r = dict(style=sty, n=len(g))
    for k, c in DEF.items():
        if k in THR:
            conc = 10 ** g[c]
            r[f"{k}_exc"] = int((conc > THR[k][1]).sum())
            r[f"{k}_p50"] = float(np.nanmedian(conc))
        else:
            r[f"{k}_p50"] = float(np.nanmedian(10 ** g[c])) if False else float(np.nanmedian(10 ** g[c]))
    recs.append(r)
ex = pd.DataFrame(recs).sort_values("n", ascending=False)
print(ex.to_string())
ex.to_csv(OUT + r"\s1_style_exceedance.csv", index=False, encoding="utf-8-sig")
