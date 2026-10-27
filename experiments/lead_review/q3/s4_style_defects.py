# -*- coding: utf-8 -*-
"""S4: per-style defect baseline with CORRECT units (replaces the buggy early table).

DMS and X4VG are log10 coordinates in attachment 1; X2.3.butanedione and
X.E..2.nonenal are raw mg/L. Thresholds from 8.xlsx 'Sensory thresholds'
(median applied per style, only DMS / diacetyl / 4VG have thresholds).
"""
import numpy as np
import pandas as pd

ATT = r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件"
OUT = r"C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件\experiments\lead_review\q3"

df = pd.read_csv(ATT + r"\1.csv", low_memory=False)
st = pd.read_excel(ATT + r"\8.xlsx", sheet_name="Sensory thresholds")


def th(f, w):
    r = st[st["field_name_python"].astype(str).str.strip() == f]
    return None if len(r) == 0 or pd.isna(r[w].iloc[0]) else float(r[w].iloc[0])


TD = [("DMS", "DMS", "log10", th("DMS", "threshold_median"), th("DMS", "threshold_mean"), "ug/L"),
      ("Diacetyl", "X2.3.butanedione", "raw", th("2,3-butanedione", "threshold_median"),
       th("2,3-butanedione", "threshold_mean"), "mg/L"),
      ("4VG", "X4VG", "log10", th("X4VG", "threshold_median"), th("X4VG", "threshold_mean"), "mg/L"),
      ("t2n", "X.E..2.nonenal", "raw", None, None, "mg/L")]

rows = []
for s in sorted(df["tasting_category_fine"].unique()):
    sub = df[df["tasting_category_fine"] == s]
    r = dict(style=s, n=len(sub))
    for name, col, tr, tm, tmean, unit in TD:
        v = sub[col].astype(float)
        c = 10 ** v if tr == "log10" else v
        r[name + "_unit"] = unit
        r[name + "_p50"] = float(np.nanmedian(c))
        r[name + "_p90"] = float(np.nanpercentile(c.dropna(), 90)) if c.notna().any() else np.nan
        r[name + "_n_missing"] = int(c.isna().sum())
        if tm is not None:
            r[name + "_T_median"] = tm
            r[name + "_T_mean"] = tmean
            r[name + "_exc_median"] = int((c > tm).sum())
            r[name + "_frac_exc_median"] = float((c > tm).mean())
            r[name + "_exc_mean"] = int((c > tmean).sum())
            r[name + "_frac_exc_mean"] = float((c > tmean).mean())
    rows.append(r)

res = pd.DataFrame(rows)
res.to_csv(OUT + r"\s2_style_defect_exceedance.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
order = ["Pils/Lager", "Tripel", "Kriek", "Lambic"]
print(res[res.style.isin(order)][["style", "n", "DMS_p50", "DMS_exc_median", "Diacetyl_p50",
                                  "Diacetyl_exc_median", "Diacetyl_exc_mean", "4VG_p50",
                                  "4VG_exc_median", "t2n_p50"]].to_string(index=False))
print("\nall styles, DMS (ug/L) and diacetyl (mg/L) median + n over median threshold:")
print(res[["style", "n", "DMS_p50", "DMS_exc_median", "Diacetyl_p50", "Diacetyl_exc_median",
           "4VG_p50", "4VG_exc_median", "t2n_p50"]].sort_values("DMS_p50", ascending=False).to_string(index=False))
