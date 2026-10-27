# -*- coding: utf-8 -*-
"""S2c: Q3 cross-style migration with defensible criteria and honest style geometry.

Coordinate space : attachment-1 (231 columns; 3 columns are metadata)
Units            : per-column log10/raw decision inferred from 8.xlsx
                   'Calibrated chemical dataset' (35 log10, 35 raw)
Thresholds       : 8.xlsx 'Sensory thresholds', field_name_python
                   DMS mean 17.25 / median 19.0 ug/L
                   diacetyl (2,3-butanedione) mean 0.29925 / median 0.09 mg/L
                   4VG (X4VG) mean 0.206333 / median 0.20 mg/L
                   trans-2-nonenal: NO threshold row -> only "do not worsen"
                   stale hops: no threshold, no compound coordinate -> 6 hop-oxidation
                   proxy coordinates, constrained to "do not worsen"

Style model : nearest-centroid classifier in a whitened 8-D global PCA space, plus a
              multinomial-logistic baseline; both with leave-one-out accuracy, so the
              reader can see how identifiable tasting_category_fine really is.

Migration   : z(lam) = (1-lam) z_src + lam * mean_{target}       (convex blend, 231 coords)
              then a hard defect projection
                  threshold defect  : coordinate <= min(source, 0.9*T)
                  no-threshold      : coordinate <= source
                  hop-oxidation     : coordinate <= source
              lam* = largest lam on the 0.05 grid for which the target centroid is the
              nearest centroid; fallback lam minimising the distance to the target centroid.

Criteria    : A  literal problem statement, set form: the set of over-threshold defects may
                not expand and severity may not increase
              B  per-defect e: e_after <= e_before, e = max(c/T-1, 0)
              C  aggregated severity: sum_d e_d(after) <= sum_d e_d(before)
              (D = the team's extra "five calibrated sensory probabilities must not
               increase" is audited from its own artefact, not re-implemented here)
Both the median and the mean threshold scenario are evaluated.

Outputs -> experiments/lead_review/q3/
"""
import json
import numpy as np
import pandas as pd

ATT = r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件"
OUT = r"C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件\experiments\lead_review\q3"
np.set_printoptions(suppress=True)

df = pd.read_csv(ATT + r"\1.csv", low_memory=False)
st = pd.read_excel(ATT + r"\8.xlsx", sheet_name="Sensory thresholds")
cal = pd.read_excel(ATT + r"\8.xlsx", sheet_name="Calibrated chemical dataset")
META = ["beer", "beer_id", "tasting_category_fine"]
COLS = [c for c in df.columns if c not in META]
STYLE = df["tasting_category_fine"].to_numpy()
ID = df["beer"].to_numpy()


def th(field, which):
    r = st[st["field_name_python"].astype(str).str.strip() == field]
    return None if len(r) == 0 or pd.isna(r[which].iloc[0]) else float(r[which].iloc[0])


T = {k: dict(mean=th(f, "threshold_mean"), median=th(f, "threshold_median"))
     for k, f in [("DMS", "DMS"), ("Diacetyl", "2,3-butanedione"), ("4VG", "X4VG")]}

m = df.merge(cal, on="beer_id", how="inner", suffixes=("", "_cal"))
TMAP = {}
for c in [c for c in cal.columns if c not in META and c in df.columns]:
    sub = m[[c, c + "_cal"]].dropna()
    if len(sub) < 40:
        continue
    r_log = np.corrcoef(10 ** sub[c], sub[c + "_cal"])[0, 1]
    r_raw = np.corrcoef(sub[c], sub[c + "_cal"])[0, 1]
    TMAP[c] = "raw" if not np.isfinite(r_log) else ("log10" if abs(r_log) > abs(r_raw) + 0.02 else "raw")
TMAP.update({"DMS": "log10", "X4VG": "log10", "X2.3.butanedione": "raw", "X.E..2.nonenal": "raw"})

DEFECT = {"DMS": "DMS", "Diacetyl": "X2.3.butanedione", "4VG": "X4VG", "t2n": "X.E..2.nonenal"}
STALE = [c for c in ["humulene.epoxide.II", "caryophyllene.oxide",
                     "dihydroisocaryophyllene.epoxide", "X4.8.alpha.epoxy.caryophyllane",
                     "X14.hydroxy.alpha.humulene", "X14.hydroxy.9.epi..E..caryophyllene"] if c in df.columns]

KEEP = [c for c in COLS if df[c].isna().mean() <= 0.25]
X = df[KEEP].copy()
for c in KEEP:
    if X[c].isna().any():
        sm = df.groupby("tasting_category_fine")[c].median()
        X[c] = X[c].fillna(df["tasting_category_fine"].map(sm)).fillna(X[c].median())
X = X.to_numpy(float)
med = np.median(X, 0)
iqr = np.percentile(X, 75, 0) - np.percentile(X, 25, 0)
sd = X.std(0)
scale = np.where(iqr > 1e-9, iqr / 1.349, np.where(sd > 0, sd, 1.0))
Z = (X - med) / scale
ci = {c: i for i, c in enumerate(KEEP)}


def conc(c, v):
    return 10.0 ** np.asarray(v, float) if TMAP.get(c, "raw") == "log10" else np.asarray(v, float)


def zc(c, coord):
    """coordinate -> standardised coordinate (input is already a coordinate)."""
    return (coord - med[ci[c]]) / scale[ci[c]]


def zc_conc(c, concentration):
    """physical concentration -> standardised coordinate (applies the log10 transform)."""
    v = np.log10(concentration) if (TMAP.get(c, "raw") == "log10" and concentration > 0) else concentration
    return (v - med[ci[c]]) / scale[ci[c]]


# ---------------- style geometry: whitened global PCA (8 comps) ----------------
Zc = Z - Z.mean(0)
_, _, Vt = np.linalg.svd(Zc, full_matrices=False)
NPCS = 8
P = Vt[:NPCS].T
PC = Zc @ P
pcw = PC.std(0)
W = Zc @ (P / pcw)
STYLES = list(np.unique(STYLE))
mu = {s: W[STYLE == s].mean(0) for s in STYLES}


def wdist(z):
    w = (z - Z.mean(0)) @ (P / pcw)
    return np.array([np.linalg.norm(w - mu[s]) for s in STYLES])


hit = 0
for i in range(len(W)):
    best, bl = None, 1e18
    for s in STYLES:
        idx = np.where(STYLE == s)[0]
        idx = idx[idx != i]
        if len(idx) < 2:
            continue
        d = np.linalg.norm(W[i] - W[idx].mean(0))
        if d < bl:
            bl, best = d, s
    hit += int(best == STYLE[i])
nc_acc = hit / len(W)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=0.5, max_iter=800, solver="lbfgs")
hit2 = 0
for i in range(len(PC)):
    tr = np.arange(len(PC)) != i
    lr.fit(PC[tr], STYLE[tr])
    hit2 += int(lr.predict(PC[i:i + 1])[0] == STYLE[i])
lr_acc = hit2 / len(PC)
maj = np.bincount(pd.factorize(STYLE)[0]).max() / len(STYLE)
print(f"style identifiability: nearest-centroid LOO acc={nc_acc:.3f} | "
      f"multinomial-logit 8PC LOO acc={lr_acc:.3f} | majority baseline={maj:.3f} | 22 classes")


def project(z, si, tgt):
    z = z.copy()
    notes = {}
    for k, c in DEFECT.items():
        if c not in ci:
            continue
        j = ci[c]
        src = df.iloc[si][c]
        Tv = T[k]["median"] if k in T else None
        if pd.isna(src):
            tv = float(np.nanmedian(df.loc[df["tasting_category_fine"] == tgt, c]))
            capv = min(tv, 0.9 * Tv) if Tv is not None else tv
            z[j] = min(z[j], zc_conc(c, capv))
            notes[k] = f"src_NA -> cap=min(target_median,0.9T)={capv:.5g} (conditional)"
        else:
            sv = float(src)
            z[j] = min(z[j], zc(c, sv))                   # never above the source
            if Tv is not None:
                z[j] = min(z[j], zc_conc(c, 0.9 * Tv))    # and below 0.9*threshold
            notes[k] = "cap=min(src,0.9T)"
    for c in STALE:
        sv = df.iloc[si][c]
        if pd.notna(sv):
            z[ci[c]] = min(z[ci[c]], zc(c, float(sv)))
    return z, notes


def dtab(si, z):
    o = {}
    for k, c in DEFECT.items():
        j = ci[c]
        src = df.iloc[si][c]
        cb = np.nan if pd.isna(src) else float(conc(c, float(src)))
        ca = float(conc(c, z[j] * scale[j] + med[j]))
        o[k] = dict(cb=cb, ca=ca, miss=bool(pd.isna(src)))
    return o


def check(si, z, w):
    """criteria A/B/C for threshold scenario w in {mean, median}."""
    d = dtab(si, z)
    A = Bb = Cc = True
    sb = sa = 0.0
    cond = False
    for k in DEFECT:
        Tv = T[k][w] if k in T else None
        if Tv is None:
            continue
        if d[k]["miss"]:
            cond = True
            ea = max(d[k]["ca"] / Tv - 1, 0)
            A &= ea == 0.0
            Bb &= ea == 0.0
            sa += ea
            continue
        eb = max(d[k]["cb"] / Tv - 1, 0)
        ea = max(d[k]["ca"] / Tv - 1, 0)
        A &= (ea == 0.0) if eb == 0.0 else (ea <= eb + 1e-12)
        Bb &= ea <= eb + 1e-12
        sb += eb
        sa += ea
    return bool(A), bool(Bb), bool(sa <= sb + 1e-12), sb, sa, bool(cond)


SOURCES = df.loc[df["tasting_category_fine"] == "Pils/Lager", "beer"].to_numpy()
TARGETS = ["Tripel", "Kriek", "Lambic"]
LAMS = np.round(np.arange(0.05, 1.0001, 0.05), 3)
rows, cnt = [], []
for tgt in TARGETS:
    tmu = Z[STYLE == tgt].mean(0)
    ti = STYLES.index(tgt)
    for b in SOURCES:
        si = int(np.where(ID == b)[0][0])
        cands = []
        for lam in LAMS:
            z1, notes = project((1 - lam) * Z[si] + lam * tmu, si, tgt)
            dd = wdist(z1)
            cands.append(dict(lam=lam, z=z1, d=dd, notes=notes, am=int(np.argmin(dd))))
        ok = [c for c in cands if c["am"] == ti]
        pick = max(ok, key=lambda c: c["lam"]) if ok else min(cands, key=lambda c: c["d"][ti])
        d0 = wdist(Z[si])
        r = dict(target=tgt, source_beer=int(b), lambda_star=pick["lam"],
                 d_target_before=float(d0[ti]), d_target_after=float(pick["d"][ti]),
                 dist_reduction_pct=float(100 * (1 - pick["d"][ti] / d0[ti])),
                 style_before=STYLES[int(np.argmin(d0))], style_after=STYLES[pick["am"]],
                 style_is_target=bool(pick["am"] == ti),
                 notes=json.dumps(pick["notes"], ensure_ascii=False))
        for w in ("mean", "median"):
            A, Bb, Cc, sb, sa, cond = check(si, pick["z"], w)
            r[f"A_{w}"], r[f"B_{w}"], r[f"C_{w}"] = A, Bb, Cc
            r[f"Esum_{w}_b"], r[f"Esum_{w}_a"], r[f"cond_{w}"] = sb, sa, cond
        dt = dtab(si, pick["z"])
        for k in DEFECT:
            r[k + "_b"], r[k + "_a"] = dt[k]["cb"], dt[k]["ca"]
        rows.append(r)
        c = dict(target=tgt, source_beer=int(b), n_lambda=len(LAMS),
                 n_lambda_target_nearest=len(ok))
        for w in ("mean", "median"):
            chk = [check(si, x["z"], w) for x in cands]
            c[f"n_pass_A_{w}"] = int(sum(y[0] for y in chk))
            c[f"n_pass_B_{w}"] = int(sum(y[1] for y in chk))
            c[f"n_pass_C_{w}"] = int(sum(y[2] for y in chk))
        cnt.append(c)

res = pd.DataFrame(rows)
res.to_csv(OUT + r"\s2c_migration_results.csv", index=False, encoding="utf-8-sig")
cc = pd.DataFrame(cnt)
cc.to_csv(OUT + r"\s2c_candidate_counts.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 400)
print("\n== per-target coverage (11 Pils/Lager sources each) ==")
for t in TARGETS:
    g = res[res.target == t]
    print(f"{t:7s} A(median) {int(g.A_median.sum())}/11  B {int(g.B_median.sum())}/11  C {int(g.C_median.sum())}/11"
          f" | target-is-nearest {int(g.style_is_target.sum())}/11 | dist reduced {int((g.dist_reduction_pct > 0).sum())}/11"
          f" | src-missing-conditional {int(g.cond_median.sum())}/11 | mean-lambda {g.lambda_star.mean():.2f}")
print("\n== candidates surviving the literal criteria (20 lambda values x 11 sources) ==")
for t in TARGETS:
    g = cc[cc.target == t]
    print(f"{t:7s} A_median {g.n_pass_A_median.sum()}/{g.n_lambda.sum()}  B_median {g.n_pass_B_median.sum()}/{g.n_lambda.sum()}"
          f"  C_median {g.n_pass_C_median.sum()}/{g.n_lambda.sum()}  A_mean {g.n_pass_A_mean.sum()}/{g.n_lambda.sum()}"
          f"  target-nearest {g.n_lambda_target_nearest.sum()}/{g.n_lambda.sum()}")
print("\n== Tripel per-source (median-threshold scenario) ==")
print(res[res.target == "Tripel"][["source_beer", "lambda_star", "style_before", "style_after",
                                   "d_target_before", "d_target_after", "dist_reduction_pct",
                                   "DMS_b", "DMS_a", "Diacetyl_b", "Diacetyl_a", "4VG_b", "4VG_a",
                                   "t2n_b", "t2n_a", "A_median", "B_median", "C_median",
                                   "cond_median"]].to_string())
print("\n== Kriek / Lambic ==")
print(res[res.target != "Tripel"][["target", "source_beer", "lambda_star", "style_after",
                                   "d_target_before", "d_target_after", "dist_reduction_pct",
                                   "A_median", "B_median", "cond_median"]].to_string())

sr = []
for b in SOURCES:
    si = int(np.where(ID == b)[0][0])
    row = dict(beer=int(b))
    for k, c in DEFECT.items():
        v = df.iloc[si][c]
        row[k + "_coord"] = v
        row[k + "_conc"] = np.nan if pd.isna(v) else float(conc(c, float(v)))
        for w in ("mean", "median"):
            Tv = T[k][w] if k in T else None
            row[f"{k}_T_{w}"] = Tv
            row[f"{k}_exc_{w}"] = np.nan if (pd.isna(v) or Tv is None) else int(float(conc(c, float(v))) > Tv)
            row[f"{k}_e_{w}"] = np.nan if (pd.isna(v) or Tv is None) else max(float(conc(c, float(v))) / Tv - 1, 0)
    for c in STALE:
        row["stale_" + c] = df.iloc[si][c]
    sr.append(row)
sd = pd.DataFrame(sr)
sd.to_csv(OUT + r"\s2_source_defects.csv", index=False, encoding="utf-8-sig")
print("\n== Pils/Lager source defect table ==")
print(sd[["beer", "DMS_conc", "DMS_exc_mean", "Diacetyl_conc", "Diacetyl_exc_median",
          "4VG_conc", "4VG_exc_median", "t2n_conc"]].to_string())
print("\nDMS exc>17.25:", int(sd.DMS_exc_mean.sum()), "/11  exc>19:", int(sd.DMS_exc_median.sum()), "/11")
print("Diacetyl exc>0.29925:", int(sd.Diacetyl_exc_mean.sum()), "/11  exc>0.09:",
      int(sd.Diacetyl_exc_median.sum()), "/", int(sd.Diacetyl_exc_median.notna().sum()), "non-missing")
print("4VG exc>0.2063:", int(sd["4VG_exc_mean"].sum()), "/11  exc>0.20:", int(sd["4VG_exc_median"].sum()), "/11")
