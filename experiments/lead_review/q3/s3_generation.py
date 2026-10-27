# -*- coding: utf-8 -*-
"""S3: independent generation-quality audit for Q3 (flavour-fingerprint generation).

Compares, on the 224 coordinates shared by 1.csv and the delivered generated profiles:
  T  = team's delivered 100 profiles / style  (method=interpolate)
  G1 = local 2-donor convex interpolation, same style   (reproduces the shrinkage mechanism)
  G2 = 3-donor Dirichlet barycentric recombination + small jitter
  G3 = style mean+covariance sampling in a shrunk PCA space, then empirical clip
       and defect projection to the threshold-safe halfspace
Metrics per style: variance ratio, near-duplicate rate, mean pairwise distance ratio,
Mahalanobis distance to the real style centre, coverage inside the real 95% radius,
marginal Wasserstein-1 (mean over coordinates).
"""
import numpy as np
import pandas as pd

ATT = r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件"
ROOT = r"C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件"
OUT = ROOT + r"\experiments\lead_review\q3"
rng = np.random.default_rng(20261027)
STY = ["Tripel", "Kriek", "Lambic"]

df = pd.read_csv(ATT + r"\1.csv", low_memory=False)
gen = pd.read_csv(ROOT + r"\outputs\task6_final_20261026\paper\tables\q3__generated_profiles.csv")
META = ["beer", "beer_id", "tasting_category_fine"]
gcoords = [c for c in gen.columns if c not in ["generated_id", "style", "method"]]
C = [c for c in gcoords if c in df.columns]
print(f"coordinates usable: {len(C)} (gen {len(gcoords)}, 1.csv {len(df.columns)-3})")
assert gen.groupby("style").size().eq(100).all(), gen.groupby("style").size()

# thresholds (same resolution as s2)
st = pd.read_excel(ATT + r"\8.xlsx", sheet_name="Sensory thresholds")


def th(f, w):
    r = st[st["field_name_python"].astype(str).str.strip() == f]
    return None if len(r) == 0 or pd.isna(r[w].iloc[0]) else float(r[w].iloc[0])


TD = {"DMS": ("DMS", 10.0 ** th("DMS", "threshold_median"), "log10"),
      "Diacetyl": ("X2.3.butanedione", th("2,3-butanedione", "threshold_median"), "raw"),
      "4VG": ("X4VG", th("X4VG", "threshold_median"), "log10")}

allreal = df[C].to_numpy(float)
med = np.nanmedian(allreal, 0)
med = np.where(np.isfinite(med), med, 0.0)
sdv_c = np.nanstd(allreal, 0)
sdv_c = np.where(sdv_c > 0, sdv_c, 1.0)
_sdf = df[C].copy()
_sdf = _sdf.fillna(pd.Series(med, index=C))
R = {s: _sdf.loc[df["tasting_category_fine"] == s].to_numpy(float) for s in STY}
allreal = _sdf.to_numpy(float)

# per-coordinate transform map: re-use calibrated sheet
cal = pd.read_excel(ATT + r"\8.xlsx", sheet_name="Calibrated chemical dataset")
m = df.merge(cal, on="beer_id", how="inner", suffixes=("", "_cal"))
TMAP = {}
for c in C:
    if c not in cal.columns:
        TMAP[c] = "raw"
        continue
    sub = m[[c, c + "_cal"]].dropna()
    if len(sub) < 40:
        TMAP[c] = "raw"
        continue
    rl = np.corrcoef(10 ** sub[c], sub[c + "_cal"])[0, 1]
    rr = np.corrcoef(sub[c], sub[c + "_cal"])[0, 1]
    TMAP[c] = "raw" if not np.isfinite(rl) else ("log10" if abs(rl) > abs(rr) + 0.02 else "raw")
TMAP["DMS"] = "log10"
TMAP["X4VG"] = "log10"


def z(x):
    return (x - med) / sdv_c


def unz(x):
    return x * sdv_c + med


def toconc(c, v):
    return 10.0 ** np.asarray(v, float) if TMAP.get(c, "raw") == "log10" else np.asarray(v, float)


def project(Xg):
    """defect-safe projection: every threshold defect <= min(style median, 0.9*T)."""
    Xg = Xg.copy()
    for k, (c, Tv, tr) in TD.items():
        if c not in C:
            continue
        j = C.index(c)
        capc = 0.9 * Tv
        cap = np.log10(capc) if tr == "log10" else capc
        cj = toconc(c, Xg[:, j])
        Xg[:, j] = np.where(cj > capc, cap, Xg[:, j])
    for c in ["humulene.epoxide.II", "caryophyllene.oxide", "dihydroisocaryophyllene.epoxide"]:
        if c in C:
            j = C.index(c)
            Xg[:, j] = np.minimum(Xg[:, j], np.nanmax(allreal[:, j]))  # never above library max
    return Xg


def metrics(Xg, Xr, ref=None):
    """Xg generated matrix, Xr real matrix (both raw coords, same columns); ref = real centroid set."""
    zr, zg = z(Xr), z(Xg)
    vr = np.nanvar(zr, 0)
    vg = np.nanvar(zg, 0)
    ok = vr > 1e-9
    var_ratio = float(np.nanmedian(vg[ok] / vr[ok]))
    # pairwise distances in z-space (subsample for speed)
    def pd_(M, n=1):
        idx = rng.choice(len(M), size=min(len(M), 60), replace=False)
        S = M[idx]
        D = np.sqrt(((S[:, None, :] - S[None, :, :]) ** 2).sum(-1))
        return D[np.triu_indices(len(S), 1)]
    dr, dg = pd_(zr), pd_(zg)
    near = float(np.mean(dg < 0.25 * np.median(dr)))
    dist_ratio = float(np.median(dg) / np.median(dr))
    # 8-PC whitened Mahalanobis to real style centre
    mu = zr.mean(0)
    Zc = zr - mu
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    k = min(8, len(Xr) - 1)
    P = Vt[:k].T
    lam = np.var(Zc @ P, 0)
    lam = np.maximum(lam, 1e-9)
    def md(M):
        q = (z(M) - mu) @ P
        return np.sqrt((q ** 2 / lam).sum(1))
    drm = md(Xr)
    dgm = md(Xg)
    cov = float(np.mean(dgm <= np.percentile(drm, 95)))
    # marginal W1
    qs = np.linspace(0.05, 0.95, 19)
    w1 = float(np.nanmean(np.abs(np.nanquantile(zg, qs, axis=0) - np.nanquantile(zr, qs, axis=0))))
    return dict(var_ratio=var_ratio, trace_ratio=float(np.nanmean(vg[ok] / vr[ok])),
                top3_eig_ratio=float(np.mean(np.sort(np.linalg.eigvalsh(np.cov(zg.T)))[-3:] /
                                             np.maximum(np.sort(np.linalg.eigvalsh(np.cov(zr.T)))[-3:], 1e-9))),
                near_dup_rate=near, pair_dist_ratio=dist_ratio,
                maha_mean=float(np.mean(dgm)), maha_real_mean=float(np.mean(drm)),
                coverage95=cov, marginal_w1=w1)


rows = []
for s in STY:
    Xr = R[s]
    # T: team delivered
    T = gen.loc[gen["style"] == s, C].to_numpy(float)
    # G1: 2-donor convex interpolation
    n = 100
    i1 = rng.integers(0, len(Xr), n)
    i2 = rng.integers(0, len(Xr), n)
    w = rng.uniform(0.2, 0.8, n)[:, None]
    G1 = w * Xr[i1] + (1 - w) * Xr[i2]
    # G2: 3-donor barycentric
    idx = rng.integers(0, len(Xr), (n, 3))
    Wt = rng.dirichlet([1, 1, 1], n)
    G2 = np.einsum("nk,nkc->nc", Wt, Xr[idx])
    G2 = G2 + rng.normal(0, 0.05, G2.shape) * sdv_c  # small multiplicative-ish jitter
    # G3: mean+cov sampling in shrunk PCA space, then clip + defect projection
    mu = Xr.mean(0)
    Zc = Xr - mu
    U, Sv, Vt = np.linalg.svd(Zc, full_matrices=False)
    k = min(8, len(Xr) - 1)
    P = Vt[:k].T
    sc = Zc @ P
    Cs = np.cov(sc.T) * (1 - 0.3) + np.diag(np.diag(np.cov(sc.T))) * 0.3
    samp = rng.multivariate_normal(np.zeros(k), Cs, n)
    G3 = mu + samp @ P.T
    lo, hi = np.nanmin(Xr, 0), np.nanmax(Xr, 0)
    G3 = np.clip(G3, lo, hi)
    G3 = project(G3)
    # G4: full-covariance (Ledoit-Wolf) Gaussian in z space + per-coordinate moment
    #     matching to the real style margins + clip + defect projection
    from sklearn.covariance import LedoitWolf
    Zr = z(Xr)
    mz = Zr.mean(0)
    lw = LedoitWolf(assume_centered=False).fit(Zr)
    Cs = lw.covariance_
    w_, V_ = np.linalg.eigh(Cs)
    w_ = np.clip(w_, 1e-12, None)
    samp = rng.standard_normal((n, len(w_))) @ (V_ * np.sqrt(w_)).T + mz
    sdr = Zr.std(0)
    sdr = np.where(sdr > 1e-9, sdr, 1e-9)
    sds = samp.std(0)
    sds = np.where(sds > 1e-9, sds, 1e-9)
    samp = mz + (samp - mz) * (sdr / sds)          # moment match per coordinate
    G4 = unz(samp)
    G4 = np.clip(G4, np.nanmin(Xr, 0), np.nanmax(Xr, 0))
    G4 = project(G4)
    print(f"  [{s}] Ledoit-Wolf shrinkage = {lw.shrinkage_:.3f}")
    # G5: 8-PC score resampling with exact real score covariance + per-coordinate
    #     residual-variance top-up (eigenvalue-calibrated Gaussian copula), then projection
    mu5 = Xr.mean(0)
    Zc5 = Xr - mu5
    U5, S5, V5 = np.linalg.svd(Zc5, full_matrices=False)
    k5 = min(8, len(Xr) - 1)
    P5 = V5[:k5].T
    sc5 = Zc5 @ P5
    samp5 = rng.multivariate_normal(np.zeros(k5), np.cov(sc5.T), n)
    recon = mu5 + samp5 @ P5.T
    target = np.nanvar(Zc5, 0) - np.var(recon, 0)
    resid_sd = np.sqrt(np.clip(target, 0, None))
    G5 = recon + rng.standard_normal((n, len(resid_sd))) * resid_sd
    G5 = np.clip(G5, np.nanmin(Xr, 0), np.nanmax(Xr, 0))
    G5 = project(G5)
    # G6: G5 + empirical quantile mapping of every coordinate onto the real style
    #     marginal (rank matching), then defect projection -> exact marginals, exact dispersion
    G6 = np.empty_like(G5)
    for j in range(G5.shape[1]):
        rj = np.sort(Xr[:, j])
        u = (np.argsort(np.argsort(G5[:, j])) + 0.5) / len(G5[:, j])
        G6[:, j] = np.interp(u, (np.arange(len(rj)) + 0.5) / len(rj), rj)
    G6 = project(G6)
    for name, G in [("team_interpolate", T), ("repro_2donor_convex", G1),
                    ("barycentric_3donor", G2), ("pca_mean_cov_projected", G3),
                    ("ledoitwolf_moment_matched", G4),
                    ("pc_score_resample_residual_topup", G5),
                    ("copula_quantile_mapped_projected", G6)]:
        d = metrics(G, Xr)
        d.update(style=s, method=name)
        rows.append(d)

res = pd.DataFrame(rows)[["style", "method", "var_ratio", "near_dup_rate", "pair_dist_ratio",
                          "maha_mean", "maha_real_mean", "coverage95", "marginal_w1"]]
res.to_csv(OUT + r"\s3_generation_quality.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 400)
print("\n== generation quality (real style sample = reference) ==")
for s in STY:
    print(f"\n--- {s} (n_real={len(R[s])}) ---")
    print(res[res.style == s].drop(columns="style").to_string(index=False))

print("\n== per-style real spread (for calibration) ==")
for s in STY:
    zr = z(R[s])
    print(f"{s:7s} real sd median={np.nanmedian(np.nanstd(zr,0)):.3f}  "
          f"n={len(R[s])}")
