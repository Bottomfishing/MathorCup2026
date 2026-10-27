# -*- coding: utf-8 -*-
"""
Q2 rebuild: full 50-dimension x Top-3 compound delivery using 2.csv co-expression graph
+ 1.csv chemistry, with bootstrap stability and multi-method Jaccard.

Outputs -> experiments/lead_review/q2/
Run: .venv\\Scripts\\python.exe experiments\\lead_review\\q2\\02_rebuild_q2.py
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件")
SRC = Path(r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件")
OUT = ROOT / "experiments" / "lead_review" / "q2"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20261027)
ALPHA = 1000.0
LAM = 1000.0
GRAPH_THRESHOLD = 0.5
B_BOOT = 200

EXCLUDE = ["pH", "color.EBC.", "bitterness", "GFS", "unfermentables", "kcalperc",
           "esters_sum", "aroma_hops_sum", "acids_sum", "sulfur_sum",
           "protein.g.L.", "betaglucan.mg.L."]

# ---------------------------------------------------------------- helpers
def canon(s):
    s = unicodedata.normalize("NFKC", str(s))
    for g, w in [("α", "alpha"), ("β", "beta"), ("γ", "gamma"), ("δ", "delta")]:
        s = s.replace(g, w)
    s = s.lower()
    return re.sub(r"[^a-z0-9]", "", s)


fd = pd.read_csv(ROOT / "outputs/p0_audit_20261023_final/feature_dictionary.csv")
Xraw = pd.read_csv(SRC / "1.csv", encoding="utf-8-sig")
meta_cols = ["beer", "beer_id", "tasting_category_fine"]
feat_cols = [c for c in Xraw.columns if c not in meta_cols]
assert len(feat_cols) == 231, (len(feat_cols), Xraw.shape)

# 2.csv is the Spearman matrix of 1.csv, row order == 1.csv column order (verified:
# recomputing Spearman on 1.csv matches 2.csv positionally to 6.7e-16). Use positional map.
m2 = pd.read_csv(SRC / "2.csv", encoding="utf-8-sig", index_col=0)
m2.index = [str(x) for x in m2.index]
m2.columns = [str(x) for x in m2.columns]
assert list(m2.index) == list(m2.columns)
assert len(m2.index) == 231
node_names = list(m2.index)
node2field = {node_names[i]: feat_cols[i] for i in range(231)}
print(f"[map] 2.csv positional map to 1.csv: 231/231 nodes, 231 unique fields: "
      f"{len(set(node2field.values())) == 231}")

# fd fields are the 1.csv column names in field_name_R
fd["field"] = fd["field_name_R"]
key2field = {}
for _, r in fd.iterrows():
    for raw in (r["chemical_name"], r["chemical_name_short"], r["field_name_R"]):
        key2field.setdefault(canon(raw), r["field"])

# target column order (4.csv) -> 6.csv column names
t4 = pd.read_csv(SRC / "4.csv", encoding="utf-8-sig")
META4 = ["beer", "beer_id", "tasting_category_fine"]
TARGETS = [c for c in t4.columns if c not in META4]
m6 = pd.read_csv(SRC / "6.csv", encoding="utf-8-sig")
cols6 = [str(c) for c in m6.columns[1:]]
special6 = {"alcohol": "Alcohol", "aftertaste": "Aftertaste", "body": "Body",
            "co2": "Carbonation", "overall": "Overall appreciation"}
col6_of_target = {}
for i, t in enumerate(TARGETS):
    if i < 45:
        col6_of_target[t] = cols6[i]
    else:
        hit = [c for c in cols6 if c.replace("\r\n", " ").strip().lower().startswith(special6[t].lower())]
        col6_of_target[t] = hit[0] if hit else cols6[i]
# map 6.csv rows (chemical) -> field
row6_field = {}
for i, name in enumerate(m6.iloc[:, 0].astype(str)):
    k = canon(name)
    if k in key2field:
        row6_field[i] = key2field[k]
r6 = np.full((len(m6), len(TARGETS)), np.nan)
for i, fld in row6_field.items():
    for j, t in enumerate(TARGETS):
        r6[i, j] = m6.iloc[i][col6_of_target[t]]
field2r6 = {}
for i, fld in row6_field.items():
    field2r6[fld] = i
print(f"[map] 6.csv rows mapped={len(row6_field)}/{len(m6)}  cols aligned=50")

# ---------------------------------------------------------------- graph
A_full = m2.abs().to_numpy()
np.fill_diagonal(A_full, 0.0)
node_names = list(m2.index)
idx = {n: i for i, n in enumerate(node_names)}
fields_order = [node2field[n] for n in node_names]

try:
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(node_names)))
    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):
            if A_full[i, j] >= GRAPH_THRESHOLD:
                G.add_edge(i, j, weight=float(A_full[i, j]))
    communities = list(nx.algorithms.community.louvain_communities(G, seed=42))
    communities = sorted(communities, key=len, reverse=True)
    node_comm = {}
    for cid, members in enumerate(communities):
        for n in members:
            node_comm[n] = cid
    print(f"[graph] threshold={GRAPH_THRESHOLD} nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
          f"components={nx.number_connected_components(G)} communities={len(communities)} "
          f"largest_comm={len(communities[0])}")
except ImportError:
    raise SystemExit("networkx required")

comm_rows = []
for i, n in enumerate(node_names):
    comm_rows.append({"chemical_name": n, "field": fields_order[i],
                      "degree_graph": int(G.degree(i)),
                      "community_id": node_comm[i],
                      "community_size": len(communities[node_comm[i]]),
                      "top3_neighbors": "; ".join(
                          f"{node_names[k]}({A_full[i, k]:.3f})"
                          for k in sorted(G.neighbors(i), key=lambda k: -A_full[i, k])[:3])})
pd.DataFrame(comm_rows).to_csv(OUT / "q2_network_communities.csv", index=False, encoding="utf-8-sig")
edges = [{"source": node_names[u], "target": node_names[v], "weight": float(d["weight"])}
         for u, v, d in G.edges(data=True)]
edges.sort(key=lambda e: -e["weight"])
pd.DataFrame(edges).to_csv(OUT / "q2_network_edges_0_5.csv", index=False, encoding="utf-8-sig")
print(f"[graph] wrote nodes/edges tables")

# ---------------------------------------------------------------- matrix
y_df = t4[TARGETS].copy()
Y = y_df.to_numpy(dtype=float)
X = Xraw[feat_cols].to_numpy(dtype=float)
col_idx = {c: i for i, c in enumerate(feat_cols)}
cand = [c for c in feat_cols if c not in EXCLUDE]
cand_idx = [col_idx[c] for c in cand]
print(f"[data] X={X.shape} Y={Y.shape} candidates={len(cand)}")

L = np.zeros((len(cand), len(cand)))
pos = {f: i for i, f in enumerate(cand)}
for e in edges:
    f1, f2 = node2field[e["source"]], node2field[e["target"]]
    if f1 in pos and f2 in pos:
        L[pos[f1], pos[f2]] = L[pos[f2], pos[f1]] = 1.0
deg = L.sum(1)
deg_safe = np.where(deg > 0, deg, 1.0)
L_sym = np.eye(len(cand)) - L / np.sqrt(np.outer(deg_safe, deg_safe))
print(f"[graph] laplacian on candidates: {len(cand)}x{len(cand)} mean_deg={deg.mean():.2f}")


def preprocess(Xr, Yr):
    """median impute (train columns), asinh, standardize."""
    Xf = Xr.copy()
    med = np.nanmedian(Xf, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    nan = np.isnan(Xf)
    if nan.any():
        Xf[nan] = np.take(med, np.where(nan)[1])
    Xa = np.arcsinh(Xf)
    mu, sd = Xa.mean(0), Xa.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Xs = (Xa - mu) / sd
    y = Yr - Yr.mean(0, keepdims=True)
    return Xs, y


def solve(Xs, y, alpha):
    """Ridge coefficients on the 219 individual-analyte candidates (team alpha=1000)."""
    Xc = Xs[:, cand_idx]
    G = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
    return np.linalg.solve(G, Xc.T @ y)  # p x 50


def score_graph(B):
    """Graph-neighbourhood weighted importance: s_j = |b_j| + 0.5 * mean_j_neighbors |b|.

    Uses the 2.csv co-expression graph (|rho| >= GRAPH_THRESHOLD) as neighbourhood
    context, so a compound whose graph neighbours also carry signal is promoted.
    The self term keeps direct identity markers on top (chemical interpretability).
    """
    S = np.zeros_like(B)
    for j in range(B.shape[1]):
        a = np.abs(B[:, j])
        nb = (L @ a) / deg_safe
        S[:, j] = a + 0.5 * nb
    return S


def top3(B, cols):
    out = []
    for j in range(B.shape[1]):
        order = np.argsort(-np.abs(B[:, j]), kind="stable")[:3]
        out.append([cols[k] for k in order])
    return out


Xs, Yc = preprocess(X, Y)
B_net = solve(Xs, Yc, ALPHA)
B_ridge = B_net
S_net = score_graph(B_net)
sets_net = top3(S_net, cand)
sets_ridge = top3(B_ridge, cand)

# ExtraTrees importance
from sklearn.ensemble import ExtraTreesRegressor
ets = np.zeros((len(cand), len(TARGETS)))
Xc = Xs[:, cand_idx]
for j, t in enumerate(TARGETS):
    et = ExtraTreesRegressor(n_estimators=300, max_features=0.33, min_samples_leaf=2,
                             random_state=42, n_jobs=-1)
    et.fit(Xc, y_df[t].to_numpy())
    ets[:, j] = et.feature_importances_
sets_et = top3(ets, cand)

# marginal Spearman
from scipy.stats import spearmanr
Xdf = pd.DataFrame(X, columns=feat_cols)
Xdf = Xdf.apply(lambda s: s.fillna(s.median()))
marg = np.zeros((len(cand), len(TARGETS)))
Xc_raw = Xdf[cand].to_numpy()
for j, t in enumerate(TARGETS):
    yv = y_df[t].to_numpy()
    rho = np.array([spearmanr(Xc_raw[:, k], yv).statistic for k in range(len(cand))])
    marg[:, j] = rho
sets_marg = top3(marg, cand)
print("[model] full-data fits done")

# ---------------------------------------------------------------- bootstrap by beer
B_net_refit = {t: [] for t in TARGETS}
n_beer = X.shape[0]
for b in range(B_BOOT):
    rs = RNG.integers(0, n_beer, n_beer)
    Xs_b, Yc_b = preprocess(X[rs], Y[rs])
    Bb = solve(Xs_b, Yc_b, ALPHA)
    Sb = score_graph(Bb)
    for j, t in enumerate(TARGETS):
        order = np.argsort(-Sb[:, j], kind="stable")[:3]
        B_net_refit[t].append(tuple(cand[k] for k in order))
    if (b + 1) % 50 == 0:
        print(f"  bootstrap {b+1}/{B_BOOT}")
print("[bootstrap] done")

# ---------------------------------------------------------------- stability tables
freq_rows, jac_rows, top_rows = [], [], []
from itertools import combinations
PAIRS = [("graphridge", "ridge"), ("graphridge", "extratrees"), ("graphridge", "marginal_spearman"),
         ("ridge", "extratrees"), ("ridge", "marginal_spearman"), ("extratrees", "marginal_spearman")]
setmap = {"graphridge": sets_net, "ridge": sets_ridge, "extratrees": sets_et, "marginal_spearman": sets_marg}
pair_vals = {p: [] for p in PAIRS}
pair_ov = {p: [] for p in PAIRS}
target_jac_mean, target_jac_min, target_all_ge05 = {}, {}, {}
boot_jac = {}
boot_counts = {}

for j, t in enumerate(TARGETS):
    counts = {}
    for rep in B_net_refit[t]:
        for f in rep:
            counts[f] = counts.get(f, 0) + 1
    boot_counts[t] = counts
    freqs = {f: counts.get(f, 0) / B_BOOT for f in sets_net[j]}
    top_rows.append({"target": t, "rank1": sets_net[j][0], "rank2": sets_net[j][1], "rank3": sets_net[j][2],
                     "freq_rank1": freqs[sets_net[j][0]], "freq_rank2": freqs[sets_net[j][1]],
                     "freq_rank3": freqs[sets_net[j][2]],
                     "mean_freq": float(np.mean(list(freqs.values())))})
    freq_rows.append({"target": t, **{f"rank{k+1}_freq": freqs[sets_net[j][k]] for k in range(3)},
                      "mean_top3_freq": float(np.mean(list(freqs.values()))),
                      "min_top3_freq": float(np.min(list(freqs.values()))),
                      "n_slots_freq_ge_0.5": int(sum(v >= 0.5 for v in freqs.values())),
                      "bootstrap_jaccard_mean": np.nan,
                      "bootstrap_jaccard_min": np.nan})
    jv = [len(set(rep) & set(sets_net[j])) / len(set(rep) | set(sets_net[j])) for rep in B_net_refit[t]]
    boot_jac[t] = (float(np.mean(jv)), float(np.min(jv)), float(np.percentile(jv, 10)))
    freq_rows[-1]["bootstrap_jaccard_mean"] = float(np.mean(jv))
    freq_rows[-1]["bootstrap_jaccard_min"] = float(np.min(jv))
    jrow = {"target": t}
    for a, b in PAIRS:
        inter = len(set(setmap[a][j]) & set(setmap[b][j]))
        uni = len(set(setmap[a][j]) | set(setmap[b][j]))
        v = inter / uni
        jrow[f"jaccard_{a}_vs_{b}"] = v
        jrow[f"overlap_n_{a}_vs_{b}"] = inter
        pair_vals[(a, b)].append(v)
        pair_ov[(a, b)].append(inter)
    vals = [jrow[f"jaccard_{a}_vs_{b}"] for a, b in PAIRS]
    jrow["jaccard_mean"] = float(np.mean(vals))
    jrow["jaccard_min"] = float(np.min(vals))
    jrow["n_pairs_ge_0.5"] = int(sum(v >= 0.5 for v in vals))
    jac_rows.append(jrow)
    target_jac_mean[t], target_jac_min[t] = float(np.mean(vals)), float(np.min(vals))
    target_all_ge05[t] = int(sum(v >= 0.5 for v in vals))

freq_df = pd.DataFrame(freq_rows)
jac_df = pd.DataFrame(jac_rows)
freq_df.to_csv(OUT / "q2_stability_bootstrap.csv", index=False, encoding="utf-8-sig")
jac_df.to_csv(OUT / "q2_method_jaccard.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(top_rows).to_csv(OUT / "q2_top3_wide.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- chemical rationale
# class_key -> (Chinese class name, mechanism text, affinity families of sensory targets)
CLASS_RULES = [
    (("diacetyl", "butanedione"), "diacetyl", "双乙酰(2,3-丁二酮)",
     "奶油/黄油香，啤酒中典型发酵副产物，阈值约0.1 mg/L量级；6.csv 'Diacetyl' 行与 diacetyl 维度交叉相关可作经验佐证",
     {"diacetyl"}),
    (("dimethylsulfide", "dms", "diethylsulfide", "dimethyldisulfide", "methanethiol", "hydrogensulfide",
      "h2s", "sulfurdioxide", "so2", "cs2", "carbon disulfide"),
     "sulfur", "硫化物类", "还原味/熟蔬菜/硫味；DMS 阈值约17-25 μg/L，off_flavors 标注 cooked vegetables/corn/onion",
     {"dms", "stale", "sulfur"}),
    (("vinylguaiacol", "guaiacol", "ethylguaiacol", "ethylphenol", "cresol", "eugenol", "vanillin",
      "isoeugenol", "methoxycinnamaldehyde", "cinnamaldehyde", "salicylaldehyde"),
     "phenol", "酚类/香料", "丁香/烟熏/香料；4VG 为阿魏酸脱羧产物，对应 X4vg 与 clove 维度；酚类阈值低、感知强",
     {"phenol", "clove", "coriander", "malt"}),
    (("furfural", "furfuryl", "acetylfuran", "furanone", "furaneol", "maltol", "isomaltol", "furoate",
      "methylfurfural", "hydroxymethylfurfural", "furan"),
     "maillard", "呋喃/美拉德产物", "焦糖/面包/烘烤香；美拉德反应与麦芽焙烤产物，是麦芽香与焦糖维度的直接化学来源",
     {"malt", "sweet", "body", "overall", "aftertaste"}),
    (("pyrazine",), "pyrazine", "吡嗪类", "烘烤/坚果/可可香；美拉德反应产物，常见于深色麦芽",
     {"malt", "overall", "aftertaste"}),
    (("linalool", "geraniol", "citronellol", "nerol", "nerolidol", "terpineol", "terpinen", "terpinolene",
      "ocimene", "ocimenone", "phellandrene", "limonene", "eucalyptol", "camphor", "isoborneol",
      "caryophyllene", "humulene", "farnesene", "farnesol", "farnesal", "calacorene", "muurolol",
      "eudesmol", "cubenol", "cadinene", "ionone", "damascenone", "cyclocitral", "cyclohomocitral",
      "terpinyl", "cymenene", "cymene", "manool", "kongol", "eoeniculin", "tobacarol", "epicubenol",
      "dihydroapofarnesal", "linalool oxide", "nerol oxide", "linalooloxide"),
     "terpene", "酒花萜类/倍半萜", "花香/柑橘/草本/木质与酒花香气；酒花来源，萜烯在发酵与老化中转化，是 hops/orange 维度的核心化学基础",
     {"hops", "orange", "coriander", "esters", "overall"}),
    (("acetate", "butyrate", "butanoate", "hexanoate", "octanoate", "decanoate", "propionate", "isovalerate",
      "isobutyrate", "tiglate", "benzoate", "cinnamate", "phenylacetate", "succinate", "caprylate",
      "lactate", "formate", "isobutyrate", "heptanoate", "nonanoate", "pentanoate"),
     "ester", "酯类", "果香/香蕉/花香/nail polish；酵母乙酰/酰基转移酶合成，是 esters 与 fruity 维度的直接来源",
     {"esters", "hops", "sweet", "overall", "alcohol"}),
    (("acetaldehyde", "nonanal", "nonenal", "decadienal", "benzaldehyde", "phenylacetaldehyde",
      "methylbutanal", "methyl-2-butenal", "methylbutyraldehyde", "isovaleraldehyde", "hexanal",
      "heptanal", "octanal", "methional"),
     "aldehyde", "醛类", "青苹果/生青/纸板/坚果；乙醛为乙醇前体，壬烯醛为脂肪氧化陈腐标志物，醛类是 acetaldehyde/t2n 维度的直接来源",
     {"acetaldehyde", "stale", "malt", "overall"}),
    (("ethanol", "propanol", "butanol", "pentanol", "hexanol", "heptanol", "octanol", "nonanol", "decanol",
      "isoamylalcohol", "isobutanol", "phenethylalcohol", "benzylalcohol", "methylbutanol", "hexenol",
      "octenol", "methylhexanol", "methyloctanol", "methionol"),
     "alcohol", "醇类", "酒精/溶剂/玫瑰香；高级醇由酵母氨基酸代谢生成，是 alcohol 与 body 维度的重要贡献者",
     {"alcohol", "body", "overall", "esters"}),
    (("lactic", "acetic", "butyric", "isovaleric", "hexanoic", "octanoic", "decanoic", "succinic", "propionic"),
     "acid", "有机酸", "酸味/酸奶/醋味；乳酸菌与醋酸菌代谢产物，是 acidity/lactic/acetic 维度的直接来源",
     {"acid", "lactic", "acetic", "overall"}),
    (("glycerol",), "polyol", "甘油", "甜润/酒体厚重感；发酵副产物，主要影响 body 与甜感",
     {"body", "sweet", "aftertaste", "overall"}),
    (("iron",), "iron", "铁离子", "金属味；铁离子催化氧化并直接贡献 metallic 味",
     {"iron"}),
    (("isoalphaacids", "bitterness", "alphaacid", "betaacid"), "bitter", "异α-酸/苦味物质",
     "苦味；酒花α-酸异构化产物，是 bitterness 维度的直接化学基础",
     {"bitter"}),
    (("carbondioxide", "co2"), "gas", "二氧化碳", "碳酸感/杀口感；直接对应 co2 维度",
     {"gas"}),
    (("sugars", "gfs", "glucose", "fructose", "maltose", "unfermentables", "kcalperc"),
     "sugar", "糖类/热量", "甜味/麦芽香/酒体；可发酵糖与残糖直接贡献 sweetness/body",
     {"sweet", "body", "malt", "overall"}),
    (("protein", "betaglucan", "glycerin"), "protein", "蛋白/β-葡聚糖", "酒体/口感厚度；大分子物质影响 body 与 aftertaste",
     {"body", "aftertaste"}),
    (("color", "ebc"), "color", "色度", "焦香/烘烤外观与麦芽香；色度与深色麦芽焙烤产物共变，可作 malt 维度的外观佐证",
     {"malt", "body", "overall"}),
    (("ammonia", "methylthio", "dimethyltrisulfide", "hydrogen sulfide"), "off", "含硫/含氮异味物",
     "还原味/异味；off_flavors 标注的异味化合物，与 stale/硫味维度相关", {"stale", "dms", "sulfur"}),
    (("undecanone", "nonanone", "heptanone", "pentanone", "methylheptenone", "acetophenone", "methyl ketone",
      "butanone", "methylbutanone"), "ketone", "酮类", "果香/奶油/青草；部分甲基酮是发酵与氧化产物",
     {"esters", "malt", "overall"}),
]

TARGET_FAMILY = {}
for t in TARGETS:
    tl = t.lower()
    if tl.startswith(("a_malt", "f_malt")):
        fam = {"malt"}
    elif tl.startswith(("a_hops", "f_hops")):
        fam = {"hops"}
    elif tl.startswith(("a_esters", "f_esters")):
        fam = {"esters"}
    else:
        fam = {"acidity": {"acid"}, "bitternes": {"bitter"}, "sweetness": {"sweet"},
               "x4vg": {"phenol", "clove"}, "diacetyl": {"diacetyl"}, "dms": {"dms", "sulfur"},
               "metallic": {"iron"}, "stale_hops": {"stale", "sulfur"}, "t2n": {"stale"},
               "orange": {"hops", "terpene"}, "coriander": {"terpene", "phenol"},
               "clove": {"phenol", "clove"}, "lactic": {"acid", "lactic"}, "acetic": {"acid", "acetic"},
               "barnyard": {"other"}, "alcohol": {"alcohol"}, "aftertaste": {"body", "aftertaste", "overall"},
               "body": {"body", "protein", "sugar"}, "co2": {"gas"},
               "overall": {"malt", "hops", "esters", "alcohol", "body", "sweet"}}.get(tl, {"overall"})
    TARGET_FAMILY[t] = fam


def classify(name):
    n = canon(name)
    for keys, ck, cn, mech, fam in CLASS_RULES:
        if any(k in n for k in keys):
            return ck, cn, mech, fam
    return "other", "其他化合物", "通过共表达图邻域与1.csv丰度关联进入Top-3；机理直接证据不足，列为探索性关联", {"other"}


ORIGIN_CN = {"Yeast": "酵母代谢", "Hops": "酒花来源", "Malt": "麦芽来源", "Spices": "香料添加",
             "Others": "其他来源", "Wild flora": "野生菌群", "Internal standard solution": "内标"}
DIRECT_CLASSES = {"diacetyl", "sulfur", "phenol", "acid", "iron", "bitter", "gas", "sugar"}
name_by_field = dict(zip(fd["field"], fd["chemical_name"]))
origin_by_field = dict(zip(fd["field"], fd["origin_paper"]))
off_by_field = dict(zip(fd["field"], fd["off_flavors"]))
threshold_by_field = dict(zip(fd["field"], fd["threshold_mean"]))

identity = {}
ev = ROOT / "configs/q2_evidence.json"
if ev.exists():
    cfg_ev = json.loads(ev.read_text(encoding="utf-8"))
    for k, v in cfg_ev.get("identity_markers", {}).items():
        vals = v if isinstance(v, list) else [v]
        identity[k] = [x for x in vals if x in feat_cols]
identity.update({"diacetyl": ["X2.3.butanedione"], "dms": ["DMS"], "X4vg": ["X4VG"],
                 "metallic": ["iron"], "t2n": ["X.E..2.nonenal"], "lactic": ["lactic_acid.mg.L."],
                 "acetic": ["acetic_acid.g.L."], "alcohol": ["ethanol..v.v."], "co2": ["CO2.PSI."]})

detail_rows = []
node_by_field = {node2field[n]: n for n in node_names}
node_index_by_field = {node2field[n]: i for i, n in enumerate(node_names)}
node_deg_by_field = {node2field[n]: G.degree(i) for i, n in enumerate(node_names)}
for j, t in enumerate(TARGETS):
    counts = boot_counts[t]
    fam = TARGET_FAMILY[t]
    for rank, fld in enumerate(sets_net[j], 1):
        cname = name_by_field.get(fld, fld)
        origin = origin_by_field.get(fld, "Others")
        ck, cls_name, mech, cls_fam = classify(cname)
        affinity = bool(fam & cls_fam)
        r6v = r6[field2r6[fld], j] if fld in field2r6 else np.nan
        sign_agree = (np.sign(r6v) == np.sign(B_net[pos[fld], j])) if (fld in field2r6 and not np.isnan(r6v)) else np.nan
        is_id = fld in identity.get(t, [])
        freq = counts.get(fld, 0) / B_BOOT
        # evidence tier
        if is_id:
            tier = "A"
            basis = f"身份标记物：{cname} 为 {t} 维度同名/直系化合物" + (
                f"，6.csv 交叉相关 r={r6v:.2f}" if not np.isnan(r6v) else "")
        elif affinity and (not np.isnan(r6v)) and abs(r6v) >= 0.2 and sign_agree is True:
            tier = "A"
            basis = f"机理同族({cls_name}) + 6.csv 交叉相关 r={r6v:.2f} 同号"
        elif affinity and ck in DIRECT_CLASSES and freq >= 0.5:
            tier = "A"
            basis = f"直接机理物质({cls_name}) 且 bootstrap 入选率 {freq:.2f}≥0.5"
        elif affinity:
            tier = "B"
            basis = f"化学机理同族({cls_name})，与目标感官语义一致"
        elif (not np.isnan(r6v)) and abs(r6v) >= 0.2 and sign_agree is True:
            tier = "B"
            basis = f"6.csv 交叉相关 r={r6v:.2f} 同号，但机理类别与目标语义非直接对应"
        else:
            tier = "C"
            basis = "图邻域+1.csv 关联证据；机理仅间接，列为探索性候选"
        if tier == "A" and freq >= 0.5:
            confidence = "high"
        elif tier in ("A", "B") and freq >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        detail_rows.append({
            "target": t, "target_family": "/".join(sorted(fam)), "rank": rank,
            "field": fld, "chemical_name": cname, "origin_paper": origin,
            "ridge_coef": float(B_net[pos[fld], j]),
            "graph_importance": float(S_net[pos[fld], j]),
            "importance_rank": int(np.where(np.argsort(-S_net[:, j]) == pos[fld])[0][0] + 1),
            "bootstrap_freq": freq, "bootstrap_jaccard_vs_full": boot_jac[t][0],
            "r6_cross_corr": float(r6v) if not np.isnan(r6v) else np.nan,
            "r6_sign_agree": bool(sign_agree) if not np.isnan(r6v) else "NA",
            "is_identity_marker": bool(is_id), "mechanism_class": cls_name,
            "class_key": ck, "mechanism_note": mech,
            "target_semantic_match": affinity, "chemical_evidence": tier,
            "evidence_basis": basis, "confidence": confidence,
            "off_flavors": str(off_by_field.get(fld, "") or ""),
            "threshold_mean": float(threshold_by_field.get(fld)) if pd.notna(threshold_by_field.get(fld)) else np.nan,
            "degree_graph": int(node_deg_by_field[fld]),
            "community_id": int(node_comm[node_index_by_field[fld]]),
        })
det_df = pd.DataFrame(detail_rows)
det_df.to_csv(OUT / "q2_top3_full50_graphridge.csv", index=False, encoding="utf-8-sig")

# overlap with the team's existing full-data Ridge table (cross-pipeline consistency)
team = pd.read_csv(ROOT / "outputs/task6_final_20261026/paper/tables/q2_standardized__top3_panel_all50.csv")
team["target"] = team["target"].astype(str)
team_sets = team.groupby("target")["field"].apply(lambda s: set(s)).to_dict()
overlap = {}
for t in TARGETS:
    a, b = set(sets_net[TARGETS.index(t)]), team_sets.get(t, set())
    if a and b:
        overlap[t] = len(a & b) / len(a | b)
ov = pd.Series(overlap)
print(f"[overlap] vs team Ridge table: mean Jaccard={ov.mean():.3f} median={ov.median():.3f} "
      f"n>=0.5={(ov >= 0.5).sum()}/{len(ov)}")

# ---------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
ax.hist(det_df["bootstrap_freq"], bins=15, color="#4c72b0", edgecolor="white")
ax.axvline(0.5, color="#c44e52", ls="--", lw=1.2)
ax.set_xlabel("bootstrap selection frequency of Top-3 slots")
ax.set_ylabel("number of slots")
ax.set_title("(a) 150 Top-3 slots (200x beer bootstrap)")
ax = axes[1]
mf = freq_df.sort_values("mean_top3_freq")
ax.bar(range(len(mf)), mf["mean_top3_freq"], color="#55a868")
ax.axhline(0.5, color="#c44e52", ls="--", lw=1)
ax.set_xlabel("50 sensory targets (sorted)")
ax.set_ylabel("mean bootstrap frequency")
ax.set_title("(b) per-target mean Top-3 frequency")
plt.tight_layout()
plt.savefig(OUT / "fig_q2_bootstrap_stability.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
labels = [f"{a}\nvs {b}" for a, b in PAIRS]
means = [np.mean(pair_vals[p]) for p in PAIRS]
ax.bar(range(len(PAIRS)), means, color="#4c72b0")
ax.set_xticks(range(len(PAIRS)))
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("mean Top-3 Jaccard")
ax.set_title("(a) pairwise method agreement")
ax = axes[1]
ax.hist(jac_df["jaccard_mean"], bins=12, color="#8172b2", edgecolor="white")
ax.axvline(0.5, color="#c44e52", ls="--", lw=1.2)
ax.set_xlabel("per-target mean Jaccard over 6 method pairs")
ax.set_ylabel("number of targets")
ax.set_title("(b) 50 targets")
plt.tight_layout()
plt.savefig(OUT / "fig_q2_method_jaccard.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
ax = axes[0]
order = ["A", "B", "C"]
cnt = [int((det_df.chemical_evidence == k).sum()) for k in order]
bars = ax.bar(order, cnt, color=["#55a868", "#dd8452", "#c44e52"])
ax.set_ylabel("number of Top-3 slots")
ax.set_title("(a) chemical evidence tier (150 slots)")
for b, v in zip(bars, cnt):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center")
ax = axes[1]
conf = det_df["confidence"].value_counts().reindex(["high", "medium", "low"]).fillna(0)
bars = ax.bar(conf.index, conf.values, color=["#55a868", "#dd8452", "#c44e52"])
for b, v in zip(bars, conf.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, str(int(v)), ha="center")
ax.set_ylabel("number of Top-3 slots")
ax.set_title("(b) per-entry confidence (evidence & frequency)")
plt.tight_layout()
plt.savefig(OUT / "fig_q2_evidence_confidence.png", dpi=160)
plt.close(fig)
print("[fig] wrote 3 figures")

# ---------------------------------------------------------------- summary
tier_counts = det_df["chemical_evidence"].value_counts().to_dict()
conf_counts = det_df["confidence"].value_counts().to_dict()
n_targets_id = int(det_df[det_df.is_identity_marker].target.nunique())
freqs_all = det_df["bootstrap_freq"].to_numpy()
boot_stats = pd.DataFrame([{"target": t, "boot_jaccard_mean": v[0], "boot_jaccard_min": v[1],
                            "boot_jaccard_p10": v[2]} for t, v in boot_jac.items()])
summary = {
    "generated": "2026-10-27",
    "method": f"Ridge (alpha={ALPHA}) on 219 individual analytes + 2.csv |rho|>={GRAPH_THRESHOLD} "
              f"co-expression graph neighbourhood weighting (score = |coef| + 0.5*neighbour_mean|coef|) "
              f"+ 200x beer bootstrap + 4-method Top-3 Jaccard",
    "n_targets": len(TARGETS),
    "n_top3_rows": len(det_df),
    "n_candidate_compounds": len(cand),
    "graph": {"threshold": GRAPH_THRESHOLD, "nodes": int(G.number_of_nodes()),
              "edges": int(G.number_of_edges()), "communities": len(communities),
              "largest_community": int(len(communities[0])),
              "mean_degree": float(np.mean([G.degree(i) for i in range(len(node_names))]))},
    "coverage": {"targets_with_3_rows": int(det_df.groupby("target").size().eq(3).sum()),
                 "total_targets": len(TARGETS)},
    "bootstrap_freq": {"mean_all_slots": float(freqs_all.mean()),
                       "median_all_slots": float(np.median(freqs_all)),
                       "share_slots_ge_0.5": float((freqs_all >= 0.5).mean()),
                       "n_slots_ge_0.5": int((freqs_all >= 0.5).sum()),
                       "target_mean_freq_min": float(freq_df.mean_top3_freq.min()),
                       "target_mean_freq_median": float(freq_df.mean_top3_freq.median()),
                       "target_mean_freq_max": float(freq_df.mean_top3_freq.max())},
    "bootstrap_jaccard": {"mean_over_targets": float(boot_stats.boot_jaccard_mean.mean()),
                          "median_over_targets": float(boot_stats.boot_jaccard_mean.median()),
                          "min_over_targets": float(boot_stats.boot_jaccard_mean.min()),
                          "n_targets_mean_ge_0.5": int((boot_stats.boot_jaccard_mean >= 0.5).sum())},
    "method_jaccard": {f"{a}_vs_{b}": {"mean": float(np.mean(v)), "median": float(np.median(v)),
                                       "min": float(np.min(v)),
                                       "mean_overlap_n": float(np.mean(pair_ov[(a, b)])),
                                       "n_targets_ge_0.5": int(np.sum(np.array(v) >= 0.5))}
                       for (a, b), v in pair_vals.items()},
    "method_jaccard_overall": {"target_mean_min": float(jac_df.jaccard_mean.min()),
                               "target_mean_median": float(jac_df.jaccard_mean.median()),
                               "target_mean_mean": float(jac_df.jaccard_mean.mean()),
                               "mean_overlap_across_six_pairs": float(np.mean([np.mean(pair_ov[p]) for p in PAIRS])),
                               "n_targets_mean_ge_0.5": int((jac_df.jaccard_mean >= 0.5).sum()),
                               "n_targets_all_six_pairs_ge_0.5": int((jac_df["n_pairs_ge_0.5"] == 6).sum())},
    "team_ridge_overlap": {"mean_jaccard": float(ov.mean()), "median_jaccard": float(ov.median()),
                           "n_targets_ge_0.5": int((ov >= 0.5).sum()), "n_targets": int(len(ov))},
    "evidence": {"tier_counts": tier_counts,
                 "tier_A_share": float((det_df.chemical_evidence == "A").mean()),
                 "tier_AB_share": float(det_df.chemical_evidence.isin(["A", "B"]).mean()),
                 "confidence_counts": conf_counts,
                 "targets_with_identity_marker_in_top3": n_targets_id,
                 "n_rows_6csv_crosscorr_available": int(det_df.r6_cross_corr.notna().sum())},
    "previous_team_table": {"q2_standardized__top3_panel_all50_rows": 150, "ridge_coverage": "50/50",
                            "gbr_raw_nonzero_targets": 17, "gbr_identified_rows": 51,
                            "interpretation": "17/50 is a GBR-raw-score sub-table, not Q2 coverage; Ridge sub-table already 50/50"},
}
(OUT / "q2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["coverage", "bootstrap_freq", "bootstrap_jaccard",
                                          "method_jaccard_overall", "team_ridge_overlap", "evidence"]},
                 ensure_ascii=False, indent=1))
print(f"[out] {OUT}")

