# -*- coding: utf-8 -*-
import json
import pandas as pd

d = pd.read_csv("experiments/lead_review/q2/q2_top3_full50_graphridge.csv")
print(d.shape)
print("evidence:", d.chemical_evidence.value_counts().to_dict())
print("confidence:", d.confidence.value_counts().to_dict())
named = ["diacetyl", "dms", "X4vg", "metallic", "t2n", "lactic", "acetic", "acidity", "alcohol",
         "co2", "clove", "orange", "bitternes", "stale_hops"]
for t in named:
    g = d[d.target == t]
    if len(g) == 0:
        continue
    parts = [f"{r.chemical_name}[{r.chemical_evidence},{r.confidence},f={r.bootstrap_freq:.2f}]"
             for r in g.itertuples()]
    print(t, "|", " ; ".join(parts))
print("\n== selected aroma/taste targets ==")
for t in ["A_malt_bread", "A_hops_citrus", "A_esters_banana", "F_esters_fruity", "F_malt_all"]:
    g = d[d.target == t]
    if len(g):
        parts = [f"{r.chemical_name}[{r.chemical_evidence},f={r.bootstrap_freq:.2f}]" for r in g.itertuples()]
        print(t, "|", " ; ".join(parts))
print("\nper-target evidence mix:")
print(pd.crosstab(d.target, d.chemical_evidence).head(20).to_string())
s = json.load(open("experiments/lead_review/q2/q2_summary.json", encoding="utf-8"))
print("\nbootstrap_freq:", s["bootstrap_freq"])
print("bootstrap_jaccard:", s["bootstrap_jaccard"])
print("method_jaccard_overall:", s["method_jaccard_overall"])
print("method_jaccard pairs:", {k: (round(v["mean"], 3), v["mean_overlap_n"]) for k, v in s["method_jaccard"].items()})
print("team_ridge_overlap:", s["team_ridge_overlap"])
print("evidence:", s["evidence"])
print("graph:", s["graph"])
