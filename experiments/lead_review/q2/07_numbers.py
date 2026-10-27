# -*- coding: utf-8 -*-
import json
import pandas as pd

s = json.load(open("experiments/lead_review/q2/q2_summary.json", encoding="utf-8"))
print("method:", s["method"])
print("graph:", s["graph"])
print("bootstrap_freq:", json.dumps(s["bootstrap_freq"], ensure_ascii=False))
print("bootstrap_jaccard:", json.dumps(s["bootstrap_jaccard"], ensure_ascii=False))
print("method_jaccard_overall:", json.dumps(s["method_jaccard_overall"], ensure_ascii=False))
for k, v in s["method_jaccard"].items():
    print("  pair", k, json.dumps(v, ensure_ascii=False))
print("team_ridge_overlap:", s["team_ridge_overlap"])
print("evidence:", json.dumps(s["evidence"], ensure_ascii=False))
d = pd.read_csv("experiments/lead_review/q2/q2_top3_full50_graphridge.csv")
print("\nA rows:")
for r in d[d.chemical_evidence == "A"].itertuples():
    print(f"  {r.target:12s} {r.chemical_name:22s} f={r.bootstrap_freq:.2f} r6={r.r6_cross_corr}")
print("\nB examples by mechanism class:")
print(d[d.chemical_evidence == "B"].groupby("mechanism_class").size().to_string())
w = pd.read_csv("experiments/lead_review/q2/q2_top3_wide.csv")
print("\nwide cols:", list(w.columns))
print(w.head(3).to_string())
st = pd.read_csv("experiments/lead_review/q2/q2_stability_bootstrap.csv")
print("\nstability cols:", list(st.columns))
print("targets mean freq >=0.5:", int((st.mean_top3_freq >= 0.5).sum()))
print("targets bootstrap jaccard >=0.5:", int((st.bootstrap_jaccard_mean >= 0.5).sum()))
mj = pd.read_csv("experiments/lead_review/q2/q2_method_jaccard.csv")
print("targets all six pairs >=0.5:", int((mj["n_pairs_ge_0.5"] == 6).sum()))
print("targets at least 1 pair >=0.5:", int((mj["n_pairs_ge_0.5"] >= 1).sum()))
print("targets mean jaccard >=0.5:", int((mj.jaccard_mean >= 0.5).sum()))
