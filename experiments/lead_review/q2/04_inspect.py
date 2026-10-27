# -*- coding: utf-8 -*-
import pandas as pd

d = pd.read_csv("experiments/lead_review/q2/q2_top3_full50_graphridge.csv")
print(d.shape)
print("evidence:", d.chemical_evidence.value_counts().to_dict())
for t, g in list(d.groupby("target", sort=False))[:10]:
    parts = [f"{r.chemical_name}(f={r.bootstrap_freq:.2f},{r.chemical_evidence})" for r in g.itertuples()]
    print(t, "|", " ; ".join(parts))
print("\n== A rows ==")
print(d[d.chemical_evidence == "A"][["target", "chemical_name", "r6_cross_corr", "evidence_basis"]].to_string())
print("\n== B rows ==")
print(d[d.chemical_evidence == "B"][["target", "chemical_name", "r6_cross_corr", "evidence_basis"]].to_string())
print("\n== freq distribution ==")
print(d.bootstrap_freq.describe())
print("method jaccard head:")
print(pd.read_csv("experiments/lead_review/q2/q2_method_jaccard.csv").head(6).to_string())
