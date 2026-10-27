import pandas as pd
from pathlib import Path

out = Path("outputs/q3_migration_20261028")
new = pd.read_csv(out / "s2c_sensory_before_after_52.csv")
old = pd.read_csv("experiments/lead_review/q3/s2c_sensory_before_after_52_pre_N2fix.csv")
rows = []
for col in ["grade_raw", "prob_ge1", "prob_ge5", "prob_6", "raw_median_estimate"]:
    a = pd.to_numeric(old[col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(new[col], errors="coerce").to_numpy(dtype=float)
    both = ~pd.isna(a) & ~pd.isna(b)
    diff = both & (abs(a - b) > 1e-12)
    delta = abs(a[diff] - b[diff])
    rows.append({"column": col, "labelled_cells": int(both.sum()),
                 "mismatch_cells": int(diff.sum()),
                 "max_abs_delta": None if delta.size == 0 else round(float(delta.max()), 6),
                 "after_mismatch": int(diff[(new["role"].to_numpy() == "after") & diff].sum()),
                 "before_mismatch": int(diff[(new["role"].to_numpy() == "before") & diff].sum()),
                 "cases": int(new.loc[diff, "case_id"].nunique())})
table = pd.DataFrame(rows)
print(table.to_string(index=False))
print("grade_raw old unique:", sorted(old["grade_raw"].dropna().unique().tolist()))
print("grade_raw new unique:", sorted(new["grade_raw"].dropna().unique().tolist()))
