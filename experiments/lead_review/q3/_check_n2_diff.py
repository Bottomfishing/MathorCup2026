import pandas as pd
from pathlib import Path

out = Path("outputs/q3_migration_20261028")
new = pd.read_csv(out / "s2c_sensory_before_after_52.csv")
old = pd.read_csv("experiments/lead_review/q3/s2c_sensory_before_after_52_pre_N2fix.csv")
label = ["grade_raw", "prob_ge1", "prob_ge5", "prob_6", "raw_median_estimate"]
cell = 0
max_delta = 0.0
cases = set()
for col in label:
    a = pd.to_numeric(old[col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(new[col], errors="coerce").to_numpy(dtype=float)
    an, bn = ~pd.isna(a), ~pd.isna(b)
    diff = (an & bn) & (abs(a - b) > 1e-12)
    cell += int(diff.sum())
    if diff.any():
        max_delta = max(max_delta, float(abs(a[diff] - b[diff]).max()))
        cases |= set(new.loc[diff, "case_id"])
print("old vs new label cells:", cell, "| max|delta|:", max_delta, "| cases:", len(cases))
print("grades old/new distinct:", sorted(old["grade_raw"].dropna().unique()), sorted(new["grade_raw"].dropna().unique()))
print("legacy median column present:", "grade_legacy_integer_median" in new.columns)
print("new rows/cols:", new.shape)
