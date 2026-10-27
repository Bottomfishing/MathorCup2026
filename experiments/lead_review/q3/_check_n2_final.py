import json
import pandas as pd
from pathlib import Path

out = Path("outputs/q3_migration_20261028")
new = pd.read_csv(out / "s2c_sensory_before_after_52.csv")
row = new[(new["case_id"] == "S209_TO_Tripel") & (new["role"] == "after") &
          (new["target_52"] == "F_malt_all")]
print("audit example S209_TO_Tripel after F_malt_all ->", row[["grade_raw", "prob_ge5", "prob_6"]].to_dict("records"))
v = json.loads((out / "verification.json").read_text(encoding="utf-8"))
print("checks:", len(v["checks"]), "failed:", [c["name"] for c in v["checks"] if not c["passed"]])
print("replay check names:", [c["name"] for c in v["checks"]
                              if c["name"].startswith("sensory_replay") or
                              c["name"].startswith("sensory_decision")])
print("counts:", v["counts"])
print("claim:", v["claim_level_counts"], "| style:", v["style_diagnostic_counts"])
print("defect:", v["defect_status_counts"])
s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
print("summary replay:", {k: s["sensory_replay"][k] for k in ("rows", "mismatch_cells", "max_abs_delta")})
