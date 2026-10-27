import hashlib
import json
from pathlib import Path

out = Path("outputs/q3_migration_20261028")
v = json.loads((out / "verification.json").read_text(encoding="utf-8"))
print("status:", v["status"], "| checks:", len(v["checks"]),
      "| failed:", [c["name"] for c in v["checks"] if not c["passed"]])
r = v["sensory_replay"]
for key in ("table", "rows", "columns_compared", "tolerance", "mismatch_cells",
            "max_abs_delta", "model_sha256", "delivered_table_sha256", "replayed_at"):
    print(key + ":", r[key])
print("mismatch cols:", r["mismatches_per_column"])
print("protocol model sha:", v["sensory_pipeline"]["q1_model_sha256"])
print("table sha now:", hashlib.sha256((out / "s2c_sensory_before_after_52.csv").read_bytes()).hexdigest())
s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
print("summary replay:", s["sensory_replay"]["mismatch_cells"], s["sensory_replay"]["max_abs_delta"])
print("counts:", {k: s["counts"][k] for k in ("sensory", "spectra_long", "spectra_wide", "evidence")})
