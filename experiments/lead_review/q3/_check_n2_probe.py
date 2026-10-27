import json
from pathlib import Path

out = Path("outputs/q3_migration_20261028")
v = json.loads((out / "verification.json").read_text(encoding="utf-8"))
r = v["sensory_replay"]
print(json.dumps({
    "checks": len(v["checks"]),
    "failed": [c["name"] for c in v["checks"] if not c["passed"]],
    "replay": {k: r[k] for k in ("rows", "columns_compared", "tolerance", "mismatch_cells",
                                 "max_abs_delta", "model_sha256", "delivered_table_sha256",
                                 "replayed_at")},
    "pipeline": v["sensory_pipeline"],
    "files": {p.name: p.stat().st_size for p in sorted(out.iterdir())},
}, ensure_ascii=False, indent=1))
