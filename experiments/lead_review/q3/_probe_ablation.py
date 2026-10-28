import pandas as pd
from pathlib import Path

for name in ("q3_copula_20261028", "q3_copula_v2_20261028", "q3_copula_v3_20261028"):
    d = Path("outputs") / name
    print("=" * 20, name, sorted(p.name for p in d.glob("*.csv")))
    v = pd.read_csv(d / "generation_validation.csv")
    print("validation:", v.shape, "| methods:", v["method"].unique().tolist())
    print("rows per method:", v["method"].value_counts().to_dict())
    print("cols:", [c for c in v.columns if c in
                    ("outer_fold", "style", "method", "seed", "marginal_w1", "energy",
                     "variance_ratio", "rank_correlation_rmse", "heldout_coverage",
                     "exact_training_copy_rate", "internal_near_duplicate_rate",
                     "model_median_w1")])
    for gate_name in ("generation_gate.csv", "generation_gate_comparison.csv"):
        p = d / gate_name
        if p.exists():
            g = pd.read_csv(p)
            print(gate_name, g.shape, g["rule"].tolist())
