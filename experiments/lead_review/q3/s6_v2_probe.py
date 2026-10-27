"""Smoke/statistics probe for the v2 donor-residual generator (read-only on outputs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from r2_raw_q1 import PROTOCOL, RAW_MANIFEST, load_inputs  # noqa: E402

from beer_models.generation import StyleGenerator  # noqa: E402
from beer_models.generation_copula import DonorResidualGenerator  # noqa: E402

SOURCE = Path(sys.argv[1])
d = load_inputs(SOURCE, RAW_MANIFEST.parent / "Raw sensory data anonymized.csv",
                json.loads(PROTOCOL.read_text(encoding="utf-8")))
x, styles, features = d["x"], d["ids"]["tasting_category_fine"].to_numpy(), d["features"]
thresholds = {"DMS": 19.0, "X2.3.butanedione": 0.09, "X4VG": 0.20}
generator = DonorResidualGenerator(features=features, defect_thresholds=thresholds).fit(x, styles)
print("support radius:", {k: round(v["radius"], 4) for k, v in generator.donor_pools_.items()
                          if k in ("Tripel", "Kriek", "Lambic")})
for style in ("Tripel", "Kriek", "Lambic"):
    generated, metadata = generator.sample(style, 100, 20341024)
    print(style, "shape", generated.shape,
          "support_projected", metadata["n_support_projected"],
          "cap_hits", metadata["projection"],
          "unique_rows", len(np.unique(np.round(generated, 10), axis=0)),
          "var_ratio", round(float(np.var(generated, 0).mean() / np.var(x[styles == style], 0).mean()), 4))
