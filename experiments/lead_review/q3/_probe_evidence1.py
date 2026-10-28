import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from beer_models import q3_migration as qm  # noqa: E402

SOURCE = Path(r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件")
b = joblib.load("outputs/current_models/q3.joblib")
d = qm.load(SOURCE)
print("bundle features == official minus diacetyl:",
      list(b["features"]) == [f for f in d.official_features if f != "X2.3.butanedione"])
print("bundle n features:", len(b["features"]), "| official:", len(d.official_features))
print("keep_indices sorted/identity?:", np.array_equal(np.sort(b["keep_indices"]), np.arange(len(b["features"]))))
print("config:", json.loads(Path("configs/q3_style_protocol.json").read_text(encoding="utf-8")))
print("styles:", {s: int((d.styles == s).sum()) for s in ("Tripel", "Kriek", "Lambic", "Pils/Lager")})
print("thresholds:", d.thresholds)
trip = d.styles == "Tripel"
print("tripel n:", int(trip.sum()))
print("tripel DMS physical:", np.round([d.defect_concentration(d.standardised[i], "DMS")
                                       for i in np.flatnonzero(trip)], 3).tolist())
print("tripel diacetyl:", np.round([d.defect_concentration(d.standardised[i], "Diacetyl")
                                    for i in np.flatnonzero(trip)], 4).tolist())
print("tripel 4VG:", np.round([d.defect_concentration(d.standardised[i], "4VG")
                               for i in np.flatnonzero(trip)], 4).tolist())
print("defect caps median:", {s.name: d.physical_cap(s.name, "median") for s in qm.DEFECT_SPECS})
print("defect caps mean:", {s.name: d.physical_cap(s.name, "mean") for s in qm.DEFECT_SPECS})
