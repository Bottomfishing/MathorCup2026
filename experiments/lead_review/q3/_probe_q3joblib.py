import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, "src")

b = joblib.load("outputs/current_models/q3.joblib")
print("keys:", list(b))
print("panel type:", type(b["panel"]))
print("panel keys:" , list(b["panel"]) if isinstance(b["panel"], dict) else dir(b["panel"])[:10])
clf = b["classifiers"]
print("classifiers type:", type(clf))
if isinstance(clf, dict):
    for k, v in clf.items():
        print("  ", k, type(v))
        if hasattr(v, "classes_"):
            print("     classes:", list(v.classes_)[:30], "n=", len(v.classes_))
        if hasattr(v, "get_params"):
            print("     params:", {kk: vv for kk, vv in v.get_params().items()
                                   if kk in ("n_estimators", "max_depth", "C", "n_neighbors",
                                             "algorithm", "random_state")})
print("keep_indices:", len(b["keep_indices"]), type(b["keep_indices"]))
print("features:", len(b["features"]))
print("omitted_input:", b["omitted_input"])
