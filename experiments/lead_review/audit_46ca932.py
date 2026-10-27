"""Read-only audit of review claims on commit 46ca932 (issues 3, 4, 7, 9)."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "deliverables" / "current_20261028"


def check_file_index() -> dict:
    rows = list(csv.DictReader((PKG / "file_index.csv").open(encoding="utf-8-sig")))
    cols = list(rows[0].keys())
    bad = []
    for r in rows:
        rel = r["file"]
        p = PKG / rel
        if not p.is_file():
            bad.append((rel, "missing"))
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != r["sha256"]:
            bad.append((rel, "mismatch"))
    return {"columns": cols, "n": len(rows), "bad": len(bad), "examples": bad[:5]}


def check_hierarchy() -> dict:
    src = ROOT / "outputs" / "q1_decision_20261028" / "q1__evaluation_long_12500.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    grades = {}
    fam = {}
    for r in rows:
        grades[(r["beer"], r["target"])] = int(r["grade_0_6"])
        fam[r["target"]] = r["target_family"]
    parent = {}
    for fam in ("A", "F"):
        for grp, kids in {
            "hops_all": ["hops_citrus", "hops_noble", "hops_tropical", "hops_woody"],
            "malt_all": ["malt_bread", "malt_burn", "malt_cara", "malt_grain"],
            "esters_all": ["esters_banana", "esters_apple", "esters_isoamyl", "esters_ethyl"],
        }.items():
            for kid in kids:
                parent[f"{fam}_{kid}"] = f"{fam}_{grp}"
    # derive parent map from scorecard if names differ
    card = ROOT / "data" / "reference" / "zenodo23624915"
    violations = []
    for (beer, target), g in grades.items():
        par = parent.get(target)
        if par and par in grades and (beer, par) in grades:
            if g > grades[(beer, par)]:
                violations.append((beer, target, g, par, grades[(beer, par)]))
    return {"n_violations": len(violations), "examples": violations[:8]}


def check_high_grade_metrics() -> dict:
    src = ROOT / "outputs" / "q1_decision_20261028" / "q1__evaluation_long_12500.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    panel = [int(r["panel_median_reference"]) if r["panel_median_reference"] not in ("", "nan", "NaN") else None for r in rows]
    pred = [int(r["grade_0_6"]) for r in rows]
    out = {}
    for name, fn in {
        "exact5": lambda t, p: t == 5 and p == 5,
        "exact6": lambda t, p: t == 6 and p == 6,
        "ge5": lambda t, p: (t is not None and t >= 5) and p >= 5,
    }.items():
        pred_n = sum(1 for p in pred if p >= 5) if name == "ge5" else sum(1 for p in pred if p == (5 if name == "exact5" else 6))
        tp = sum(1 for t, p in zip(panel, pred) if t is not None and fn(t, p))
        obs = sum(1 for t in panel if t is not None and (t >= 5 if name == "ge5" else t == (5 if name == "exact5" else 6)))
        out[name] = {"pred": pred_n, "tp": tp, "precision": round(tp / pred_n, 4) if pred_n else None,
                     "observed": obs, "recall": round(tp / obs, 4) if obs else None}
    return out


def check_verify_paths() -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    from beer_models import delivery_current  # noqa: E402

    out = {}
    try:
        delivery_current.verify(PKG, models=ROOT / "no_such_models", source=ROOT / "no_such_source")
        out["bogus_paths"] = "passed (BAD)"
    except Exception as exc:  # noqa: BLE001
        out["bogus_paths"] = f"failed: {type(exc).__name__}: {exc}"
    return out


if __name__ == "__main__":
    result = {
        "file_index": check_file_index(),
        "hierarchy": check_hierarchy(),
        "high_grade": check_high_grade_metrics(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
