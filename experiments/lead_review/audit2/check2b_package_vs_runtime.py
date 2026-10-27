"""Independent cross-check: delivered 'official' full-fit table vs current.py predict.

Finding under test: the package table
``deliverables/current_20261028/tables/current__official_Q1_fullfit_scores52.csv``
is described as the official Q1 scores, but the runtime ``predict`` path applies a
different decision rule.  This script quantifies the disagreement and shows which
column of the runtime long table the package table actually equals.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

AUDIT = ROOT / "experiments" / "lead_review" / "audit2"
PACKAGE = ROOT / "deliverables" / "current_20261028" / "tables" / "current__official_Q1_fullfit_scores52.csv"
LONG = AUDIT / "runtime_predict" / "scores_all52_long.csv"


def _distribution(series: pd.Series) -> dict:
    return {str(int(k)): int(v) for k, v in Counter(series.tolist()).items()}


def main() -> int:
    package = pd.read_csv(PACKAGE)
    runtime = pd.read_csv(LONG)
    assert package.shape == (13000, 19), package.shape
    assert runtime.shape[0] == 13000, runtime.shape
    labelled = runtime.loc[runtime["panel_target_available"].astype(bool), "target"].unique()
    labelled = [t for t in labelled if t in set(package["target"])]
    pkg = package[package["target"].isin(labelled)][["beer", "target", "estimated_grade_0_6"]]
    run = runtime[runtime["target"].isin(labelled)][
        ["beer", "target", "estimated_grade_0_6", "grade_legacy_integer_median"]
    ]
    m = pkg.merge(run, on=["beer", "target"], suffixes=("_package", "_runtime"), validate="one_to_one")
    cells = len(m)
    diff = (m["estimated_grade_0_6_package"] - m["estimated_grade_0_6_runtime"]).abs()
    legacy = (m["estimated_grade_0_6_package"] - m["grade_legacy_integer_median"]).abs()
    changed = int((diff > 1e-12).sum())
    legacy_changed = int((legacy > 1e-12).sum())
    per_target = (m.assign(d=diff)[m.assign(d=diff)["d"] > 1e-12]
                  .groupby("target").size().sort_values(ascending=False))
    payload = {
        "package_table": str(PACKAGE.relative_to(ROOT)),
        "runtime_predict": str(LONG.relative_to(ROOT)),
        "package_shape": list(package.shape),
        "labelled_cells": cells,
        "package_vs_runtime_decision_changed_cells": changed,
        "package_vs_runtime_decision_pct": round(100.0 * changed / cells, 4),
        "package_vs_runtime_decision_max_abs_diff": float(diff.max()),
        "package_vs_runtime_legacy_changed_cells": legacy_changed,
        "package_vs_runtime_legacy_max_abs_diff": float(legacy.max()),
        "package_distribution": _distribution(m["estimated_grade_0_6_package"]),
        "runtime_decision_distribution": _distribution(m["estimated_grade_0_6_runtime"]),
        "runtime_legacy_distribution": _distribution(m["grade_legacy_integer_median"]),
        "targets_with_any_change": {k: int(v) for k, v in per_target.head(10).items()},
        "reading": (
            "the delivered 'official' full-fit table equals the runtime *legacy* integer-median "
            "column, not the decision-applied estimated_grade_0_6; the decision layer is live in "
            "predict but the delivered table was rendered from portrait() and keeps the legacy "
            "rule. verify's replay compares portrait output with exactly this table, so the "
            "mismatch is invisible to verify."
        ),
    }
    out = AUDIT / "check2b_package_vs_runtime.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "labelled_cells", "package_vs_runtime_decision_changed_cells",
        "package_vs_runtime_legacy_changed_cells", "package_distribution",
        "runtime_decision_distribution", "runtime_legacy_distribution")}, ensure_ascii=False))
    print(f"written {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
