"""Probe the official Q1 entry schema for the N2 fix (task-16)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from beer_models.inference import run  # noqa: E402

model = ROOT / "outputs" / "current_models" / "q1.joblib"
bundle = joblib.load(model)
print("bundle grades:", bundle.get("grade_method"), "|", bundle.get("point_semantics"))
decision = bundle.get("q1_decision", {})
print("decision keys:", sorted(decision.keys()))
print("decision rule:", decision.get("rule"), decision.get("legacy_grade_column"))
print("n pairs:", len(decision.get("parent_child_pairs", [])))

spectra = pd.read_csv(ROOT / "outputs" / "q3_migration_20261028" / "s2c_migration_spectra_224_long.csv")
features = [c for c in spectra.columns if c not in
            ("case_id", "source_beer", "target_style", "lambda", "role")]
sample = spectra.head(4)[["case_id", "role", *features]].copy()
sample["beer"] = sample["case_id"] + "|" + sample["role"]
work = ROOT / "experiments" / "lead_review" / "q3" / "tmp_probe_n2"
if work.exists():
    shutil.rmtree(work)
work.mkdir(parents=True)
input_csv = work / "probe.csv"
sample[["beer", *features]].to_csv(input_csv, index=False, encoding="utf-8")
run(model, input_csv, work / "out")
long = pd.read_csv(work / "out" / "scores_all52_long.csv")
print("shape:", long.shape)
print("columns:", list(long.columns))
print(long.head(3)[["beer", "target", "estimated_grade_0_6"]].to_string())
print("grade_legacy col present:", "grade_legacy_integer_median" in long.columns)
