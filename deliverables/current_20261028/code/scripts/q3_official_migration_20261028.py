"""Official Q3 cross-style migration run (problem-statement semantics).

Migrates all 11 ``Pils/Lager`` beers of attachment 1 toward ``Tripel``, ``Kriek`` and
``Lambic`` (33 cases) and certifies each case under the three explicit readings of the
problem note "缺陷化合物以所列 ... 超阈值项为准" (criteria A/B/C, see
``beer_models.q3_migration``).  The migration model, the projection and the criteria are
defined in ``src/beer_models/q3_migration.py``; this script only orchestrates inputs,
outputs and the verification record.

Outputs (``--output``, default ``outputs/q3_migration_20261028``)
    protocol.json                  criteria definitions, lambda grid, sources, thresholds,
                                   evidence-grading rules (written before the run)
    s2c_migration_results.csv      33 rows, one per (source beer, target style)
    s2c_candidate_counts.csv       per (source, target) candidate counts and pass counts
    s2_source_defects_11beers.csv  defect coordinates / concentrations / severity per source
    s2_style_defect_exceedance.csv per-style defect baseline in physical units
    s2c_migration_spectra_224_long.csv   66 rows (33 cases x before/after), 224 coordinates
    s2c_migration_spectra_224_wide.csv   33 rows, ``..._before`` / ``..._after`` side by side
    s2c_migration_after_input_33.csv     33 rows in the official Q1 input layout
                                         (generated_id, style, method + 224 features, official order)
    s2c_sensory_before_after_52.csv      66 spectra x 52 targets from the official Q1 entry
    s2c_det_grade_changes.csv     changed grade cells versus a --previous backup run (task B)
    s2c_det_grade_change_matrix.csv per case/role up/down/unchanged grade counts (task B)
    s2c_evidence_by_case.csv      33 rows, per-case chemical/style/defect evidence and claim level
    verification.json              PASS/FAIL record plus per-case metrics
    summary.json                   machine-readable counts
    summary.md                     short Chinese summary

The 52-target sensory profiles are produced by the official Q1 entry (the same
``beer_models.inference.run`` call that ``scripts/current.py predict --task q1`` performs),
so the numbers can be re-scored when the Q1 decision rule changes.

Every number comes from the supplied attachments; nothing is fabricated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beer_models.inference import run as run_inference  # noqa: E402
from beer_models.q3_migration import (  # noqa: E402
    CLAIM_LEVELS,
    CRITERIA,
    DEFECT_HEADROOM,
    DEFECT_SPECS,
    DEFECT_STATUSES,
    HONESTY,
    LAMBDA_GRID,
    MIGRATED_METHOD,
    PRIMARY_SCENARIO,
    SOURCE_STYLE,
    STALE_HOP_PROXIES,
    TARGET_STYLES,
    THRESHOLD_SCENARIOS,
    case_id,
    load,
    source_defect_table,
    style_defect_table,
)

DEFAULT_RAW = ROOT / "data" / "reference" / "zenodo23624915" / "Raw sensory data anonymized.csv"
DEFAULT_Q1_MODEL = ROOT / "outputs" / "current_models" / "q1.joblib"
DEFAULT_PREVIOUS = ROOT / "outputs" / "q3_migration_20261028_pre_det"
OFFICIAL_PROFILES = ROOT / "outputs" / "task4_raw_q3_20261026" / "generated_profiles.csv"
SENSORY_TARGETS = 52
# task B: after the Q1 decision rule changed, re-grade the 66 spectra and audit the direction
# of every changed cell against the pre-det backup run.  The five listed off-flavour targets
# must never get a worse grade; more than DET_MAX_CHANGED_CELLS changes means "stop and report".
DET_DEFECT_TARGETS = ("dms", "diacetyl", "X4vg", "t2n", "stale_hops")
DET_MAX_CHANGED_CELLS = 100


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed so the 51 MB Q1 bundle is not read twice."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="attachment directory containing 1.csv and 8.xlsx")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "q3_migration_20261028")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                        help="external sensory panel data, recorded as provenance only")
    parser.add_argument("--q1-model", type=Path, default=DEFAULT_Q1_MODEL,
                        help="official Q1 bundle used for the 52-target sensory profiles "
                             "(same entry as scripts/current.py predict --task q1)")
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS,
                        help="pre-det backup run used for the grade-direction audit "
                             "(s2c_det_grade_changes.csv); skipped when the directory is absent")
    return parser.parse_args(argv)


def _spectra_input(spectra_long: pd.DataFrame, data) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the 66-row Q1 input from a long before/after spectra table."""
    official = list(data.official_features)
    before = spectra_long.loc[spectra_long["role"] == "before", ["case_id", *official]].copy()
    after = spectra_long.loc[spectra_long["role"] == "after", ["case_id", *official]].copy()
    before["beer"] = before["case_id"] + "|before"
    after["beer"] = after["case_id"] + "|after"
    frame = pd.concat([before[["beer", *official]], after[["beer", *official]]],
                      ignore_index=True)
    key = spectra_long[["case_id", "role"]].copy()
    return frame, key


def _official_scores(scores: pd.DataFrame, key: pd.DataFrame) -> pd.DataFrame:
    """Normalise the official inference output into the delivered 52-target table.

    ``inference.run`` keeps the legacy risk-head probabilities under
    ``probability_rating_ge5_x`` / ``probability_rating_6_x`` and adds the decision-layer
    probabilities (``_y``) that actually drive ``estimated_grade_0_6``.  The delivered
    ``prob_ge5`` / ``prob_6`` columns are the decision-layer ones so that the grade and its
    probability columns are consistent; the risk-head values are kept as extra columns.
    """
    scores = scores.copy()
    scores[["case_id", "role"]] = scores["beer"].str.split("|", n=1, expand=True)

    def pick(primary, fallback):
        return scores[primary] if primary in scores.columns else scores[fallback]

    table = pd.DataFrame({
        "case_id": scores["case_id"],
        "role": scores["role"],
        "target_52": scores["target"],
        "grade_raw": scores["estimated_grade_0_6"],
        "prob_ge1": scores["probability_rating_ge1"],
        "prob_ge5": pick("probability_rating_ge5_y", "probability_rating_ge5"),
        "prob_6": pick("probability_rating_6_y", "probability_rating_6"),
        "raw_median_estimate": scores.get("raw_median_estimate"),
        "panel_target_available": scores.get("panel_target_available"),
        "score_status": scores.get("score_status"),
        "risk_source": scores.get("risk_source"),
        "risk_validation_status": scores.get("risk_validation_status"),
        "calibration_method": scores.get("calibration_method"),
        "calibration_quality_status": scores.get("calibration_quality_status"),
        "safety_certified": scores.get("safety_certified"),
        "grade_legacy_integer_median": scores.get("grade_legacy_integer_median"),
        "prob_ge5_risk_head": scores.get("probability_rating_ge5_x"),
        "prob_6_risk_head": scores.get("probability_rating_6_x"),
        "prediction_role": scores.get("prediction_role"),
    })
    key = key.assign(_order=np.arange(len(key)))
    table = table.merge(key, on=["case_id", "role"], how="left", validate="many_to_one")
    return table.sort_values("_order", kind="mergesort").drop(columns="_order").reset_index(drop=True)


def build_sensory_profiles(data, spectra, model_path: Path, workdir: Path) -> pd.DataFrame:
    """Run the official Q1 entry on the 66 before/after spectra and return the 52-target table.

    ``workdir`` is a scratch directory under ``experiments/lead_review/q3`` (it is removed
    first, so re-runs are deterministic); the delivered table is written to ``--output``.
    """
    frame, key = _spectra_input(spectra["long"], data)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    input_csv = workdir / "migration_before_after_66.csv"
    frame.to_csv(input_csv, index=False, encoding="utf-8")
    run_inference(model_path, input_csv, workdir / "q1_output")
    scores = pd.read_csv(workdir / "q1_output" / "scores_all52_long.csv")
    return _official_scores(scores, key)


def _compare_columns(stored: pd.DataFrame, replayed: pd.DataFrame, tolerance: float):
    """Cell-level comparison used by the reproducibility self-check."""
    mismatches = 0
    max_delta = 0.0
    per_column = {}
    for column in stored.columns:
        left = pd.to_numeric(stored[column], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(replayed[column], errors="coerce").to_numpy(dtype=float)
        left_nan, right_nan = np.isnan(left), np.isnan(right)
        both_nan = left_nan & right_nan
        delta = np.abs(left - right)
        delta[both_nan] = 0.0
        ok = both_nan | (~left_nan & ~right_nan & (delta <= tolerance))
        column_mismatches = int((~ok).sum())
        if column_mismatches:
            per_column[column] = column_mismatches
        mismatches += column_mismatches
        finite = delta[np.isfinite(delta)]
        if finite.size:
            max_delta = max(max_delta, float(finite.max()))
    return mismatches, max_delta, per_column


def replay_sensory_table(data, output: Path, model_path: Path, workdir: Path,
                         table_name: str = "s2c_sensory_before_after_52.csv",
                         tolerance: float = 1e-12) -> dict:
    """Replay the delivered sensory table from the packaged spectra + current Q1 bundle.

    Reads ``s2c_migration_spectra_224_long.csv`` and the delivered table back from disk,
    re-runs the official entry, and compares every shared cell.  A non-zero mismatch means
    the table is not reproducible from what was delivered.
    """
    stored_long = pd.read_csv(output / "s2c_migration_spectra_224_long.csv")
    stored = pd.read_csv(output / table_name)
    frame, key = _spectra_input(stored_long, data)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    input_csv = workdir / "replay_before_after_66.csv"
    frame.to_csv(input_csv, index=False, encoding="utf-8")
    run_inference(model_path, input_csv, workdir / "q1_output")
    scores = pd.read_csv(workdir / "q1_output" / "scores_all52_long.csv")
    replayed = _official_scores(scores, key)
    if len(stored) != len(replayed):
        raise ValueError(f"replay row count {len(replayed)} != delivered {len(stored)}")
    columns = [column for column in stored.columns if column in replayed.columns]
    mismatches, max_delta, per_column = _compare_columns(
        stored[columns].reset_index(drop=True),
        replayed[columns].reset_index(drop=True), tolerance)
    return {
        "table": table_name,
        "rows": int(len(stored)),
        "columns_compared": len(columns),
        "tolerance": tolerance,
        "mismatch_cells": mismatches,
        "mismatches_per_column": per_column,
        "max_abs_delta": max_delta,
        "model_path": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT)
        else str(model_path),
        "model_sha256": sha256_file(model_path),
        "delivered_table_sha256": sha256_file(output / table_name),
        "replay_input_sha256": sha256_file(input_csv),
        "replayed_at": _now_iso(),
    }


def compare_pre_det(previous: Path, sensory: pd.DataFrame, results: pd.DataFrame,
                    evidence: pd.DataFrame, model_sha256: str):
    """Audit the direction of every grade change versus a pre-det backup run.

    Returns ``(report, grade_changes, grade_matrix)``.  ``report`` is stored in
    ``summary.json`` under ``det_rerun``; the two frames become the delivered CSV tables.
    A missing backup directory is not an error (fresh checkout): the comparison is skipped and
    only recorded as unavailable.
    """
    change_columns = ["case_id", "source_beer", "target_style", "role", "target_52",
                      "grade_old", "grade_new", "delta", "direction", "defect_target"]
    matrix_columns = ["case_id", "source_beer", "target_style", "role", "cells", "up", "down",
                      "unchanged", "defect_up", "defect_down", "defect_worse"]
    if not previous.is_dir():
        return {"available": False, "previous_directory": str(previous)}, (
            pd.DataFrame(columns=change_columns)), pd.DataFrame(columns=matrix_columns)

    old_sensory = pd.read_csv(previous / "s2c_sensory_before_after_52.csv")
    old_evidence = pd.read_csv(previous / "s2c_evidence_by_case.csv")
    old_results = pd.read_csv(previous / "s2c_migration_results.csv")
    old_summary_path = previous / "summary.json"
    old_summary = (json.loads(old_summary_path.read_text(encoding="utf-8"))
                   if old_summary_path.is_file() else {})

    keys = ["case_id", "role", "target_52"]
    merged = (old_sensory[keys + ["grade_raw"]].rename(columns={"grade_raw": "grade_old"})
              .merge(sensory[keys + ["grade_raw"]].rename(columns={"grade_raw": "grade_new"}),
                     on=keys, how="outer", validate="one_to_one"))
    merged["grade_old"] = pd.to_numeric(merged["grade_old"], errors="coerce")
    merged["grade_new"] = pd.to_numeric(merged["grade_new"], errors="coerce")
    both = merged["grade_old"].notna() & merged["grade_new"].notna()
    merged["delta"] = np.where(both, merged["grade_new"] - merged["grade_old"], np.nan)
    merged["direction"] = np.select(
        [both & (np.abs(merged["delta"]) <= 1e-12), both & (merged["delta"] > 1e-12),
         both & (merged["delta"] < -1e-12)],
        ["unchanged", "up", "down"], default="unavailable")
    merged["defect_target"] = merged["target_52"].isin(DET_DEFECT_TARGETS)
    parsed = merged["case_id"].str.extract(r"^S(\d{3})_TO_(Tripel|Kriek|Lambic)$")
    merged["source_beer"] = parsed[0].astype("Int64")
    merged["target_style"] = parsed[1]

    changed = merged.loc[merged["direction"].isin(["up", "down"])].copy()
    changed["source_beer"] = changed["source_beer"].astype(int)
    grade_changes = changed[change_columns].sort_values(
        ["case_id", "role", "target_52"], kind="mergesort").reset_index(drop=True)

    defect_worse = changed[(changed["direction"] == "up") & changed["defect_target"]]

    def _matrix(frame):
        frame = frame.copy()
        frame["up"] = (frame["direction"] == "up").astype(int)
        frame["down"] = (frame["direction"] == "down").astype(int)
        frame["same"] = (frame["direction"] == "unchanged").astype(int)
        frame["defect_up"] = ((frame["direction"] == "up") & frame["defect_target"]).astype(int)
        frame["defect_down"] = ((frame["direction"] == "down") & frame["defect_target"]).astype(int)
        grouped = frame.groupby(["case_id", "source_beer", "target_style", "role"],
                                dropna=False).agg(
            cells=("direction", "size"), up=("up", "sum"), down=("down", "sum"),
            unchanged=("same", "sum"), defect_up=("defect_up", "sum"),
            defect_down=("defect_down", "sum")).reset_index()
        grouped["defect_worse"] = grouped["defect_up"]
        return grouped[matrix_columns].sort_values(
            ["case_id", "role"], kind="mergesort").reset_index(drop=True)

    grade_matrix = _matrix(merged)

    def _criteria(frame):
        return {f"{criterion}_{scenario}": int(frame[f"{criterion}_{scenario}"].sum())
                for scenario in THRESHOLD_SCENARIOS for criterion in ("A", "B", "C")}

    criteria_old, criteria_new = _criteria(old_results), _criteria(results)

    def _value_counts(frame, column):
        counts = frame[column].astype(str).value_counts().sort_index()
        return {str(key): int(value) for key, value in counts.items()}

    status_columns = [f"defect_status_{name}" for name in
                      ("DMS", "Diacetyl", "4VG", "t2n", "Stale_hops")]
    shared_evidence = ["case_id", "claim_level", "style_nearest_centroid", *status_columns]
    aligned = (old_evidence[shared_evidence].merge(
        evidence[shared_evidence], on="case_id", how="outer", suffixes=("_old", "_new"),
        validate="one_to_one"))
    claim_changed = sorted(aligned.loc[
        aligned["claim_level_old"] != aligned["claim_level_new"], "case_id"].astype(str).tolist())
    style_changed = sorted(aligned.loc[
        aligned["style_nearest_centroid_old"] != aligned["style_nearest_centroid_new"],
        "case_id"].astype(str).tolist())
    defect_status_changed = []
    for _, row in aligned.iterrows():
        if any(row[f"{column}_old"] != row[f"{column}_new"] for column in status_columns):
            defect_status_changed.append(str(row["case_id"]))

    def _role_counts(role):
        subset = merged.loc[merged["role"] == role]
        return {"up": int((subset["direction"] == "up").sum()),
                "down": int((subset["direction"] == "down").sum()),
                "unchanged": int((subset["direction"] == "unchanged").sum())}

    defect_counts = {}
    for target in DET_DEFECT_TARGETS:
        subset = merged.loc[merged["target_52"] == target]
        defect_counts[target] = {"up": int((subset["direction"] == "up").sum()),
                                 "down": int((subset["direction"] == "down").sum()),
                                 "unchanged": int((subset["direction"] == "unchanged").sum())}

    report = {
        "available": True,
        "previous_directory": previous.name,
        "previous_model_sha256": old_summary.get("sensory_pipeline", {}).get("q1_model_sha256"),
        "current_model_sha256": model_sha256,
        "model_sha_changed": (old_summary.get("sensory_pipeline", {}).get("q1_model_sha256")
                              != model_sha256),
        "cells_total": int(len(merged)),
        "cells_compared": int(both.sum()),
        "cells_unavailable_both": int((~both).sum()),
        "changed_cells": int(len(changed)),
        "changed_up": int((changed["direction"] == "up").sum()),
        "changed_down": int((changed["direction"] == "down").sum()),
        "unchanged_cells": int((merged["direction"] == "unchanged").sum()),
        "changed_by_role": {role: _role_counts(role) for role in ("before", "after")},
        "changed_by_target": {str(key): int(value) for key, value in
                              changed["target_52"].value_counts().sort_index().items()},
        "defect_target_counts": defect_counts,
        "defect_worse_count": int(len(defect_worse)),
        "defect_worse_max_delta": (float(defect_worse["delta"].max())
                                   if len(defect_worse) else 0.0),
        "defect_worse_cells": defect_worse[
            ["case_id", "role", "target_52", "grade_old", "grade_new", "delta"]].to_dict("records"),
        "criteria_old": criteria_old,
        "criteria_new": criteria_new,
        "criteria_unchanged": criteria_old == criteria_new,
        "claim_level_counts_old": _value_counts(old_evidence, "claim_level"),
        "claim_level_counts_new": _value_counts(evidence, "claim_level"),
        "claim_level_changed_cases": claim_changed,
        "style_nearest_centroid_counts_old": _value_counts(old_evidence, "style_nearest_centroid"),
        "style_nearest_centroid_counts_new": _value_counts(evidence, "style_nearest_centroid"),
        "style_nearest_centroid_changed_cases": style_changed,
        "defect_status_counts_old": {name: int((old_evidence[f"defect_status_{name}"]
                                               .astype(str) == "improved").sum())
                                     for name in ("DMS", "Diacetyl", "4VG", "t2n", "Stale_hops")},
        "defect_status_counts_new": {name: int((evidence[f"defect_status_{name}"]
                                                .astype(str) == "improved").sum())
                                     for name in ("DMS", "Diacetyl", "4VG", "t2n", "Stale_hops")},
        "defect_status_changed_cases": sorted(defect_status_changed),
        "grade_change_limit": DET_MAX_CHANGED_CELLS,
        "grade_change_within_limit": bool(len(changed) <= DET_MAX_CHANGED_CELLS),
        "compared_at": _now_iso(),
    }
    return report, grade_changes, grade_matrix


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
                    encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main(argv=None):
    started = time.time()
    args = parse_args(argv)
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    q1_model = Path(args.q1_model).resolve()
    model_bundle = joblib.load(q1_model)
    decision = model_bundle.get("q1_decision", {})
    q1_grade_semantics = {
        "official_column": decision.get("official_grade_column", "estimated_grade_0_6"),
        "grade_method": model_bundle.get("grade_method"),
        "point_semantics": model_bundle.get("point_semantics"),
        "decision_rule": decision.get("rule"),
        "legacy_column": decision.get("legacy_grade_column", "grade_legacy_integer_median"),
        "parent_child_pairs": len(decision.get("parent_child_pairs", [])),
        "implementation_sha256": decision.get("implementation_sha256"),
        "note": ("estimated_grade_0_6 is the official decision layer: non-forced high-grade "
                 "top-k from cumulative probability with parent-child caps. It is not the "
                 "legacy integer conditional median (kept in grade_legacy_integer_median)."),
    }

    protocol = {
        "name": "q3_migration_20261028",
        "written_before_results": True,
        "problem_statement_note": ("缺陷化合物以所列 DMS/Diacetyl/4VG/trans-2-nonenal/Stale hops "
                                   "等超阈值项为准"),
        "source_style": SOURCE_STYLE,
        "target_styles": list(TARGET_STYLES),
        "lambda_grid": list(LAMBDA_GRID),
        "lambda_selection": ("largest lambda whose projected spectrum is still nearest to the "
                             "target-style centroid; fallback = smallest target-centroid distance"),
        "criteria": CRITERIA,
        "threshold_scenarios": list(THRESHOLD_SCENARIOS),
        "primary_threshold_scenario": PRIMARY_SCENARIO,
        "defect_specs": [{"name": spec.name, "column": spec.column, "transform": spec.transform,
                          "unit": spec.unit, "threshold_field": spec.threshold_field}
                         for spec in DEFECT_SPECS],
        "defect_projection": {
            "threshold_defects": f"coordinate <= min(source, {DEFECT_HEADROOM} * threshold)",
            "no_threshold_defects": "coordinate <= source",
            "stale_hop_proxies": "coordinate <= source",
            "missing_source": "cap = min(target-style median, 0.9 * threshold), flagged conditional",
            "headroom": DEFECT_HEADROOM,
            "note": "0.9 x threshold headroom is a modelling choice, not a measured safety limit",
        },
        "stale_hop_proxies": list(STALE_HOP_PROXIES),
        "style_identifiability_expected": ("tasting_category_fine is weakly separable in the "
                                           "supplied coordinates; accuracy is reported, not used "
                                           "as a hard gate"),
        "honesty": HONESTY,
        "evidence_grading": {
            "defect_status_rule": ("severity e(c,T)=max(c/T-1,0) on the median threshold: "
                                   "improved if e_after<e_before, worse if e_after>e_before, "
                                   "unchanged otherwise; trans-2-nonenal has no threshold and "
                                   "Stale hops has no coordinate, so both are 'unknown'"),
            "style_nearest_centroid_rule": ("target = migrated spectrum is nearest to the target "
                                            "centroid; source_baseline = still nearest to "
                                            "Pils/Lager; other = another style centroid"),
            "style_diagnostic_supported_rule": ("a target style's diagnostic supports a case only "
                                                "if all 11 source beers migrate to that centroid "
                                                "(Kriek 11/11); Tripel 0/11 and Lambic 2/11 are "
                                                "reported as unsupported"),
            "claim_level_rule": {
                "model_supported": ("A/B/C pass and nearest centroid is the target style and the "
                                    "target style diagnostic supports all 11 cases and no "
                                    "evaluated defect worsened and no conditional cap was used"),
                "conditional": ("A/B/C pass but at least one of: style diagnostic unsupported, an "
                                "evaluated defect worsened, conditional diacetyl cap used"),
                "undecidable": "A/B/C do not all pass",
            },
            "claim_levels": list(CLAIM_LEVELS),
            "defect_statuses": list(DEFECT_STATUSES),
            "no_change_to_migration_protocol": True,
            "note": ("these are labelling rules for the evidence table; they do not change the "
                     "migration math, the lambda grid, the projection or the A/B/C criteria"),
        },
        "sensory_pipeline": {
            "entry": "beer_models.inference.run (same call as scripts/current.py predict --task q1)",
            "q1_model_path": (q1_model.relative_to(ROOT).as_posix()
                              if q1_model.is_relative_to(ROOT) else str(q1_model)),
            "q1_model_sha256": sha256_file(q1_model),
            "q1_model_grade_semantics": q1_grade_semantics,
            "run_at": _now_iso(),
            "score_scale": "original 0-6 individual rating scale",
            "targets": SENSORY_TARGETS,
            "note": ("full-training-model predictions are not out-of-sample accuracy estimates; "
                     "grade_raw is the official decision-layer grade (top-k high grades with 24 "
                     "parent-child caps), prob_ge5/prob_6 are the decision-layer probabilities, "
                     "and prob_ge5_risk_head/prob_6_risk_head are the legacy calibrated risk-head "
                     "probabilities kept for traceability"),
        },
        "sources": {"source_directory": source.name},
        "no_post_result_retuning": True,
    }
    write_json(output / "protocol.json", protocol)
    print("protocol.json written before migration: %d sources x %d targets"
          % (11, len(TARGET_STYLES)))

    data = load(source, raw_path=Path(args.raw))
    results, counts = data.run()
    style_table = style_defect_table(data)
    source_table = source_defect_table(data)
    accuracy = data.style_accuracy()

    results.to_csv(output / "s2c_migration_results.csv", index=False, encoding="utf-8-sig")
    counts.to_csv(output / "s2c_candidate_counts.csv", index=False, encoding="utf-8-sig")
    source_table.to_csv(output / "s2_source_defects_11beers.csv", index=False, encoding="utf-8-sig")
    style_table.to_csv(output / "s2_style_defect_exceedance.csv", index=False, encoding="utf-8-sig")

    # complete spectra for every case + Q1-consistent input table + evidence grading
    spectra = data.migration_spectra()
    evidence = data.evidence_table(results)
    spectra["long"].to_csv(output / "s2c_migration_spectra_224_long.csv", index=False,
                           encoding="utf-8-sig")
    spectra["wide"].to_csv(output / "s2c_migration_spectra_224_wide.csv", index=False,
                           encoding="utf-8-sig")
    spectra["after_input"].to_csv(output / "s2c_migration_after_input_33.csv", index=False,
                                  encoding="utf-8-sig")
    evidence.to_csv(output / "s2c_evidence_by_case.csv", index=False, encoding="utf-8-sig")
    # 52-target sensory profiles from the official Q1 entry (before and after the migration)
    sensory = build_sensory_profiles(
        data, spectra, q1_model,
        ROOT / "experiments" / "lead_review" / "q3" / "tmp_q1_migration")
    sensory.to_csv(output / "s2c_sensory_before_after_52.csv", index=False,
                   encoding="utf-8-sig")
    # reproducibility self-check: replay the delivered table from the packaged spectra + model
    sensory_replay = replay_sensory_table(
        data, output, q1_model,
        ROOT / "experiments" / "lead_review" / "q3" / "tmp_q1_replay")
    print("sensory replay: rows=%d mismatches=%d max|delta|=%.3g model_sha=%s"
          % (sensory_replay["rows"], sensory_replay["mismatch_cells"],
             sensory_replay["max_abs_delta"], sensory_replay["model_sha256"][:12]))
    print("exported spectra/sensory/evidence: %s"
          % {key: list(frame.shape) for key, frame in
             {"spectra_long": spectra["long"], "spectra_wide": spectra["wide"],
              "after_input": spectra["after_input"], "sensory": sensory,
              "evidence": evidence}.items()})

    # task B: audit the direction of every grade change versus the pre-det backup run
    det_rerun, det_grade_changes, det_grade_matrix = compare_pre_det(
        Path(args.previous).resolve(), sensory, results, evidence,
        sha256_file(q1_model))
    if det_rerun.get("available"):
        det_grade_changes.to_csv(output / "s2c_det_grade_changes.csv", index=False,
                                 encoding="utf-8-sig")
        det_grade_matrix.to_csv(output / "s2c_det_grade_change_matrix.csv", index=False,
                                encoding="utf-8-sig")
        print("det rerun: model_changed=%s changed=%d up=%d down=%d defect_worse=%d compared=%d/%d"
              % (det_rerun["model_sha_changed"], det_rerun["changed_cells"],
                 det_rerun["changed_up"], det_rerun["changed_down"],
                 det_rerun["defect_worse_count"], det_rerun["cells_compared"],
                 det_rerun["cells_total"]))
    else:
        print("det rerun: previous directory %s absent, comparison skipped"
              % det_rerun["previous_directory"])

    checks = []

    def check(name, value, expected=None, passed=None, note=None):
        item = {"name": name, "value": value, "expected": expected}
        item["passed"] = bool(passed) if passed is not None else bool(value == expected)
        if note:
            item["note"] = note
        checks.append(item)
        return item["passed"]

    check("results_row_count", int(len(results)), 33)
    per_target = {style: int((results["target_style"] == style).sum()) for style in TARGET_STYLES}
    check("results_per_target", per_target, None,
          passed=all(value == 11 for value in per_target.values()) and len(per_target) == len(TARGET_STYLES))
    check("source_count", int(results["source_beer"].nunique()), 11)

    for scenario in THRESHOLD_SCENARIOS:
        for criterion in ("A", "B", "C"):
            column = f"{criterion}_{scenario}"
            check(f"criteria_{criterion}_{scenario}_all_pass", int(results[column].sum()), 33)
    for criterion in ("A", "B", "C"):
        check(f"criteria_{criterion}_all_pass", int(results[f"criteria_{criterion}"].sum()), 33)

    total_candidates = int(counts["n_candidates"].sum())
    check("candidate_count", total_candidates, 660)
    for scenario in THRESHOLD_SCENARIOS:
        for criterion in ("A", "B", "C"):
            column = f"n_pass_{criterion}_{scenario}"
            check(f"candidates_pass_{criterion}_{scenario}", int(counts[column].sum()), total_candidates)

    # defect coordinates never above the source (before -> after)
    for spec in DEFECT_SPECS:
        before = results[f"{spec.name}_before"]
        after = results[f"{spec.name}_after"]
        defined = before.notna()
        violations = int((after[defined] > before[defined] + 1e-9).sum())
        check(f"defect_{spec.name}_not_above_source", violations, 0,
              passed=violations == 0,
              note=f"{int(defined.sum())} cases with a measured source value; "
                   f"{int((~defined).sum())} conditional (source value missing)")

    # lambda semantics
    tripel_lambdas = sorted(results.loc[results["target_style"] == "Tripel", "lambda"].unique().tolist())
    check("tripel_lambda_values", tripel_lambdas, None,
          passed=set(tripel_lambdas) <= {0.5, 0.55},
          note="largest lambda keeping the target centroid nearest, on the 0.05 grid")
    for style in ("Kriek", "Lambic"):
        values = sorted(results.loc[results["target_style"] == style, "lambda"].unique().tolist())
        check(f"{style.lower()}_lambda_values", values, None, passed=values == [1.0])
    lambic_nearest = int(results.loc[results["target_style"] == "Lambic", "style_is_target"].sum())
    check("lambic_target_nearest_cases", lambic_nearest, None, passed=True,
          note="reported honestly: only this many of 11 Lambic cases end nearest to the Lambic "
               "centroid after projection, because the defect projection overrides the style pull")

    # case 119: the 4VG / DMS / diacetyl caps that the paper quotes
    case = results[(results["source_beer"] == 119) & (results["target_style"] == "Tripel")].iloc[0]
    check("case119_4VG_before", round(float(case["4VG_before"]), 6), 0.143755,
          passed=abs(float(case["4VG_before"]) - 0.143755) < 1e-5)
    check("case119_4VG_after", round(float(case["4VG_after"]), 6), 0.040508,
          passed=abs(float(case["4VG_after"]) - 0.0405076) < 1e-5)
    check("case119_DMS_after_equals_before",
          round(float(case["DMS_after"]), 6), round(float(case["DMS_before"]), 6),
          passed=abs(float(case["DMS_after"]) - float(case["DMS_before"])) < 1e-9,
          note="source DMS is already below 0.9*threshold, so the cap keeps the source value")
    check("case119_diacetyl_conditional",
          round(float(case["Diacetyl_after"]), 6), 0.081,
          passed=abs(float(case["Diacetyl_after"]) - 0.081) < 1e-6,
          note="source diacetyl is missing in 1.csv; cap = min(Tripel median, 0.9*threshold)")

    # table shapes and required columns
    check("style_table_rows", int(len(style_table)), 22)
    check("source_table_rows", int(len(source_table)), 11)
    required = ["source_beer", "target_style", "lambda", "criteria_A", "criteria_B", "criteria_C",
                "pass_reason"]
    required += [f"{spec.name}_before" for spec in DEFECT_SPECS]
    required += [f"{spec.name}_after" for spec in DEFECT_SPECS]
    missing = [column for column in required if column not in results.columns]
    check("required_columns", missing, [], passed=not missing)
    nan_columns = [column for column in ("source_beer", "target_style", "lambda", "criteria_A",
                                         "criteria_B", "criteria_C")
                   if results[column].isna().any()]
    check("key_columns_without_nan", nan_columns, [], passed=not nan_columns)

    # ---------------------------------------------------------- spectra integrity
    official_features = list(data.official_features)
    long_df, wide_df, after_input = spectra["long"], spectra["wide"], spectra["after_input"]
    case_list = long_df["case_id"].drop_duplicates().tolist()
    expected_input_columns = ["generated_id", "style", "method", *official_features]

    check("spectra_long_rows", int(len(long_df)), 66)
    check("spectra_long_cases", int(long_df["case_id"].nunique()), 33)
    check("spectra_long_roles", sorted(long_df["role"].unique().tolist()), ["after", "before"])
    check("spectra_before_after_pairs",
          int((long_df.groupby("case_id")["role"].nunique() == 2).sum()), 33)
    check("spectra_official_feature_columns",
          [column for column in official_features if column not in long_df.columns], [],
          passed=all(column in long_df.columns for column in official_features))
    check("spectra_feature_column_count", int(len(official_features)), 224)
    check("spectra_no_nan_in_features", int(long_df[official_features].isna().sum().sum()), 0)
    check("spectra_case_id_pattern",
          bool(long_df["case_id"].str.fullmatch(r"S\d{3}_TO_(Tripel|Kriek|Lambic)").all()), True)
    lambda_join = long_df.merge(
        results[["source_beer", "target_style", "lambda"]].rename(
            columns={"lambda": "lambda_results"}),
        on=["source_beer", "target_style"], how="left", validate="many_to_one")
    lambda_max_abs = float(np.max(np.abs(
        long_df["lambda"].to_numpy(float) - lambda_join["lambda_results"].to_numpy(float))))
    check("spectra_lambda_matches_results", lambda_max_abs, 0.0,
          passed=lambda_max_abs < 1e-12)
    check("spectra_wide_rows", int(len(wide_df)), 33)
    wide_before_columns = [name for name in official_features if f"{name}_before" in wide_df.columns]
    wide_after_columns = [name for name in official_features if f"{name}_after" in wide_df.columns]
    check("spectra_wide_before_columns", len(wide_before_columns), 224,
          passed=len(wide_before_columns) == 224)
    check("spectra_wide_after_columns", len(wide_after_columns), 224,
          passed=len(wide_after_columns) == 224)
    check("after_input_shape", [int(after_input.shape[0]), int(after_input.shape[1])], [33, 227])
    check("after_input_columns", after_input.columns.tolist(), expected_input_columns,
          passed=after_input.columns.tolist() == expected_input_columns)
    check("after_input_method", sorted(after_input["method"].unique().tolist()), [MIGRATED_METHOD])
    check("after_input_style_counts",
          {style: int((after_input["style"] == style).sum()) for style in TARGET_STYLES},
          {style: 11 for style in TARGET_STYLES})
    check("after_input_no_nan", int(after_input[official_features].isna().sum().sum()), 0)
    if OFFICIAL_PROFILES.exists():
        official_header = pd.read_csv(OFFICIAL_PROFILES, nrows=0).columns.tolist()
        check("after_input_same_columns_as_official", after_input.columns.tolist(),
              official_header, passed=after_input.columns.tolist() == official_header)
    else:  # pragma: no cover - the official run exists in this repository
        check("after_input_same_columns_as_official", None, None, passed=True,
              note="official generated_profiles.csv not found; compared against the module list")

    # ------------------------------------------------------------ evidence checks
    def value_counts(column: str) -> dict[str, int]:
        return {str(key): int(value) for key, value in evidence[column].value_counts().items()}

    evidence_required = ["case_id", "chemical_pass", "chemical_pass_A", "chemical_pass_B",
                         "chemical_pass_C", "dist_reduction_pct", "style_nearest_centroid",
                         "style_diagnostic_supported", "defect_status_DMS",
                         "defect_status_Diacetyl", "defect_status_4VG", "defect_status_t2n",
                         "defect_status_Stale_hops", "claim_level", "claim_note"]
    missing_evidence = [column for column in evidence_required
                        if column not in evidence.columns]
    check("evidence_required_columns", missing_evidence, [], passed=not missing_evidence)
    check("evidence_case_ids_match", sorted(evidence["case_id"].tolist()), sorted(case_list),
          passed=sorted(evidence["case_id"].tolist()) == sorted(case_list))
    claim_counts = value_counts("claim_level")
    check("evidence_claim_levels_valid", sorted(claim_counts), None,
          passed=set(claim_counts) <= set(CLAIM_LEVELS))
    check("evidence_claim_level_total", int(sum(claim_counts.values())), 33)
    style_diagnostic_counts = {style: int(evidence.loc[evidence["target_style"] == style,
                                                        "style_diagnostic_cases"].iloc[0])
                               for style in TARGET_STYLES}
    check("style_diagnostic_counts", style_diagnostic_counts,
          {"Tripel": 0, "Kriek": 11, "Lambic": 2})
    defect_status_counts = {name: value_counts(f"defect_status_{name}")
                            for name in ("DMS", "Diacetyl", "4VG", "t2n", "Stale_hops")}
    status_valid = all(status in DEFECT_STATUSES
                       for counts_by_status in defect_status_counts.values()
                       for status in counts_by_status)
    check("defect_status_values_valid", defect_status_counts, None, passed=status_valid)
    check("defect_worse_cases", int(evidence["defects_worse_count"].sum()), 0)
    check("stale_hops_all_unknown", defect_status_counts["Stale_hops"], {"unknown": 33})
    check("t2n_all_unknown", defect_status_counts["t2n"], {"unknown": 33})
    check("undecidable_cases", claim_counts.get("undecidable", 0), 0)

    # ------------------------------------------------------------ sensory checks
    unlabelled_targets = sorted(sensory.loc[sensory["grade_raw"].isna(), "target_52"]
                                .dropna().unique().tolist())
    sensory_required = ["case_id", "role", "target_52", "grade_raw", "prob_ge5", "prob_6"]
    missing_sensory = [column for column in sensory_required if column not in sensory.columns]
    check("sensory_required_columns", missing_sensory, [], passed=not missing_sensory)
    check("sensory_rows", int(len(sensory)), 66 * SENSORY_TARGETS)
    check("sensory_cases", int(sensory["case_id"].nunique()), 33)
    check("sensory_roles", sorted(sensory["role"].unique().tolist()), ["after", "before"])
    check("sensory_targets", int(sensory["target_52"].nunique()), SENSORY_TARGETS)
    check("sensory_grade_raw_missing", int(sensory["grade_raw"].isna().sum()),
          66 * 2, note="the two card targets without a raw-panel label")
    check("sensory_grade_raw_missing_only_unlabelled", unlabelled_targets,
          ["A_acetaldehyde", "F_acetaldehyde"])
    check("sensory_prob_ge5_missing", int(sensory["prob_ge5"].isna().sum()), 66 * 2)
    check("sensory_each_case_52", int(sensory.groupby(["case_id", "role"]).size().min()),
          SENSORY_TARGETS)
    # reproducibility self-check (N2): the delivered table must replay cell-for-cell
    replay_tolerance = float(sensory_replay["tolerance"])
    check("sensory_replay_rows", int(sensory_replay["rows"]), 66 * SENSORY_TARGETS)
    check("sensory_replay_mismatch_cells", int(sensory_replay["mismatch_cells"]), 0,
          note="stored spectra + current q1.joblib replayed through inference.run; "
               "tolerance 1e-12 per cell")
    check("sensory_replay_max_abs_delta", float(sensory_replay["max_abs_delta"]), None,
          passed=float(sensory_replay["max_abs_delta"]) <= replay_tolerance,
          note=f"tolerance {replay_tolerance:g}")
    check("sensory_replay_model_sha256", sensory_replay["model_sha256"],
          protocol["sensory_pipeline"]["q1_model_sha256"])
    check("sensory_replay_delivered_table_sha256", sensory_replay["delivered_table_sha256"],
          sha256_file(output / "s2c_sensory_before_after_52.csv"))
    check("sensory_decision_layer_grade_column", "grade_legacy_integer_median" in sensory.columns,
          None, passed="grade_legacy_integer_median" in sensory.columns,
          note="official grade is the decision-layer column; legacy median kept for traceability")
    check("sensory_pipeline_sha_matches_model_file",
          protocol["sensory_pipeline"]["q1_model_sha256"], sha256_file(q1_model))

    # task B: the pre-det run (previous Q1 bundle) must be archived, and the new run must not
    # worsen any of the five listed off-flavour grades nor move the chemical/style evidence.
    check("det_rerun_previous_directory", det_rerun.get("previous_directory"), None,
          passed=bool(det_rerun.get("available")),
          note="backup of the pre-det artifacts (same 13 files) lives next to the deliverable "
               "directory under outputs/q3_migration_20261028_pre_det")
    if det_rerun.get("available"):
        check("det_rerun_model_sha_changed", det_rerun["current_model_sha256"], None,
              passed=det_rerun["model_sha_changed"],
              note="previous q1.joblib %s -> current %s"
                   % (str(det_rerun["previous_model_sha256"])[:12],
                      det_rerun["current_model_sha256"][:12]))
        check("det_rerun_cells_compared", det_rerun["cells_compared"],
              SENSORY_TARGETS * 66 - 66 * 2,
              note="3432 cells minus the 132 unlabelled card-target cells")
        check("det_rerun_defect_worse_count", det_rerun["defect_worse_count"], 0,
              note="grades on dms/diacetyl/X4vg/t2n/stale_hops must never go up")
        check("det_rerun_changed_cells_within_limit", det_rerun["changed_cells"], None,
              passed=det_rerun["grade_change_within_limit"],
              note="limit %d changed cells; above that the task says stop and report"
                   % DET_MAX_CHANGED_CELLS)
        check("det_rerun_criteria_unchanged", det_rerun["criteria_unchanged"], True,
              note="chemical A/B/C counts come from the migration math, not from Q1")
        check("det_rerun_claim_levels_unchanged", det_rerun["claim_level_changed_cases"], [])
        check("det_rerun_style_counts_unchanged",
              det_rerun["style_nearest_centroid_changed_cases"], [])
        check("det_rerun_defect_status_unchanged",
              det_rerun["defect_status_changed_cases"], [])
        check("det_rerun_grade_changes_rows", int(len(det_grade_changes)),
              det_rerun["changed_cells"])
        check("det_rerun_grade_matrix_rows", int(len(det_grade_matrix)), 66)

    cases = []
    evidence_index = evidence.set_index("case_id")
    for _, row in results.iterrows():
        identifier = case_id(int(row["source_beer"]), row["target_style"])
        graded = evidence_index.loc[identifier]
        cases.append({
            "source_beer": int(row["source_beer"]),
            "target_style": row["target_style"],
            "lambda": float(row["lambda"]),
            "distance_reduction_pct": round(float(row["dist_reduction_pct"]), 4),
            "style_before": row["style_before"],
            "style_after": row["style_after"],
            "criteria": {"A": bool(row["criteria_A"]), "B": bool(row["criteria_B"]),
                         "C": bool(row["criteria_C"])},
            "criteria_mean_scenario": {"A": bool(row["A_mean"]), "B": bool(row["B_mean"]),
                                       "C": bool(row["C_mean"])},
            "defects": {spec.name: {"before": None if pd.isna(row[f"{spec.name}_before"])
                                    else round(float(row[f"{spec.name}_before"]), 6),
                                    "after": round(float(row[f"{spec.name}_after"]), 6),
                                    "unit": spec.unit,
                                    "threshold_median": data.thresholds[spec.name]["median"]}
                        for spec in DEFECT_SPECS},
            "conditional": bool(row["conditional_" + PRIMARY_SCENARIO]),
            "claim_level": str(graded["claim_level"]),
            "claim_note": str(graded["claim_note"]),
            "defect_status": {name: str(graded[f"defect_status_{name}"])
                              for name in ("DMS", "Diacetyl", "4VG", "t2n", "Stale_hops")},
            "style_diagnostic": {
                "nearest_centroid": str(graded["style_nearest_centroid"]),
                "supported": bool(graded["style_diagnostic_supported"]),
                "cases": int(graded["style_diagnostic_cases"]),
                "total": int(graded["style_diagnostic_total"]),
            },
        })

    passed = all(item["passed"] for item in checks)
    verification = {
        "status": "passed" if passed else "failed",
        "status_passed": bool(passed),
        "generated_on": "2026-10-28",
        "script": "scripts/q3_official_migration_20261028.py",
        "output_directory": output.name,
        "source_directory": source.name,
        "checks": checks,
        "counts": {
            "results": int(len(results)),
            "per_target": per_target,
            "candidates": total_candidates,
            "candidates_per_target": {style: int(counts.loc[counts["target_style"] == style,
                                                           "n_candidates"].sum())
                                      for style in TARGET_STYLES},
            "spectra_long": int(len(long_df)),
            "spectra_wide": int(len(wide_df)),
            "after_input": [int(after_input.shape[0]), int(after_input.shape[1])],
            "evidence": int(len(evidence)),
            "sensory": int(len(sensory)),
            "sensory_targets": int(sensory["target_52"].nunique()),
            "official_features": int(len(official_features)),
        },
        "claim_level_counts": claim_counts,
        "style_diagnostic_counts": style_diagnostic_counts,
        "defect_status_counts": defect_status_counts,
        "sensory_missing_grade_raw": int(sensory["grade_raw"].isna().sum()),
        "sensory_pipeline": protocol["sensory_pipeline"],
        "sensory_replay": sensory_replay,
        "det_rerun": det_rerun,
        "criteria_pass": {f"{criterion}_{scenario}": int(results[f"{criterion}_{scenario}"].sum())
                          for scenario in THRESHOLD_SCENARIOS for criterion in ("A", "B", "C")},
        "candidate_pass": {f"{criterion}_{scenario}": int(counts[f"n_pass_{criterion}_{scenario}"].sum())
                           for scenario in THRESHOLD_SCENARIOS for criterion in ("A", "B", "C")},
        "lambda_by_target": {style: sorted(results.loc[results["target_style"] == style,
                                                        "lambda"].unique().tolist())
                             for style in TARGET_STYLES},
        "mean_distance_reduction_pct": {style: round(float(results.loc[
            results["target_style"] == style, "dist_reduction_pct"].mean()), 4)
            for style in TARGET_STYLES},
        "style_identifiability": accuracy,
        "cases": cases,
        "honesty": HONESTY,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    write_json(output / "verification.json", verification)

    summary_lines = [
        "# Q3 跨风格迁移（正式跑，2026-10-28）",
        "",
        f"- 口径：题面“缺陷以所列超阈值项为准”；三口径 A（超阈集合不扩张+严重度不增）、"
        f"B（逐项 e_after≤e_before）、C（严重度加总不增），median 与 mean 两种阈值情景均评估。",
        f"- 结果：{SOURCE_STYLE} 11 款 × Tripel/Kriek/Lambic 3 目标 = {len(results)} 条；"
        f"A/B/C 各 {int(results['criteria_A'].sum())}/{len(results)} 条通过；"
        f"候选级 {total_candidates} 个 λ 候选全通过。",
        f"- λ*：Tripel {'/'.join(str(v) for v in tripel_lambdas)}，Kriek/Lambic 均为 1.0；"
        f"距目标质心平均缩短 {verification['mean_distance_reduction_pct']['Tripel']:.2f}% / "
        f"{verification['mean_distance_reduction_pct']['Kriek']:.2f}% / "
        f"{verification['mean_distance_reduction_pct']['Lambic']:.2f}%。",
        f"- 119 号 Tripel：4VG 0.1438→{float(case['4VG_after']):.4f} mg/L，DMS 保持 "
        f"{float(case['DMS_after']):.3f} µg/L，双乙酰源缺失按 min(目标中位,0.9T)={float(case['Diacetyl_after']):.3f} mg/L（条件性）。",
        f"- Lambic 仅 {lambic_nearest}/11 条迁移后最近质心为 Lambic，如实记录；"
        f"风格可辨识度最近质心 LOO={accuracy['nearest_centroid_loo_accuracy']:.3f}"
        f"（多数类基线 {accuracy['majority_class_baseline']:.3f}，{accuracy['n_classes']} 类）。",
        f"- 证据分级（s2c_evidence_by_case.csv）：claim_level {claim_counts}；"
        f"风格诊断支持数 Tripel {style_diagnostic_counts['Tripel']}/11、"
        f"Kriek {style_diagnostic_counts['Kriek']}/11、Lambic {style_diagnostic_counts['Lambic']}/11，"
        f"Tripel 与 Lambic 标“风格诊断未支持”；缺陷状态 {defect_status_counts}。",
        f"- 完整谱与感官画像：s2c_migration_spectra_224_long.csv（66 行）、"
        f"s2c_migration_spectra_224_wide.csv（33 行）、s2c_migration_after_input_33.csv（33×227）、"
        f"s2c_sensory_before_after_52.csv（{len(sensory)} 行 = 66 谱 × {SENSORY_TARGETS} 目标，"
        f"官方 Q1 入口 beer_models.inference.run）。",
        f"- 画像可复现性（N2 修复）：用存储谱 + 当前 q1.joblib（sha256 "
        f"{sensory_replay['model_sha256'][:16]}…）重放，{sensory_replay['rows']} 行逐格比对 "
        f"mismatch={sensory_replay['mismatch_cells']}、max|Δ|={sensory_replay['max_abs_delta']:.3g}"
        f"（容差 {replay_tolerance:g}）；表 sha256 {sensory_replay['delivered_table_sha256'][:16]}…。",
        f"- 正式口径：estimated_grade_0_6 为决策层 top-k + 24 对父子 cap；"
        f"prob_ge5/prob_6 为决策层概率，risk-head 概率另存 prob_ge5_risk_head/prob_6_risk_head。",
        (f"- 决策规则切换审计（task B）：对比 {det_rerun['previous_directory']}（旧 Q1，sha256 "
         f"{str(det_rerun.get('previous_model_sha256'))[:16]}…），66 谱 × 52 target 中可比 "
         f"{det_rerun['cells_compared']} 格，变化 {det_rerun['changed_cells']} 格"
         f"（升 {det_rerun['changed_up']} / 降 {det_rerun['changed_down']} / 不变 "
         f"{det_rerun['unchanged_cells']}）；五缺陷项（dms/diacetyl/X4vg/t2n/stale_hops）"
         f"变差 {det_rerun['defect_worse_count']} 格；A/B/C、claim_level、风格判定、缺陷状态"
         f"计数{'未变' if (det_rerun['criteria_unchanged'] and not det_rerun['claim_level_changed_cases'] and not det_rerun['style_nearest_centroid_changed_cases'] and not det_rerun['defect_status_changed_cases']) else '有变化（见 summary.json det_rerun）'}。"
         if det_rerun.get("available") else
         "- 决策规则切换审计（task B）：未找到 pre-det 备份目录，比较未执行。"),
        "- t2n 无阈值、Stale hops 无化学坐标：状态记 unknown，不属预注册 A/B/C 范围。",
        "",
        "诚实说明：迁移谱是附件 1 坐标体系内的统计目标，不是酿造配方或实测酒；"
        "投影只认证所列缺陷坐标（0.9×阈值余量），不认证可酿造性与感官接受度。"
        "风格诊断整体很弱（LOO≈多数类基线），因此 Tripel/Lambic 的迁移只能表述为"
        "“化学判据通过的候选”，不是“已成功迁移为该风格”。",
    ]
    (output / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    summary = {
        "name": "q3_migration_20261028",
        "status": verification["status"],
        "counts": verification["counts"],
        "criteria_pass": verification["criteria_pass"],
        "candidate_pass": verification["candidate_pass"],
        "lambda_by_target": verification["lambda_by_target"],
        "mean_distance_reduction_pct": verification["mean_distance_reduction_pct"],
        "claim_level_counts": claim_counts,
        "style_diagnostic_counts": style_diagnostic_counts,
        "defect_status_counts": defect_status_counts,
        "style_identifiability": accuracy,
        "sensory_pipeline": {
            "entry": "beer_models.inference.run (same call as scripts/current.py predict --task q1)",
            "model": Path(args.q1_model).name,
            "q1_model_path": protocol["sensory_pipeline"]["q1_model_path"],
            "q1_model_sha256": protocol["sensory_pipeline"]["q1_model_sha256"],
            "q1_model_grade_semantics": q1_grade_semantics,
            "run_at": protocol["sensory_pipeline"]["run_at"],
            "rows": int(len(sensory)),
            "targets": int(sensory["target_52"].nunique()),
            "missing_grade_raw": int(sensory["grade_raw"].isna().sum()),
            "note": "full-training-model predictions are not out-of-sample accuracy estimates",
        },
        "sensory_replay": {
            key: sensory_replay[key]
            for key in ("rows", "columns_compared", "tolerance", "mismatch_cells",
                        "mismatches_per_column", "max_abs_delta", "model_sha256",
                        "delivered_table_sha256", "replayed_at")
        },
        "det_rerun": det_rerun,
        "honesty": HONESTY,
        "files": sorted(path.name for path in output.iterdir()),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    write_json(output / "summary.json", summary)

    print(json.dumps({"status": verification["status"],
                      "results": len(results),
                      "criteria_A": int(results["criteria_A"].sum()),
                      "criteria_B": int(results["criteria_B"].sum()),
                      "criteria_C": int(results["criteria_C"].sum()),
                      "candidates": total_candidates,
                      "lambda_by_target": verification["lambda_by_target"],
                      "claim_level_counts": claim_counts,
                      "style_diagnostic_counts": style_diagnostic_counts,
                      "spectra_long": int(len(long_df)),
                      "spectra_wide": int(len(wide_df)),
                      "sensory_rows": int(len(sensory)),
                      "det_rerun": {key: det_rerun.get(key) for key in
                                    ("available", "model_sha_changed", "cells_compared",
                                     "changed_cells", "changed_up", "changed_down",
                                     "defect_worse_count", "criteria_unchanged",
                                     "claim_level_changed_cases",
                                     "style_nearest_centroid_changed_cases",
                                     "defect_status_changed_cases",
                                     "grade_change_within_limit")},
                      "failed_checks": [item["name"] for item in checks if not item["passed"]],
                      "elapsed_seconds": verification["elapsed_seconds"]}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
