"""Official runner for the Q1 panel-mean redirection experiment (task-9, 2026-10-28).

Pre-registration
----------------
The protocol is written to the output directory *before* any candidate is
fitted (see ``protocol.json``).  All thresholds, folds, seeds, metrics and the
baseline value come from ``configs/q1_panel_mean_20261028.json``.

What it does
------------
1. Loads the frozen official inputs (250 beers x 50 targets): 1.csv features,
   Zenodo raw ratings, official folds, official panel-weighted distribution,
   official Extra Trees OOF integer median.
2. Rebuilds the panel-weighted mean/median truth and checks it against
   ``observed_panel_distribution.csv``.
3. Reproduces the reference baseline (official ET integer median + fold-wise
   style/linear calibration -> 0.4622337899166068) at 1e-9.
4. Fits two direct panel-mean candidates with fixed hyperparameters on the
   frozen official folds and three seeds (task-5 repeated-validation seeds),
   with an inner-fold audit for the primary candidate.
5. Scores all methods (pooled, per repeat), runs the paired style-stratified
   beer bootstrap and applies the four pre-registered adoption gates.
6. Writes the six tables plus protocol/verification/summary JSON files.

The script never retrains the official integer-grade model and never modifies
existing outputs.  Failure of any verification check exits non-zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from beer_models.baseline import make_pipeline, make_splits
from beer_models.panel_data import load_inputs
from beer_models.q1_decision import calibrate_panel_mean, fold_style_mean
from beer_models.q1_panel_mean import (
    gate_verdicts,
    panel_mean_from_distribution,
    panel_mean_metrics,
    panel_median_from_distribution,
    paired_beer_bootstrap,
    round_grades,
    target_family,
    within_one_rate,
    exact_rate,
)
from beer_models.transforms import RobustAsinh

PRIMARY = "extra_trees_direct"
SECONDARY = "hist_gradient_boosting_direct"
REFERENCE = "baseline_et_posthoc"
METHODS = [PRIMARY, SECONDARY]
SCOPES = ["pooled_3repeats", "repeat_0", "repeat_1", "repeat_2"]

OUTPUT_FILES = {
    "pred_primary": "q1__panel_mean_pred_12500.csv",
    "pred_secondary": "q1__panel_mean_pred_hgb_12500.csv",
    "metrics": "q1__panel_mean_metrics.csv",
    "comparison": "q1__panel_mean_comparison.json",
    "protocol": "protocol.json",
    "verification": "verification.json",
    "summary": "summary.json",
}

PRED_COLUMNS = [
    "beer",
    "target",
    "panel_mean_pred",
    "grade_round_pred",
    "panel_mean_reference",
    "panel_median_reference",
    "outer_fold",
    "seed",
]

METRIC_COLUMNS = [
    "method",
    "scope",
    "panel_mean_mae",
    "panel_mean_rmse",
    "panel_median_exact",
    "panel_median_within_one",
    "macro_spearman",
    "macro_spearman_targets",
    "pearson_with_calibrated_baseline",
    "n_cells",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_config(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "q1_panel_mean_20261028":
        raise SystemExit("unexpected protocol_id in config")
    if not config.get("frozen_before_any_candidate_result"):
        raise SystemExit("config is not marked as pre-registered")
    return config


def build_frame(ids, targets, distribution, folds, style_map):
    beers = ids["beer"].to_numpy(int)
    n_beers, n_targets = len(beers), len(targets)
    frame = pd.DataFrame(
        {
            "beer": np.repeat(beers, n_targets),
            "target": np.tile(np.asarray(targets, dtype=object), n_beers),
        }
    )
    frame["outer_fold"] = frame["beer"].map(folds).astype(int)
    frame["style"] = frame["beer"].map(style_map)
    if frame["style"].isna().any():
        raise SystemExit("missing tasting style for some beers")
    panel_mean = panel_mean_from_distribution(distribution).reshape(-1)
    panel_median = panel_median_from_distribution(distribution).reshape(-1)
    if panel_mean.size != len(frame):
        raise SystemExit("distribution shape does not match the beer x target grid")
    frame["panel_mean_reference"] = panel_mean
    frame["panel_median_reference"] = panel_median.astype(int)
    frame["target_family"] = frame["target"].map(target_family)
    return frame


def check_truth(frame, observed):
    merged = frame.merge(
        observed[["beer", "target", "panel_weighted_mean", "panel_weighted_median"]],
        on=["beer", "target"],
        how="left",
        validate="one_to_one",
    )
    if merged[["panel_weighted_mean"]].isna().any().any():
        raise SystemExit("truth merge lost cells")
    mean_diff = float(
        np.max(np.abs(merged["panel_mean_reference"] - merged["panel_weighted_mean"]))
    )
    median_mismatch = int(
        (merged["panel_median_reference"] != merged["panel_weighted_median"].astype(int)).sum()
    )
    return {"max_abs_mean_diff": mean_diff, "median_cells_mismatch": median_mismatch}


def baseline_predictions(frame, et_oof, targets):
    merged = frame.merge(
        et_oof[["beer", "target", "predicted_grade"]],
        on=["beer", "target"],
        how="left",
        validate="one_to_one",
    )
    if merged["predicted_grade"].isna().any():
        raise SystemExit("official ET OOF does not cover every cell")
    legacy = merged["predicted_grade"].to_numpy(float)
    group_ids = merged["target"].to_numpy()
    fold_ids = merged["outer_fold"].to_numpy(int)
    style_base = fold_style_mean(
        merged["panel_mean_reference"].to_numpy(float),
        merged["style"].to_numpy(),
        group_ids,
        fold_ids,
    )
    features = np.column_stack([legacy, np.asarray(style_base, dtype=float)])
    calibrated, coefficients = calibrate_panel_mean(
        features,
        merged["panel_mean_reference"].to_numpy(float),
        group_ids,
        fold_ids,
    )
    return np.asarray(calibrated, dtype=float), legacy, coefficients


def fit_direct_models(x, y, train_index, test_index, seed, et_config, hgb_config):
    predictions = {}
    et_pipeline, _ = make_pipeline("extra_trees_asinh", et_config, seed)
    et_pipeline.fit(x[train_index], y[train_index])
    predictions[PRIMARY] = np.asarray(et_pipeline.predict(x[test_index]), dtype=float)
    hgb_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("compress", RobustAsinh()),
            (
                "model",
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(**hgb_config, random_state=seed),
                    n_jobs=1,
                ),
            ),
        ]
    )
    hgb_pipeline.fit(x[train_index], y[train_index])
    predictions[SECONDARY] = np.asarray(hgb_pipeline.predict(x[test_index]), dtype=float)
    return predictions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="attachment directory with 1.csv/4.csv/8.xlsx")
    parser.add_argument("--raw", default="data/reference/zenodo23624915/Raw sensory data anonymized.csv")
    parser.add_argument("--raw-protocol", default="configs/r2_raw_panel_protocol_20261026.json")
    parser.add_argument("--final-protocol", default="configs/final_protocol.json")
    parser.add_argument("--repeated-protocol", default="configs/task5_repeated_nested_protocol_20261026.json")
    parser.add_argument("--config", default="configs/q1_panel_mean_20261028.json")
    parser.add_argument("--q1-dir", default="outputs/r2_raw_q1_20261026")
    parser.add_argument("--decision-dir", default="outputs/q1_decision_20261028")
    parser.add_argument("--output", default="outputs/q1_panelmean_20261028")
    parser.add_argument("--skip-inner-audit", action="store_true", help="development shortcut; protocol requires the audit")
    args = parser.parse_args(argv)

    started = time.time()
    config = load_config(resolve(ROOT, args.config))
    gate_min_improvement = float(re.search(r"-\s*([0-9.]+)", config["adoption_gates"]["gate_1_point_improvement"]).group(1))
    gate_within_one_floor = float(re.search(r">=\s*([0-9.]+)", config["adoption_gates"]["gate_4_within_one"]).group(1))
    source = resolve(ROOT, args.source)
    raw_path = resolve(ROOT, args.raw)
    q1_dir = resolve(ROOT, args.q1_dir)
    decision_dir = resolve(ROOT, args.decision_dir)
    output = resolve(ROOT, args.output)
    raw_protocol = json.loads(resolve(ROOT, args.raw_protocol).read_text(encoding="utf-8"))
    final_protocol = json.loads(resolve(ROOT, args.final_protocol).read_text(encoding="utf-8"))
    repeated_protocol = json.loads(resolve(ROOT, args.repeated_protocol).read_text(encoding="utf-8"))

    inputs = {
        "source_attachments": source,
        "raw_sensory": raw_path,
        "raw_protocol": resolve(ROOT, args.raw_protocol),
        "official_folds": q1_dir / "rating_weights_and_folds.csv",
        "official_panel_distribution": q1_dir / "observed_panel_distribution.csv",
        "official_et_oof": q1_dir / "oof_predictions.csv.gz",
        "reference_calibrated_panel_mean": decision_dir / "q1__calibrated_panel_mean_12500.csv",
        "final_protocol": resolve(ROOT, args.final_protocol),
        "repeated_protocol": resolve(ROOT, args.repeated_protocol),
        "config": resolve(ROOT, args.config),
    }
    for name, path in inputs.items():
        if not path.exists():
            raise SystemExit(f"missing input {name}: {path}")

    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(config)
    protocol["stage"] = "protocol_written_before_computation"
    protocol["resolved_inputs"] = {name: str(path) for name, path in inputs.items()}
    protocol["input_sha256"] = {
        name: sha256_file(path) for name, path in inputs.items() if path.is_file()
    }
    protocol["script"] = str(Path(__file__).resolve())
    protocol["script_sha256"] = sha256_file(Path(__file__).resolve())
    protocol["module_sha256"] = sha256_file(ROOT / "src/beer_models/q1_panel_mean.py")
    protocol["started_at_unix"] = int(started)
    (output / OUTPUT_FILES["protocol"]).write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------------------------------------------------------------- inputs
    data = load_inputs(source, raw_path, raw_protocol)
    ids, x, targets, distribution = data["ids"], data["x"], data["targets"], data["distribution"]
    fold_table = pd.read_csv(inputs["official_folds"])
    folds = fold_table.groupby("beer")["outer_fold"].first().astype(int)
    if folds.nunique() != 3:
        raise SystemExit("official folds must contain exactly three outer folds")
    style_table = pd.read_csv(source / "1.csv", usecols=["beer", "tasting_category_fine"])
    style_map = style_table.groupby("beer")["tasting_category_fine"].first()
    observed = pd.read_csv(inputs["official_panel_distribution"])
    frame = build_frame(ids, targets, distribution, folds, style_map)
    truth_check = check_truth(frame, observed)
    if truth_check["max_abs_mean_diff"] > 1e-12 or truth_check["median_cells_mismatch"] != 0:
        raise SystemExit(f"truth does not match the official panel distribution: {truth_check}")

    et_oof = pd.read_csv(inputs["official_et_oof"])
    et_oof = et_oof.loc[et_oof["method"].eq("extra_trees"), ["beer", "target", "predicted_grade"]]
    baseline_pred, legacy, baseline_coefficients = baseline_predictions(frame, et_oof, targets)
    baseline_reference = frame["panel_mean_reference"].to_numpy(float)
    baseline_mae = float(np.mean(np.abs(baseline_pred - baseline_reference)))
    reported_baseline = float(config["reported_baseline"]["panel_mean_mae"])
    baseline_reproduced = bool(abs(baseline_mae - reported_baseline) <= 1e-9)
    if not baseline_reproduced:
        raise SystemExit(
            f"baseline not reproduced: computed {baseline_mae!r} vs reported {reported_baseline!r}"
        )
    baseline_median = frame["panel_median_reference"].to_numpy(int)
    baseline_within_one = within_one_rate(round_grades(baseline_pred), baseline_median)

    y = distribution @ np.arange(7, dtype=float)
    fold_ids = frame["outer_fold"].to_numpy(int)
    target_ids = frame["target"].to_numpy()
    styles = frame["style"].to_numpy()
    n_beers, n_targets = len(ids), len(targets)
    beer_index = {beer: index for index, beer in enumerate(ids["beer"].to_numpy(int))}
    beer_rows = frame["beer"].map(beer_index).to_numpy(int)

    candidate_predictions = {method: np.zeros((3, n_beers, n_targets)) for method in METHODS}
    inner_audit = {repeat: [] for repeat in range(3)}
    per_repeat_fold_seeds = {}
    for repeat in range(3):
        for fold in range(3):
            test_rows = np.flatnonzero(fold_ids == fold)
            train_rows = np.flatnonzero(fold_ids != fold)
            test_beers = np.unique(beer_rows[test_rows])
            train_beers = np.unique(beer_rows[train_rows])
            if np.intersect1d(test_beers, train_beers).size:
                raise SystemExit("train/test beer overlap detected")
            seed = int(20261026 + repeat * 1000 + fold * 100 + 99)
            per_repeat_fold_seeds[f"{repeat}/{fold}"] = seed
            predictions = fit_direct_models(
                x,
                y,
                train_beers,
                test_beers,
                seed,
                {"extra_trees": final_protocol["extra_trees"]},
                {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0},
            )
            for method in METHODS:
                candidate_predictions[method][repeat][test_beers, :] = predictions[method]
            if not args.skip_inner_audit:
                inner_splits = make_splits(
                    style_table.set_index("beer").loc[ids["beer"].to_numpy(int)[train_beers], "tasting_category_fine"].to_numpy(),
                    2,
                    int(20261024 + 100 + fold),
                )
                inner_errors = []
                for inner_fold, (inner_train, inner_test) in enumerate(inner_splits):
                    inner_seed = int(20261026 + repeat * 1000 + fold * 100 + inner_fold)
                    pipeline, _ = make_pipeline(
                        "extra_trees_asinh", {"extra_trees": final_protocol["extra_trees"]}, inner_seed
                    )
                    pipeline.fit(x[train_beers][inner_train], y[train_beers][inner_train])
                    inner_pred = pipeline.predict(x[train_beers][inner_test])
                    inner_errors.append(
                        float(np.mean(np.abs(inner_pred - y[train_beers][inner_test])))
                    )
                inner_audit[repeat].append(float(np.mean(inner_errors)))

    # ---------------------------------------------------------------- scores
    calibrated_reference = None
    csv_path = inputs["reference_calibrated_panel_mean"]
    if csv_path.exists():
        reference_table = pd.read_csv(csv_path)
        merged_reference = frame.merge(
            reference_table[["beer", "target", "panel_mean_calibrated"]],
            on=["beer", "target"],
            how="left",
            validate="one_to_one",
        )
        calibrated_reference = merged_reference["panel_mean_calibrated"].to_numpy(float)
    panel_mean_true = frame["panel_mean_reference"].to_numpy(float).reshape(n_beers, n_targets)
    panel_median_true = frame["panel_median_reference"].to_numpy(int).reshape(n_beers, n_targets)
    calibrated_matrix = (
        None
        if calibrated_reference is None
        else calibrated_reference.reshape(n_beers, n_targets)
    )
    baseline_matrix = baseline_pred.reshape(n_beers, n_targets)

    metrics_rows = []
    method_summary = {}
    for method in METHODS:
        stacked = candidate_predictions[method]
        errors = np.abs(stacked - panel_mean_true[None, :, :])
        per_repeat_mae = [float(errors[repeat].mean()) for repeat in range(3)]
        pooled_pred = stacked.mean(axis=0)
        pooled_metrics = panel_mean_metrics(
            pooled_pred, panel_mean_true, panel_median_true, calibrated_matrix
        )
        pooled_mae = float(errors.mean())
        bootstrap = None
        if method in METHODS:
            baseline_errors = np.abs(baseline_matrix - panel_mean_true)
            baseline_per_beer = baseline_errors.mean(axis=1)
            candidate_per_beer = errors.mean(axis=0).mean(axis=1)
            differences = baseline_per_beer - candidate_per_beer
            order = ids["beer"].to_numpy(int)
            bootstrap = paired_beer_bootstrap(
                differences,
                style_map.loc[order].to_numpy(),
                n_boot=1000,
                seed=20261026,
                alpha=0.05,
            )
        verdict = gate_verdicts(
            pooled_mae=pooled_mae,
            repeat_maes=per_repeat_mae,
            bootstrap_lower=bootstrap["lower"],
            pooled_within_one=pooled_metrics["panel_median_within_one"],
            baseline_mae=reported_baseline,
            minimum_improvement=gate_min_improvement,
            within_one_floor=gate_within_one_floor,
        )
        verdict["bootstrap"] = bootstrap
        method_summary[method] = {
            "verdict": verdict,
            "repeat_maes": per_repeat_mae,
            "pooled_mae_from_repeat_errors": pooled_mae,
        }
        for scope, predictions in (
            ("pooled_3repeats", pooled_pred),
            ("repeat_0", stacked[0]),
            ("repeat_1", stacked[1]),
            ("repeat_2", stacked[2]),
        ):
            row = panel_mean_metrics(
                predictions, panel_mean_true, panel_median_true, calibrated_matrix
            )
            row.update({"method": method, "scope": scope, "n_cells": int(predictions.size)})
            metrics_rows.append(row)
        if method == PRIMARY and not args.skip_inner_audit:
            row = dict(pooled_metrics)
            row.update(
                {
                    "method": method,
                    "scope": "inner_audit_primary",
                    "n_cells": int(pooled_pred.size),
                    "panel_mean_mae": float(inner_audit_repeat_mean(inner_audit)),
                }
            )
            metrics_rows.append(row)

    baseline_row = panel_mean_metrics(
        baseline_pred.reshape(n_beers, n_targets),
        panel_mean_true,
        panel_median_true,
        calibrated_matrix,
    )
    baseline_row.update({"method": REFERENCE, "scope": "official_folds", "n_cells": int(baseline_pred.size)})
    metrics_rows.insert(0, baseline_row)
    metrics_frame = pd.DataFrame(metrics_rows)[METRIC_COLUMNS]

    # ------------------------------------------------------------ determinism
    deterministic = True
    determinism_delta = 0.0
    repeat0_fold0 = (fold_ids == 0)
    train_beers = np.unique(beer_rows[~repeat0_fold0])
    test_beers = np.unique(beer_rows[repeat0_fold0])
    seed_check = per_repeat_fold_seeds["0/0"]
    first, _ = make_pipeline("extra_trees_asinh", {"extra_trees": final_protocol["extra_trees"]}, seed_check)
    first.fit(x[train_beers], y[train_beers])
    second, _ = make_pipeline("extra_trees_asinh", {"extra_trees": final_protocol["extra_trees"]}, seed_check)
    second.fit(x[train_beers], y[train_beers])
    determinism_delta = float(np.max(np.abs(first.predict(x[test_beers]) - second.predict(x[test_beers]))))
    deterministic = bool(determinism_delta == 0.0)

    # ------------------------------------------------------------ comparison
    comparison = {
        "protocol_id": config["protocol_id"],
        "baseline": {
            "method": REFERENCE,
            "panel_mean_mae": baseline_mae,
            "reported_baseline": reported_baseline,
            "reproduced": baseline_reproduced,
            "panel_median_within_one": baseline_within_one,
            "coefficients": baseline_coefficients,
        },
        "candidates": method_summary,
        "truth_check": truth_check,
        "inner_audit_repeat_mean_mae": {
            str(repeat): float(np.mean(values)) for repeat, values in inner_audit.items()
        }
        if not args.skip_inner_audit
        else {},
        "determinism": {"identical_refit": deterministic, "max_abs_delta": determinism_delta},
    }
    adoption = "do_not_adopt_keep_baseline"
    passing = [method for method in METHODS if method_summary[method]["verdict"]["all_pass"]]
    if passing:
        best = min(passing, key=lambda name: method_summary[name]["verdict"]["pooled_panel_mean_mae"])
        adoption = f"adopt_{best}"
    comparison["passing_candidates"] = passing
    comparison["adoption_recommendation"] = adoption

    summary = {
        "protocol_id": config["protocol_id"],
        "task": "task-9",
        "adoption_recommendation": adoption,
        "baseline_panel_mean_mae": baseline_mae,
        "reported_baseline": reported_baseline,
        "baseline_reproduced": baseline_reproduced,
        "per_gate_verdicts": {
            method: {
                "gate_1_point_improvement": method_summary[method]["verdict"]["gates"]["gate_1_point_improvement"],
                "gate_2_bootstrap": method_summary[method]["verdict"]["gates"]["gate_2_bootstrap"],
                "gate_3_every_repeat": method_summary[method]["verdict"]["gates"]["gate_3_every_repeat"],
                "gate_4_within_one": method_summary[method]["verdict"]["gates"]["gate_4_within_one"],
                "all_pass": method_summary[method]["verdict"]["all_pass"],
            }
            for method in METHODS
        },
        "method_panel_mean_mae": {
            method: {
                "pooled": method_summary[method]["verdict"]["pooled_panel_mean_mae"],
                "repeats": method_summary[method]["repeat_maes"],
                "improvement_vs_baseline": method_summary[method]["verdict"]["point_improvement"],
                "bootstrap": method_summary[method]["verdict"]["bootstrap"],
            }
            for method in METHODS
        },
        "runtime_seconds": None,
    }

    # --------------------------------------------------------------- outputs
    def candidate_table(method, repeat=0):
        stacked = candidate_predictions[method]
        table = frame[["beer", "target", "panel_mean_reference", "panel_median_reference", "outer_fold"]].copy()
        table["panel_mean_pred"] = stacked[repeat].reshape(-1)
        table["grade_round_pred"] = round_grades(table["panel_mean_pred"].to_numpy())
        table["seed"] = table["outer_fold"].map(
            lambda fold: per_repeat_fold_seeds[f"{repeat}/{int(fold)}"]
        ).astype(int)
        columns = PRED_COLUMNS + (["method"] if method == SECONDARY else [])
        if method == SECONDARY:
            table["method"] = method
        return table[columns]

    candidate_table(PRIMARY).to_csv(output / OUTPUT_FILES["pred_primary"], index=False)
    candidate_table(SECONDARY).to_csv(output / OUTPUT_FILES["pred_secondary"], index=False)
    metrics_frame.to_csv(output / OUTPUT_FILES["metrics"], index=False)
    summary["runtime_seconds"] = round(time.time() - started, 3)
    (output / OUTPUT_FILES["comparison"]).write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (output / OUTPUT_FILES["summary"]).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    # ---------------------------------------------------------- verification
    primary_table = pd.read_csv(output / OUTPUT_FILES["pred_primary"])
    secondary_table = pd.read_csv(output / OUTPUT_FILES["pred_secondary"])
    checks = {
        "protocol_written_first": (output / OUTPUT_FILES["protocol"]).exists(),
        "rows_12500_primary": len(primary_table) == 12500,
        "rows_12500_secondary": len(secondary_table) == 12500,
        "unique_cells_primary": not primary_table.duplicated(["beer", "target"]).any(),
        "unique_cells_secondary": not secondary_table.duplicated(["beer", "target"]).any(),
        "no_missing_values": not primary_table.isna().any().any()
        and not secondary_table.isna().any().any()
        and not metrics_frame[["method", "scope"]].isna().any().any(),
        "grade_range_0_6": bool(
            primary_table["grade_round_pred"].between(0, 6).all()
            and secondary_table["grade_round_pred"].between(0, 6).all()
        ),
        "truth_matches_official_distribution": bool(
            truth_check["max_abs_mean_diff"] <= 1e-12 and truth_check["median_cells_mismatch"] == 0
        ),
        "baseline_reproduced": bool(baseline_reproduced),
        "fold_count_3": int(folds.nunique()) == 3,
        "seed_per_fold_deterministic": bool(deterministic),
        "metrics_complete": bool(metrics_frame["panel_mean_mae"].notna().all()),
        "gates_match_summary": bool(
            all(
                summary["per_gate_verdicts"][method]["all_pass"]
                == comparison["candidates"][method]["verdict"]["all_pass"]
                for method in METHODS
            )
        ),
        "adoption_recommendation_consistent": bool(
            (adoption != "do_not_adopt_keep_baseline") == bool(passing)
        ),
    }
    verification = {
        "all_pass": bool(all(checks.values())),
        "checks": checks,
        "diagnostics": {
            "baseline_panel_mean_mae": baseline_mae,
            "baseline_within_one": baseline_within_one,
            "truth_check": truth_check,
            "fold_sizes": {int(fold): int(count) for fold, count in pd.Series(fold_ids).value_counts().sort_index().items()},
            "model_seeds": per_repeat_fold_seeds,
            "determinism": {"identical_refit": deterministic, "max_abs_delta": determinism_delta},
            "n_cells": int(len(primary_table)),
        },
    }
    (output / OUTPUT_FILES["verification"]).write_text(
        json.dumps(verification, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"verification": verification, "summary": summary}, ensure_ascii=False, indent=2, default=str))
    if not verification["all_pass"]:
        return 1
    return 0


def inner_audit_repeat_mean(inner_audit):
    values = [np.mean(items) for items in inner_audit.values() if items]
    return float(np.mean(values)) if values else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())

