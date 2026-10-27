"""Official runner for the Q1 panel-mean redirection experiment (task-9/task-15, 2026-10-28).

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
3. Reproduces the **leak-free** reference baseline: the official ET integer
   median mapped by the strictly nested fold-wise style/linear calibration
   (task-12 audit fix; 0.46607189992522735, rounded within-one 0.91416) at
   1e-9.  The old 0.4622337899166068 value used cross-fold calibration features
   and is voided; it is never used as the comparison baseline here.
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
from beer_models.q1_decision import apply_nested_linear, fit_nested_linear
from beer_models.raw_panel import fit_shared
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

BASELINE_METHOD = "nested_style_linear"
BASELINE_METHOD_LINEAR = "nested_linear"
OUTER_FOLD_SEED = 20261024
INNER_FOLDS = 2

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


def outer_style_base(frame, truth_mean):
    """Per outer fold, per target/style panel-mean averages of the outer-training set."""
    data = pd.DataFrame(
        {
            "target": frame["target"].astype(str).to_numpy(),
            "fold": frame["outer_fold"].astype(int).to_numpy(),
            "style": frame["style"].astype(str).to_numpy(),
            "mean": np.asarray(truth_mean, dtype=float),
        }
    )
    result = np.empty(len(data), dtype=float)
    for fold in sorted(data["fold"].unique()):
        test = (data["fold"] == fold).to_numpy()
        train = ~test
        grouped = data[train].groupby(["target", "style"])["mean"].mean()
        global_mean = data[train].groupby("target")["mean"].mean()
        keys = pd.MultiIndex.from_arrays(
            [data.loc[test, "target"], data.loc[test, "style"]]
        )
        values = grouped.reindex(keys).to_numpy(float)
        fallback = global_mean.reindex(data.loc[test, "target"]).to_numpy(float)
        result[test] = np.where(np.isnan(values), fallback, values)
    return result


def build_inner_predictions(panel, panel_protocol, beers, fold_of_beer, targets):
    """Inner-fold OOF predictions inside every outer-training set (leak-free).

    For each outer fold the official raw-panel model is refitted on that fold's
    outer-training beers only, and two inner folds of those beers produce OOF
    integer grades plus per-target style means.  No outer-validation label is
    read here, so the calibrators fitted below cannot see the validation fold.
    """
    n_beers = int(len(beers))
    n_targets = int(len(targets))
    distribution = np.asarray(panel["distribution"], dtype=float)
    mean_matrix = (distribution * np.arange(distribution.shape[2])).sum(axis=2)
    median_matrix = panel_median_from_distribution(distribution)
    inner_grade = np.full((n_beers, n_targets), np.nan)
    inner_style = np.full((n_beers, n_targets), np.nan)
    inner_fold = np.full(n_beers, -1, dtype=int)
    styles = panel["ids"]["tasting_category_fine"].astype(str).to_numpy()
    fit_rows = {}
    for fold in sorted(set(int(value) for value in fold_of_beer.values())):
        train_index = np.array(
            [position for position, beer in enumerate(beers)
             if int(fold_of_beer[int(beer)]) != fold], dtype=int)
        splits = make_splits(styles[train_index], INNER_FOLDS, OUTER_FOLD_SEED + 100 + fold)
        for inner_index, (fit_local, valid_local) in enumerate(splits):
            fit_index = train_index[np.asarray(fit_local, dtype=int)]
            valid_index = train_index[np.asarray(valid_local, dtype=int)]
            model = fit_shared(panel, fit_index, panel_protocol,
                               panel_protocol["model_seed"])
            grades = np.asarray(model.predict(panel["x"][valid_index])["grade"], dtype=float)
            if grades.shape != (len(valid_index), n_targets):
                raise RuntimeError("inner model returned an unexpected grade shape")
            inner_grade[valid_index] = grades
            inner_fold[valid_index] = inner_index
            for column in range(n_targets):
                values = mean_matrix[fit_index, column]
                grouped = pd.Series(values).groupby(styles[fit_index]).mean()
                global_mean = float(values.mean())
                inner_style[valid_index, column] = (
                    grouped.reindex(styles[valid_index]).fillna(global_mean).to_numpy())
        fit_rows[str(fold)] = {
            "outer_training_beers": int(train_index.size),
            "inner_fits": int(len(splits)),
            "inner_cells": int(train_index.size * n_targets),
            "outer_validation_cells": int((len(beers) - train_index.size) * n_targets),
        }
    if (inner_fold < 0).any():
        raise RuntimeError("some outer-training beer has no inner-fold prediction")
    return {"grade": inner_grade, "style": inner_style, "inner_fold": inner_fold,
            "mean": mean_matrix, "median": median_matrix, "fit_rows": fit_rows}


def nested_baseline(frame, et_oof, panel, panel_protocol, beers, fold_of_beer,
                    targets, truth_mean):
    """Strictly nested calibration baseline (task-12 audit fix).

    Per outer fold the calibrator is fitted only on inner OOF cells of the
    outer-training beers; the style covariate is likewise an outer-training
    statistic.  The official ET OOF integer median is only an input score, never
    a label source, for the validation fold.
    """
    merged = frame.merge(
        et_oof[["beer", "target", "predicted_grade"]],
        on=["beer", "target"],
        how="left",
        validate="one_to_one",
    )
    if merged["predicted_grade"].isna().any():
        raise SystemExit("official ET OOF does not cover every cell")
    legacy = merged["predicted_grade"].to_numpy(float)
    inner = build_inner_predictions(panel, panel_protocol, beers, fold_of_beer, targets)
    style_base = outer_style_base(frame, truth_mean)
    target_labels = frame["target"].astype(str).to_numpy()
    fold_labels = frame["outer_fold"].astype(int).to_numpy()
    beer_position = {int(beer): index for index, beer in enumerate(beers)}
    row_beer = np.array([beer_position[int(beer)] for beer in frame["beer"].to_numpy(int)])
    predictions = {
        BASELINE_METHOD: np.full(len(frame), np.nan),
        BASELINE_METHOD_LINEAR: np.full(len(frame), np.nan),
    }
    calibration_rows = {}
    for fold in sorted(set(int(value) for value in fold_of_beer.values())):
        test_rows = np.flatnonzero(fold_labels == fold)
        train_positions = np.array(
            [position for position, beer in enumerate(beers)
             if int(fold_of_beer[int(beer)]) != fold], dtype=int)
        if set(row_beer[test_rows].tolist()) & set(train_positions.tolist()):
            raise SystemExit("outer-validation beer leaked into the calibration fit")
        flat_grade = inner["grade"][train_positions].ravel()
        flat_style = inner["style"][train_positions].ravel()
        flat_mean = inner["mean"][train_positions].ravel()
        if np.isnan(flat_grade).any():
            raise SystemExit("inner OOF grade matrix is incomplete")
        if np.isnan(flat_style).any():
            raise SystemExit("inner OOF style matrix is incomplete")
        group = np.tile(np.asarray(targets, dtype=object), train_positions.size)
        fold_array = np.full(flat_grade.size, str(fold), dtype=object)
        style_models = fit_nested_linear(
            np.column_stack([flat_grade, flat_style]), flat_mean, group, fold_array
        )
        linear_models = fit_nested_linear(
            flat_grade.reshape(-1, 1), flat_mean, group, fold_array
        )
        test_group = target_labels[test_rows]
        test_fold = fold_labels[test_rows].astype(str)
        predictions[BASELINE_METHOD][test_rows] = apply_nested_linear(
            np.column_stack([legacy[test_rows], style_base[test_rows]]),
            test_group, test_fold, style_models,
        )
        predictions[BASELINE_METHOD_LINEAR][test_rows] = apply_nested_linear(
            legacy[test_rows].reshape(-1, 1), test_group, test_fold, linear_models,
        )
        calibration_rows[str(fold)] = {
            "calibration_fit_cells": int(flat_grade.size),
            "outer_validation_cells": int(test_rows.size),
            "calibration_fit_beers": int(train_positions.size),
            "inner_folds_used": int(INNER_FOLDS),
        }
    if np.isnan(predictions[BASELINE_METHOD]).any():
        raise SystemExit("nested baseline prediction matrix is incomplete")
    return predictions, style_base, calibration_rows, inner["fit_rows"]


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
    beers_int = ids["beer"].to_numpy(int)
    fold_of_beer = {int(beer): int(fold) for beer, fold in folds.items()}
    baseline_reference = frame["panel_mean_reference"].to_numpy(float)
    nested_preds, style_base, calibration_rows, inner_fit_rows = nested_baseline(
        frame, et_oof, data, raw_protocol, beers_int, fold_of_beer, targets,
        baseline_reference,
    )
    baseline_pred = nested_preds[BASELINE_METHOD]
    baseline_linear_pred = nested_preds[BASELINE_METHOD_LINEAR]
    baseline_mae = float(np.mean(np.abs(baseline_pred - baseline_reference)))
    baseline_linear_mae = float(np.mean(np.abs(baseline_linear_pred - baseline_reference)))
    reported_baseline = float(config["reported_baseline"]["panel_mean_mae"])
    reported_baseline_linear = float(config["reported_baseline"]["panel_mean_mae_linear"])
    baseline_reproduced = bool(abs(baseline_mae - reported_baseline) <= 1e-9)
    if not baseline_reproduced:
        raise SystemExit(
            f"baseline not reproduced: computed {baseline_mae!r} vs reported {reported_baseline!r}"
        )
    baseline_linear_reproduced = bool(
        abs(baseline_linear_mae - reported_baseline_linear) <= 1e-9
    )
    if not baseline_linear_reproduced:
        raise SystemExit(
            "nested linear baseline not reproduced: "
            f"computed {baseline_linear_mae!r} vs reported {reported_baseline_linear!r}"
        )
    baseline_median = frame["panel_median_reference"].to_numpy(int)
    baseline_within_one = within_one_rate(round_grades(baseline_pred), baseline_median)
    reported_within_one = float(config["reported_baseline"]["panel_median_within_one_rounded"])
    baseline_within_one_reproduced = bool(abs(baseline_within_one - reported_within_one) <= 1e-12)
    if not baseline_within_one_reproduced:
        raise SystemExit(
            "within-one baseline not reproduced: "
            f"computed {baseline_within_one!r} vs reported {reported_within_one!r}"
        )

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
            "reported_within_one": reported_within_one,
            "within_one_reproduced": baseline_within_one_reproduced,
            "nested_ci_style_linear": {
                "panel_mean_mae": baseline_mae,
                "reported": reported_baseline,
                "reproduced": baseline_reproduced,
            },
            "nested_ci_linear": {
                "panel_mean_mae": baseline_linear_mae,
                "reported": reported_baseline_linear,
                "reproduced": baseline_linear_reproduced,
            },
            "calibration_rows": calibration_rows,
            "inner_prediction_rows": inner_fit_rows,
            "leak_fix": "task-12 audit 46ca932: calibrators and style means are fitted only "
                        "on inner OOF cells of the outer-training set; no validation-fold "
                        "label enters any fit",
            "voided_pre_fix_baseline": 0.4622337899166068,
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
        "task": "task-9 (task-15 leak-fixed baseline)",
        "adoption_recommendation": adoption,
        "baseline_panel_mean_mae": baseline_mae,
        "reported_baseline": reported_baseline,
        "baseline_reproduced": baseline_reproduced,
        "baseline_nested_ci_linear_panel_mean_mae": baseline_linear_mae,
        "baseline_panel_median_within_one": baseline_within_one,
        "reported_baseline_within_one": reported_within_one,
        "baseline_within_one_reproduced": baseline_within_one_reproduced,
        "voided_pre_fix_baseline": 0.4622337899166068,
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
        "baseline_within_one_reproduced": bool(baseline_within_one_reproduced),
        "baseline_nested_linear_reproduced": bool(baseline_linear_reproduced),
        "baseline_calibration_is_nested": bool(
            all(
                row["calibration_fit_cells"] == row["calibration_fit_beers"] * len(targets)
                and row["calibration_fit_cells"] + row["outer_validation_cells"] == len(frame)
                for row in calibration_rows.values()
            )
        ),
        "no_validation_beer_in_calibration_fit": True,  # enforced with SystemExit above
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
            "baseline_nested_linear_panel_mean_mae": baseline_linear_mae,
            "baseline_calibration_rows": calibration_rows,
            "baseline_inner_prediction_rows": inner_fit_rows,
            "voided_pre_fix_baseline": 0.4622337899166068,
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

