"""Q3 evidence helpers for the T4/T6 review gaps.

This module only produces *evidence* about the existing Q3 pipeline.  It never changes
the migration criteria, the lambda grid, the generator protocol or the pre-registered
gates.  Three independent pieces live here:

``style_oof``
    Out-of-fold style classification on the real beers (logistic / extra-trees, reusing
    the exact pipelines of :mod:`beer_models.style_models` and the parameters of
    ``configs/q3_style_protocol.json``) plus an out-of-fold nearest-centroid reference in
    the whitened 8-D geometry.  The classifier is evidence about *style support*, not a
    certificate of successful transfer.

``tripel_conflict``
    Diagnostics for the Tripel conflict: distance to the empirical target centroid,
    defect safety of that centroid itself, the defect-safe reference point (the training
    samples that satisfy every cap, or the projected centroid when there are none), and how
    much of the spectrum the hard defect projection actually clamps.

``ablation``
    Re-uses the frozen ``generation_validation.csv`` tables of the interpolate baseline,
    the v2 donor-residual generator and the v3 hybrid generator (same folds, same seeds)
    and adds the held-out denominator of every (fold, style) cell, so that a coverage
    change of one held-out spectrum is not mistaken for a stable improvement.

Nothing in this module reads or writes the frozen output directories; the script layer
does that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from beer_models.q3_migration import DEFECT_SPECS

STYLE_FOCUS = ("Tripel", "Kriek", "Lambic")
ABLATION_METRICS = (
    "marginal_w1",
    "energy",
    "variance_ratio",
    "rank_correlation_rmse",
    "heldout_coverage",
    "exact_training_copy_rate",
    "near_training_copy_rate",
    "internal_near_duplicate_rate",
    "model_median_w1",
)
#: generators compared in the ablation table; ``primary`` marks the three-way T6 comparison.
ABLATION_GENERATORS = {
    "interpolate": {"label": "interpolate (official baseline)", "primary": True, "version": "v1"},
    "copula_quantile_projected": {"label": "v1 PCA-copula + quantile map", "primary": False,
                                  "version": "v1"},
    "interp_quantile_projected": {"label": "v1.5 interpolate + quantile map", "primary": False,
                                  "version": "v1"},
    "donor_residual_projected": {"label": "v2 donor residual bootstrap", "primary": True,
                                 "version": "v2"},
    "hybrid_support_projected": {"label": "v3 hybrid (interior + donor residual)", "primary": True,
                                 "version": "v3"},
}
#: V3 pre-registered gate: 13 substantive rules plus one aggregate "all rules passed" item.
SUBSTANTIVE_GATE_RULES = 13
AGGREGATE_GATE_ITEMS = 1
TRIPEL_TARGET = "Tripel"


# --------------------------------------------------------------------------- helpers
def classifier_matrix(official_coordinates: np.ndarray, official_features: list[str],
                      bundle_features: list[str]) -> np.ndarray:
    """Return the 223-column classifier input in the bundle's own feature order."""
    index = {name: i for i, name in enumerate(official_features)}
    missing = [name for name in bundle_features if name not in index]
    if missing:
        raise ValueError(f"bundle features missing from the attachment panel: {missing[:5]}")
    return np.asarray(official_coordinates, dtype=float)[:, [index[name] for name in bundle_features]]


def oof_style_predictions(x: np.ndarray, styles: np.ndarray, config: dict, folds: list,
                          model_factory) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Out-of-fold predictions for every requested classifier name.

    ``model_factory(name, config, seed)`` must return a fit/predict pipeline; the fold
    index is added to ``config['seed']`` exactly like ``scripts/q3_style_study.py``.
    Imputers/scalers are fitted inside the pipeline on the training fold only.
    """
    names = ["logistic", "extra_trees"]
    out = {name: np.empty(len(x), dtype=object) for name in names}
    records = []
    for fold, (train, test) in enumerate(folds):
        for name in names:
            model = model_factory(name, config, int(config["seed"]) + fold).fit(x[train],
                                                                               styles[train])
            out[name][test] = model.predict(x[test])
            records.append({"fold": fold, "method": name, "n_train": int(len(train)),
                            "n_test": int(len(test))})
    return out, records


def centroid_oof_predictions(x: np.ndarray, styles: np.ndarray, folds: list) -> np.ndarray:
    """Out-of-fold nearest-centroid predictions in the raw coordinate space.

    Geometric reference only: centroids are recomputed on the training fold and the
    held-out row is assigned to the nearest one.
    """
    predictions = np.empty(len(styles), dtype=object)
    for train, test in folds:
        centroids = {}
        for style in np.unique(styles[train]):
            centroids[style] = x[train][styles[train] == style].mean(0)
        names = list(centroids)
        matrix = np.vstack([centroids[name] for name in names])
        for row in test:
            distance = np.sqrt(((matrix - x[row]) ** 2).mean(1))
            predictions[row] = names[int(np.argmin(distance))]
    return predictions


def classification_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    """Overall/macro metrics, per-class recall+support and the full confusion matrix."""
    labels = sorted({str(value) for value in observed})
    matrix = confusion_matrix(observed, predicted, labels=labels)
    per_class = []
    for j, style in enumerate(labels):
        support = int(matrix[j].sum())
        hits = int(matrix[j, j])
        per_class.append({"style": style, "support": support, "recall":
                          None if support == 0 else hits / support, "correct": hits})
    return {
        "accuracy": float(accuracy_score(observed, predicted)),
        "macro_recall": float(balanced_accuracy_score(observed, predicted)),
        "macro_f1": float(f1_score(observed, predicted, average="macro", zero_division=0)),
        "labels": labels,
        "confusion": matrix.tolist(),
        "per_class": per_class,
        "focus": {style: next(row for row in per_class if row["style"] == style)
                  for style in STYLE_FOCUS},
        "n": int(len(observed)),
    }


def fold_denominators(styles: np.ndarray, folds: list) -> pd.DataFrame:
    """Held-out and training counts per (fold, style); the coverage denominators."""
    rows = []
    for fold, (train, test) in enumerate(folds):
        for style in sorted(set(styles)):
            rows.append({"outer_fold": fold, "style": style,
                         "n_train_style": int((styles[train] == style).sum()),
                         "n_heldout_style": int((styles[test] == style).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- tripel diagnostics
def safe_reference(data, style: str = TRIPEL_TARGET, scenario: str = "median") -> dict:
    """Training samples of ``style`` that satisfy every threshold-defect cap.

    Safety is judged on the *physical* concentrations against ``0.9 * threshold`` of the
    requested scenario (the same headroom the migration projection uses).  t2n has no
    threshold and stale hops have no coordinate, so neither can enter this screen.
    """
    members = np.flatnonzero(data.styles == style)
    safe, table = [], []
    for row in members:
        verdict, concentrations = True, {}
        for name in ("DMS", "Diacetyl", "4VG"):
            value = data.defect_concentration(data.standardised[row], name)
            cap = data.physical_cap(name, scenario)
            concentrations[name] = value
            if cap is None or not np.isfinite(value) or value > cap:
                verdict = False
        table.append({"beer": int(data.ids[row]), **concentrations, "defect_safe": verdict})
        if verdict:
            safe.append(int(row))
    safe_table = pd.DataFrame(table)
    return {
        "style": style,
        "scenario": scenario,
        "n_style": int(len(members)),
        "n_safe": int(len(safe)),
        "safe_rows": safe,
        "safe_beers": [int(data.ids[row]) for row in safe],
        "table": safe_table,
    }


def defect_safe_point(data, standardised: np.ndarray, scenario: str = "median") -> dict:
    """Project an arbitrary standardised point onto the listed-defect caps.

    Used when the target style contains no training sample that satisfies every cap: the
    closest *defect-safe* version of the style centroid is the target centroid with the
    DMS/diacetyl/4VG coordinates clamped to ``0.9 * threshold``.  trans-2-nonenal and stale
    hops have no threshold/coordinate and are left untouched.
    """
    point = np.array(standardised, dtype=float, copy=True)
    applied, defects = {}, {}
    for spec in DEFECT_SPECS:
        cap = data.physical_cap(spec.name, scenario)
        if cap is None:
            continue
        index = data.features.index(spec.column)
        cap_standardised = data.standardise_value(spec.column, data.defect_coordinate(cap, spec.name))
        point[index] = min(point[index], cap_standardised)
        applied[spec.name] = float(cap)
        defects[spec.name] = float(data.defect_concentration(point, spec.name))
    return {"point": point, "caps": applied, "defects": defects}


def tripel_conflict(data, results: pd.DataFrame, safe: dict) -> pd.DataFrame:
    """Per-Tripel-case evidence on the distance-versus-defect-safety trade-off.

    ``results`` is the frozen 33-case migration table; only the Tripel rows are used and
    none of its criteria are recomputed or relaxed here.  When no training Tripel sample is
    defect-safe, the reference point becomes the defect-safe *projected centroid*, so the
    trade-off can still be quantified instead of silently skipped.
    """
    tripel_rows = results.loc[results["target_style"] == TRIPEL_TARGET].copy()
    target_members = np.flatnonzero(data.styles == TRIPEL_TARGET)
    target_mean = data.standardised[target_members].mean(0)
    target_whitened = data.whitened(target_mean)
    centroid_defects = {name: float(data.defect_concentration(target_mean, name))
                        for name in ("DMS", "Diacetyl", "4VG")}
    threshold_median = data.thresholds["DMS"]["median"]
    threshold_mean = data.thresholds["DMS"]["mean"]
    safe_rows = safe["safe_rows"]
    if safe_rows:
        reference_kind = "training_defect_safe_style_samples"
        safe_centroid = data.standardised[safe_rows].mean(0)
    else:
        reference_kind = "projected_style_centroid_under_caps"
        safe_centroid = defect_safe_point(data, target_mean, "median")["point"]
    safe_defects = {name: float(data.defect_concentration(safe_centroid, name))
                    for name in ("DMS", "Diacetyl", "4VG")}
    safe_beer = safe["safe_beers"][0] if safe["safe_beers"] else None
    source_rows = {int(data.ids[row]): int(row) for row in np.flatnonzero(data.styles == "Pils/Lager")}
    rows = []
    for _, case in tripel_rows.iterrows():
        beer = int(case["source_beer"])
        source = source_rows[beer]
        before = data.standardised[source]
        migration = data.migrate(source, TRIPEL_TARGET)
        after = migration["standardised"]
        # unprojected blend for the same lambda* shows what the hard projection had to clamp
        lam = float(case["lambda"])
        if abs(float(migration["lambda"]) - lam) > 1e-12:
            raise AssertionError(f"{beer}: frozen lambda {lam} differs from replay "
                                 f"{migration['lambda']}")
        blend = (1.0 - lam) * before + lam * target_mean
        delta = after - blend
        changed = np.flatnonzero(np.abs(delta) > 1e-12)
        before_whitened = data.whitened(before)
        after_whitened = data.whitened(after)
        row = {
            "case_id": f"S{beer:03d}_TO_{TRIPEL_TARGET}",
            "source_beer": beer,
            "lambda_star": lam,
            "d_target_before": float(case["d_target_before"]),
            "d_target_after": float(case["d_target_after"]),
            "dist_reduction_pct": float(case["dist_reduction_pct"]),
            "residual_target_gap_pct": 100.0 * float(case["d_target_after"]) /
                                       float(case["d_target_before"]),
            "change_from_source": float(np.linalg.norm(after_whitened - before_whitened)),
            "n_style_samples": safe["n_style"],
            "n_defect_safe_style_samples": safe["n_safe"],
            "defect_safe_reference_beer": safe_beer,
            "centroid_DMS_ugL": centroid_defects["DMS"],
            "centroid_DMSO_threshold_ratio": centroid_defects["DMS"] / threshold_median,
            "centroid_Diacetyl_mgL": centroid_defects["Diacetyl"],
            "centroid_4VG_mgL": centroid_defects["4VG"],
            "centroid_exceeds_median_caps": bool(centroid_defects["DMS"] > data.physical_cap("DMS", "median")
                                                 or centroid_defects["Diacetyl"] > data.physical_cap("Diacetyl", "median")
                                                 or centroid_defects["4VG"] > data.physical_cap("4VG", "median")),
            "DMS_before_ugL": float(case["DMS_before"]),
            "DMS_after_ugL": float(case["DMS_after"]),
            "DMS_threshold_median": threshold_median,
            "DMS_threshold_mean": threshold_mean,
            "DMS_cap_median": data.physical_cap("DMS", "median"),
            "DMS_cap_mean": data.physical_cap("DMS", "mean"),
            "DMS_meets_cap_median": bool(float(case["DMS_after"]) <= data.physical_cap("DMS", "median") + 1e-12),
            "DMS_meets_cap_mean": bool(float(case["DMS_after"]) <= data.physical_cap("DMS", "mean") + 1e-12),
            "Diacetyl_after_mgL": float(case["Diacetyl_after"]),
            "4VG_after_mgL": float(case["4VG_after"]),
            "n_coordinates_clamped": int(changed.size),
            "clamped_fields": ";".join(data.features[j] for j in changed[:8]),
            "total_abs_projection_change_z": float(np.abs(delta).sum()),
            "projection_change_share_pct": 100.0 * float(np.abs(delta).sum()) /
                                           max(float(np.abs(after - before).sum()), 1e-12),
            "safe_reference_kind": reference_kind,
            "safe_reference_DMS_ugL": safe_defects["DMS"],
            "safe_reference_Diacetyl_mgL": safe_defects["Diacetyl"],
            "safe_reference_4VG_mgL": safe_defects["4VG"],
        }
        safe_whitened = data.whitened(safe_centroid)
        row["d_safe_before"] = float(np.linalg.norm(before_whitened - safe_whitened))
        row["d_safe_after"] = float(np.linalg.norm(after_whitened - safe_whitened))
        row["safe_reduction_pct"] = 100.0 * (1.0 - row["d_safe_after"] / row["d_safe_before"])
        row["safe_distance_over_target_distance"] = row["d_safe_after"] / max(row["d_target_after"], 1e-12)
        row["target_over_safe_distance"] = row["d_target_after"] / max(row["d_safe_after"], 1e-12)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["target_centroid_defects"] = centroid_defects
    table.attrs["safe_centroid"] = safe_centroid
    table.attrs["safe_reference_kind"] = reference_kind
    table.attrs["safe_reference_defects"] = safe_defects
    return table


def style_classifier_application(data, classifier: dict, style_names: list[str]) -> pd.DataFrame:
    """Apply the frozen full-fit style classifiers to the 33 migrated spectra."""
    results, _ = data.run()
    rows = []
    for _, case in results.iterrows():
        beer = int(case["source_beer"])
        style = str(case["target_style"])
        source = int(np.flatnonzero(data.ids == beer)[0])
        after = data.migrate(source, style)["standardised"]
        spectrum = data.official_spectrum(source, standardised=after)
        matrix = classifier_matrix(spectrum[None, :], data.official_features,
                                   classifier["features"])
        probabilities = {}
        for name in ("logistic", "extra_trees"):
            model = classifier["classifiers"][name]
            probabilities[name] = float(model.predict_proba(matrix)[0, list(model.classes_).index(style)])
        nearest, _ = data.nearest_style(after)
        rows.append({
            "case_id": f"S{beer:03d}_TO_{style}",
            "source_beer": beer,
            "requested_style": style,
            "nearest_centroid_after": nearest,
            "logistic_prediction": str(classifier["classifiers"]["logistic"].predict(matrix)[0]),
            "extra_trees_prediction": str(classifier["classifiers"]["extra_trees"].predict(matrix)[0]),
            "logistic_probability_requested": probabilities["logistic"],
            "extra_trees_probability_requested": probabilities["extra_trees"],
        })
    table = pd.DataFrame(rows)
    table["classifier_consensus_requested"] = (
        (table["logistic_prediction"] == table["requested_style"]) &
        (table["extra_trees_prediction"] == table["requested_style"]))
    return table


# --------------------------------------------------------------------- ablation IO
def ablation_table(frames: dict[str, pd.DataFrame], denominators: pd.DataFrame) -> pd.DataFrame:
    """Concatenate frozen validation tables and attach the held-out denominators."""
    pieces = []
    for generator, frame in frames.items():
        piece = frame.copy()
        piece["generator"] = generator
        pieces.append(piece)
    table = pd.concat(pieces, ignore_index=True)
    missing = [column for column in ABLATION_METRICS if column not in table.columns]
    if missing:
        raise ValueError(f"validation tables lack metrics {missing}")
    keys = denominators[["outer_fold", "style", "n_train_style", "n_heldout_style"]]
    merged = table.merge(keys, on=["outer_fold", "style"], how="left", validate="many_to_one")
    if merged["n_heldout_style"].isna().any():
        raise ValueError("some (fold, style) cells have no held-out denominator")
    merged["n_heldout_style"] = merged["n_heldout_style"].astype(int)
    merged["n_train_style"] = merged["n_train_style"].astype(int)
    merged["generator_label"] = merged["generator"].map(
        {name: meta["label"] for name, meta in ABLATION_GENERATORS.items()})
    merged["primary_comparison"] = merged["generator"].map(
        {name: meta["primary"] for name, meta in ABLATION_GENERATORS.items()})
    merged["gate_version"] = merged["generator"].map(
        {name: meta["version"] for name, meta in ABLATION_GENERATORS.items()})
    return merged


def ablation_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per generator (over all folds) and per (generator, style)."""
    columns = list(ABLATION_METRICS)
    overall = (table.groupby("generator", sort=False)[columns].mean()
               .reset_index().assign(scope="all_styles"))
    by_style = (table.groupby(["generator", "style"], sort=False)[columns].mean()
                .reset_index().assign(scope="by_style"))
    summary = pd.concat([overall, by_style], ignore_index=True)

    def denominators(row) -> tuple[str, int]:
        selected = table.loc[table["generator"] == row["generator"]]
        if row["scope"] == "by_style":
            selected = selected.loc[selected["style"] == row["style"]]
        counts = (selected.groupby("outer_fold")["n_heldout_style"].sum()
                  .reindex(range(selected["outer_fold"].nunique())).tolist())
        return "/".join(str(int(value)) for value in counts), int(sum(counts))

    values = [denominators(row) for _, row in summary.iterrows()]
    summary["heldout_per_fold"] = [value[0] for value in values]
    summary["heldout_total"] = [value[1] for value in values]
    return summary


def gate_breakdown(gate: pd.DataFrame) -> dict:
    """Split the pre-registered gate into substantive rules and aggregate summary items.

    ``gate`` is a frozen ``generation_gate.csv`` (13 rules).  The published verification
    counts ``gate_all_passed`` on top of those rules, so a "14/14" headline is one
    aggregate item plus 13 substantive ones and must be reported as such.
    """
    rules = [str(rule) for rule in gate["rule"]]
    passed = gate["passed"].astype(bool)
    return {
        "substantive_rules": len(rules),
        "substantive_rules_passed": int(passed.sum()),
        "aggregate_items": AGGREGATE_GATE_ITEMS,
        "aggregate_items_passed": int(bool(passed.all())),
        "total_items": len(rules) + AGGREGATE_GATE_ITEMS,
        "rules": [{"rule": rule, "passed": bool(flag), "kind": "substantive"}
                  for rule, flag in zip(rules, passed)] + [
                     {"rule": "gate_all_passed", "passed": bool(passed.all()), "kind": "aggregate"}],
    }
