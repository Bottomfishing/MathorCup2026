"""v3 coverage diagnosis: why do Kriek/Lambic held-out spectra fall outside the generated support?

Reproduces the pre-registered validation folds (3 folds, seeds from
configs/task4_raw_q3_protocol_20261026.json) and records, for every held-out spectrum and style,
the nearest-generated distance versus the coverage radius used by
``beer_models.q3_raw.spectrum_diagnostics``.  Also evaluates the *oracle* generator whose sample
set is exactly the training spectra of the style, which is the tightest possible support: if the
oracle cannot cover a held-out spectrum, no generator confined to the training support can.

Writes experiments/lead_review/q3/v3_coverage_points.csv (per held-out spectrum) and
experiments/lead_review/q3/v3_fold_summary.csv (per fold/style/method coverage).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from beer_models.baseline import make_splits  # noqa: E402
from beer_models.generation import StyleGenerator  # noqa: E402
from beer_models.generation_copula import DonorResidualGenerator  # noqa: E402
from beer_models.panel_data import load_inputs  # noqa: E402

SOURCE = Path(r"C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件")
PROTOCOL = ROOT / "configs" / "task4_raw_q3_protocol_20261026.json"
BASE_PROTOCOL = ROOT / "configs" / "r2_raw_panel_protocol_20261026.json"
RAW = ROOT / "data" / "reference" / "zenodo23624915" / "Raw sensory data anonymized.csv"
OUT = Path(__file__).resolve().parent
CAP_FRACTION = 0.9
STATISTIC = "median"
METHODS = ("interpolate", "donor_residual_projected", "oracle_train_set")


def thresholds_from_reference(source: Path, statistic: str) -> dict:
    frame = pd.read_excel(source / "8.xlsx", sheet_name="Sensory thresholds").set_index("field_name_R")
    return {name: float(frame.loc[name, "threshold_" + statistic])
            for name in ("DMS", "X2.3.butanedione", "X4VG")}


def main() -> int:
    started = time.time()
    protocol = json.loads(PROTOCOL.read_text())
    d = load_inputs(SOURCE, RAW, json.loads(BASE_PROTOCOL.read_text()))
    x, styles = d["x"], d["ids"]["tasting_category_fine"].to_numpy()
    features = d["features"]
    beer = d["ids"]["beer"].to_numpy()
    thresholds = thresholds_from_reference(SOURCE, STATISTIC)
    p = len(features)
    radius_quantile = float(protocol["coverage_radius_quantile"])
    count = int(protocol["generation_count"])
    rows, fold_rows = [], []
    for fold, (train, test) in enumerate(make_splits(styles, protocol["validation_folds"],
                                                     protocol["validation_seed"])):
        interpolation = StyleGenerator().fit(x[train], styles[train])
        generator = DonorResidualGenerator(features=features, components=8,
                                           defect_thresholds=thresholds,
                                           cap_fraction=CAP_FRACTION).fit(x[train], styles[train])
        for index, style in enumerate(protocol["styles"]):
            train_index = np.flatnonzero(styles[train] == style)
            test_index = np.flatnonzero(styles[test] == style)
            ztrain = interpolation.robust_.transform(interpolation.x_[train_index])
            ztest = interpolation.robust_.transform(interpolation.impute_.transform(x[test[test_index]]))
            loo = cdist(ztrain, ztrain) / np.sqrt(p)
            np.fill_diagonal(loo, np.inf)
            radius = float(np.quantile(np.min(loo, axis=1), radius_quantile))
            seed = int(protocol["validation_generation_seed"]) + fold * 100 + index
            generated = {
                "interpolate": interpolation.sample(style, count, "interpolate", seed)[0],
                "donor_residual_projected": generator.sample(style, count, seed)[0],
                "oracle_train_set": interpolation.x_[train_index],
            }
            heldout_to_train = cdist(ztest, ztrain).min(axis=1) / np.sqrt(p)
            for method in METHODS:
                zg = interpolation.robust_.transform(generated[method])
                distance = cdist(ztest, zg) / np.sqrt(p)
                nearest_generated = distance.min(axis=1)
                nearest_index = distance.argmin(axis=1)
                fold_rows.append(dict(fold=fold, style=style, method=method, seed=seed,
                                      n_heldout=int(len(test_index)), n_train=int(len(train_index)),
                                      radius=radius,
                                      coverage=float(np.mean(nearest_generated <= radius)),
                                      n_covered=int(np.count_nonzero(nearest_generated <= radius)),
                                      max_nearest_generated=float(nearest_generated.max()),
                                      mean_nearest_generated=float(nearest_generated.mean()),
                                      mean_heldout_to_train=float(heldout_to_train.mean())))
                for position in range(len(test_index)):
                    point = ztest[position]
                    nearest = zg[nearest_index[position]]
                    difference = np.abs(point - nearest)
                    top = np.argsort(difference)[::-1][:4]
                    rows.append(dict(fold=fold, style=style, method=method,
                                     beer=int(beer[test_index[position]]), radius=radius,
                                     nearest_generated=float(nearest_generated[position]),
                                     ratio=float(nearest_generated[position] / radius),
                                     covered=bool(nearest_generated[position] <= radius),
                                     nearest_generated_index=int(nearest_index[position]),
                                     heldout_to_train=float(heldout_to_train[position]),
                                     top_coordinates=";".join(
                                         "%s:%.4f" % (features[column], float(difference[column]))
                                         for column in top)))
    points = pd.DataFrame(rows)
    folds = pd.DataFrame(fold_rows)
    points.to_csv(OUT / "v3_coverage_points.csv", index=False, encoding="utf-8")
    folds.to_csv(OUT / "v3_fold_summary.csv", index=False, encoding="utf-8")
    print(folds.groupby(["style", "method"])[["coverage", "radius", "mean_heldout_to_train",
                                              "mean_nearest_generated"]].mean().to_string())
    print("\nfailures (ratio>1) by style/method:")
    print(points[~points["covered"]].groupby(["style", "method"]).size().to_string())
    print("\nuncovered held-out points (v2 generator):")
    worst = points[(points["method"] == "donor_residual_projected") & (~points["covered"])]
    print(worst[["fold", "style", "beer", "nearest_generated", "radius", "ratio",
                 "heldout_to_train"]].sort_values("ratio", ascending=False).to_string(index=False))
    print("\nper fold/style coverage (all methods):")
    print(folds.pivot_table(index=["style", "fold"], columns="method", values="coverage").to_string())
    print("\nelapsed %.1fs" % (time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
