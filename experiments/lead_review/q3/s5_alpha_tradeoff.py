"""POST-HOC feasibility check of the Q3 replacement gate (NOT pre-registered).

The pre-registered ``copula_quantile_projected`` run (``scripts/q3_official_generation_20261028.py``)
fails three rules: rank-correlation RMSE, Kriek coverage and Lambic coverage.  Its
``donor_quantile`` variant fails only the Lambic coverage rule.  Both results raise the
question whether *any* variance-restoring generator can satisfy
``variance_ratio in [0.8, 1.2]`` **and** the per-style held-out coverage rule on styles
with 10-11 training beers, or whether the two rules are in conflict at that sample size.

This script therefore sweeps a *partial* quantile-restoration coefficient ``alpha``:

    generated(alpha) = (1 - alpha) * interpolation + alpha * quantile_mapped_interpolation

``alpha = 0`` is the delivered interpolator (variance_ratio 0.55-0.86, full coverage),
``alpha = 1`` is the diagnostic-improved donor variant (variance restored, coverage drop).
Every point is projected on the defect caps, so all points satisfy the threshold caps.

This is an *exploration*, not a result to be injected into the official delivery: the
official protocol explicitly forbids post-result retuning.  If a passing alpha exists it
only tells the team which variant to pre-register for a future run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from beer_models.baseline import make_splits  # noqa: E402
from beer_models.generation import StyleGenerator  # noqa: E402
from beer_models.generation_copula import CopulaQuantileGenerator, defect_caps  # noqa: E402
from beer_models.panel_data import load_inputs  # noqa: E402
from beer_models.q3_raw import spectrum_diagnostics  # noqa: E402
from q3_official_generation_20261028 import PROTOCOL, load_thresholds  # noqa: E402
from r2_raw_q1 import PROTOCOL as BASE_PROTOCOL, RAW_MANIFEST  # noqa: E402

ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments" / "lead_review" / "q3")
    args = parser.parse_args(argv)
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    thresholds, _ = load_thresholds(source, "median")
    caps = defect_caps(thresholds)
    d = load_inputs(source, RAW_MANIFEST.parent / "Raw sensory data anonymized.csv",
                    json.loads(BASE_PROTOCOL.read_text(encoding="utf-8")))
    x, styles, features = d["x"], d["ids"]["tasting_category_fine"].to_numpy(), d["features"]
    cap_index = {field: features.index(field) for field in caps}

    rows = []
    for fold, (train, test) in enumerate(make_splits(styles, protocol["validation_folds"],
                                                     protocol["validation_seed"])):
        interpolation = StyleGenerator().fit(x[train], styles[train])
        copula = CopulaQuantileGenerator(features=features, components=8,
                                         defect_thresholds=thresholds).fit(x[train], styles[train])
        for index, style in enumerate(protocol["styles"]):
            train_index = np.flatnonzero(styles[train] == style)
            test_index = np.flatnonzero(styles[test] == style)
            ztrain = interpolation.robust_.transform(interpolation.x_[train_index])
            ztest = interpolation.robust_.transform(interpolation.impute_.transform(x[test[test_index]]))
            seed = int(protocol["validation_generation_seed"]) + fold * 100 + index
            base = interpolation.sample(style, protocol["generation_count"], "interpolate", seed)[0]
            mapped = copula.sample(style, protocol["generation_count"], seed,
                                   proposal="donor_quantile")[0]
            for alpha in ALPHAS:
                generated = (1.0 - alpha) * base + alpha * mapped
                for field, cap in caps.items():
                    generated[:, cap_index[field]] = np.minimum(generated[:, cap_index[field]], cap)
                row = {"outer_fold": fold, "style": style, "alpha": alpha, "seed": seed}
                row.update(spectrum_diagnostics(ztrain, ztest,
                                                interpolation.robust_.transform(generated), protocol))
                rows.append(row)
        print("fold %d done" % fold)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "s5_alpha_tradeoff.csv", index=False, encoding="utf-8-sig")

    means = frame.groupby("alpha").mean(numeric_only=True)
    per_style = frame.groupby(["style", "alpha"]).mean(numeric_only=True)
    base_means = means.loc[0.0]
    base_style = per_style.xs(0.0, level="alpha")
    gate_rows = []
    for alpha in ALPHAS:
        current = means.loc[alpha]
        rules = [
            ("marginal_w1", current["marginal_w1"], base_means["marginal_w1"], "<="),
            ("energy", current["energy"], base_means["energy"], "<="),
            ("exact_training_copy_rate", current["exact_training_copy_rate"],
             base_means["exact_training_copy_rate"], "<="),
            ("variance_lower", current["variance_ratio"], 0.8, ">="),
            ("variance_upper", current["variance_ratio"], 1.2, "<="),
            ("rank_correlation_rmse", current["rank_correlation_rmse"],
             base_means["rank_correlation_rmse"] + 0.05, "<="),
        ]
        passed_all = True
        for name, value, limit, operator in rules:
            passed = value >= limit - 1e-12 if operator == ">=" else value <= limit + 1e-12
            passed_all &= bool(passed)
            gate_rows.append({"alpha": alpha, "rule": name, "value": float(value),
                              "limit": float(limit), "operator": operator, "passed": bool(passed)})
        for style in protocol["styles"]:
            style_row = per_style.loc[(style, alpha)]
            style_base = base_style.loc[style]
            energy_ok = style_row["energy"] <= style_base["energy"] * 1.1 + 1e-12
            coverage_ok = style_row["heldout_coverage"] >= style_base["heldout_coverage"] - 0.05 - 1e-12
            passed_all &= bool(energy_ok and coverage_ok)
            gate_rows.append({"alpha": alpha, "rule": style + "_energy", "value": float(style_row["energy"]),
                              "limit": float(style_base["energy"] * 1.1), "operator": "<=",
                              "passed": bool(energy_ok)})
            gate_rows.append({"alpha": alpha, "rule": style + "_coverage",
                              "value": float(style_row["heldout_coverage"]),
                              "limit": float(style_base["heldout_coverage"] - 0.05), "operator": ">=",
                              "passed": bool(coverage_ok)})
        gate_rows.append({"alpha": alpha, "rule": "ALL", "value": float(passed_all), "limit": 1.0,
                          "operator": "==", "passed": bool(passed_all)})
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(output / "s5_alpha_gate.csv", index=False, encoding="utf-8-sig")

    summary = means[["marginal_w1", "energy", "variance_ratio", "rank_correlation_rmse",
                     "heldout_coverage", "exact_training_copy_rate", "near_training_copy_rate",
                     "internal_near_duplicate_rate"]]
    print(summary.round(5).to_string())
    print()
    print(per_style[["variance_ratio", "heldout_coverage", "rank_correlation_rmse"]].round(4).to_string())
    print()
    print(gate.loc[~gate["passed"]].to_string())
    print()
    print("passing alphas:", sorted(gate.loc[gate["rule"] == "ALL"].loc[lambda f: f["passed"], "alpha"].tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
