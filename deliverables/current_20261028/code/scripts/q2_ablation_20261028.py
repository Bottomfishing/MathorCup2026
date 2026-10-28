"""Q2 graph-neighbourhood ablation: does the 2.csv co-expression term help?

Compares the delivered Q2 ranking (neighbourhood weight 0.5) with a plain
ridge ranking (weight 0.0) built from exactly the same samples, preprocessing,
ridge fit, bootstrap resampling and evidence grading.  Everything is written to
``outputs/q2_ablation_20261028/``; the delivered tables under
``outputs/q2_top3_20261028/`` are only read, never modified.

Run from the repository root::

    python scripts/q2_ablation_20261028.py

The protocol (including the pre-registered gain criteria) is written before the
results are computed so that the verdict cannot be tuned after seeing numbers.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beer_audit.audit import save_csv, sha256, write_json  # noqa: E402
from beer_models import q2_ablation as ablation  # noqa: E402
from beer_models.q2_top3 import (  # noqa: E402
    DEFAULT_BOOTSTRAP,
    DEFAULT_EXCLUDED,
    DEFAULT_SEED,
    IDENTITY_MARKERS,
    META_COLUMNS,
    adjacency_from_corr,
    adjacency_subset,
    as_compound_label,
    bootstrap_jaccard,
    bootstrap_top_k,
    candidate_features,
    canon_key,
    evidence_grade,
    neighbourhood_scores,
    preprocess,
    ridge_coefficients,
    selection_frequencies,
    top_k_fields,
)

OFFICIAL_SCRIPT = ROOT / "scripts" / "q2_official_20261028.py"
DEFAULT_CONFIG = ROOT / "configs" / "q2_top3_20261028.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "q2_ablation_20261028"
MAIN_TABLE = ROOT / "outputs" / "q2_top3_20261028" / "q2__top3_all50_150.csv"
FEATURE_DICTIONARY = ROOT / "outputs" / "p0_audit_20261023_final" / "feature_dictionary.csv"

TIER_NUMBER = {"A": 1, "B": 2, "C": 3}
TIER_LABEL = {1: "A", 2: "B", 3: "C"}


def load_official_module():
    """Load the delivered Q2 script to reuse its data-mapping helpers."""

    spec = importlib.util.spec_from_file_location("q2_official_for_ablation", OFFICIAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None, help="attachment directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args(argv)


def resolve_source(arguments, official) -> Path:
    if arguments.source is not None:
        return arguments.source
    return Path(official.FALLBACK_SOURCE)


def cross_correlation_lookup(source: Path, targets: list[str], aliases: dict, fields: list[str]):
    """Map (target, field) pairs to the 6.csv cross-correlation value."""

    official = load_official_module()
    matrix, target_column = official.map_targets_to_cross_corr(source, targets)
    row_of_field: dict[str, int] = {}
    for position, name in enumerate(matrix.index):
        field = aliases.get(canon_key(str(name)))
        if field is not None:
            row_of_field.setdefault(field, position)
    values: dict[tuple[str, str], float] = {}
    for target in targets:
        column = matrix[target_column[target]] if target_column[target] in matrix.columns else None
        if column is None:
            continue
        for field, position in row_of_field.items():
            value = float(column.iloc[position])
            if not np.isnan(value):
                values[(target, field)] = value
    return values, matrix, target_column, row_of_field


def build_protocol(context: dict) -> dict:
    return {
        "protocol_id": "q2_ablation_20261028",
        "baseline_commit": context["commit"],
        "purpose": (
            "Direct ablation of the 2.csv co-expression neighbourhood term in the delivered "
            "Q2 compound-factor ranking: weight 0.5 (delivered) versus weight 0.0 (plain ridge)."
        ),
        "question": (
            "Does adding the graph-neighbourhood term produce a reliable improvement in bootstrap "
            "stability and/or held-out predictive utility, compared with a plain ridge ranking built "
            "from exactly the same samples and pipeline?"
        ),
        "design": {
            "identical_pipeline": True,
            "samples": "250 beers, 50 panel targets from 4.csv, 219 candidate compounds (231 - 12 excluded)",
            "preprocessing": "per-column median imputation -> arcsinh -> column standardisation; Y centred",
            "model": "ridge, alpha=1000",
            "variants": {"graph": 0.5, "no_graph": 0.0},
            "bootstrap": {"repeats": 200, "grouping": "beer (row resampling)", "seed": 20261027},
            "graph_source": "2.csv (recomputed Spearman of 1.csv), threshold |rho| >= 0.5",
            "descriptive_table_graph_source": "full-sample 2.csv graph, same as the delivered table",
            "predictive_check": {
                "folds": 5,
                "grouping": "beer",
                "fold_seed": 20261028,
                "feature_selection": "top-3 per target inside the training fold only",
                "graph_rebuilt_per_fold": True,
                "reader_model": "ridge on the three selected standardised features, alpha=1.0",
                "target": "held-out 4.csv standardised sensory responses",
            },
        },
        "pre_registered_criteria": {
            "predictive_mae": {
                "gain": "MAE(no_graph) - MAE(graph), averaged over the 50 targets",
                "threshold": ablation.MAE_GAIN_THRESHOLD,
                "significance": "two-sided Wilcoxon signed-rank p < 0.05",
                "breadth": "at least 60% of targets not worse for the graph variant",
            },
            "predictive_spearman": {
                "gain": "rho(graph) - rho(no_graph), averaged over the 50 targets",
                "threshold": ablation.SPEARMAN_GAIN_THRESHOLD,
                "significance": "two-sided Wilcoxon signed-rank p < 0.05",
                "breadth": "at least 60% of targets not worse for the graph variant",
            },
            "stability": {
                "jaccard_delta": ablation.STABILITY_JACCARD_DELTA,
                "frequency_delta": ablation.STABILITY_FREQ_DELTA,
                "note": "either margin counts as a stability improvement",
            },
            "verdict_mapping": {
                "reliable_gain": "predictive reliable AND stability reliable",
                "partial_gain": "exactly one of the two axes reliable",
                "no_reliable_gain": "neither axis reliable",
            },
        },
        "interpretation_limits": [
            "Jaccard measures similarity only; it does not establish which ranking is correct.",
            "Predictive utility is not causal evidence.",
            "The delivered table uses the full-sample 2.csv graph; the predictive check rebuilds the graph inside each training fold so held-out beers never inform the graph.",
            "The existing threshold and resolution sensitivity outputs are reused as-is and are not re-run here.",
            "No parameter is tuned to favour the graph variant; 'no_reliable_gain' is an acceptable outcome.",
        ],
        "reproduction": "python scripts/q2_ablation_20261028.py",
        "inputs": context["inputs"],
        "parameters": context["parameters"],
    }


def figure(output: Path, stability: pd.DataFrame, predictive: pd.DataFrame, overlap: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_path = output / "fig_q2_ablation.png"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ["graph w=0.5", "no graph w=0"]
    frequency_means = stability["freq_mean"].to_numpy()
    jaccard_means = stability["bootstrap_jaccard_mean"].to_numpy()
    positions = np.arange(2)
    axes[0].bar(positions - 0.2, frequency_means, width=0.4, label="bootstrap freq mean")
    axes[0].bar(positions + 0.2, jaccard_means, width=0.4, label="bootstrap Jaccard mean")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Bootstrap stability by variant")
    axes[0].legend(fontsize=8)
    differences = predictive["mae_gain"].to_numpy()
    axes[1].hist(differences, bins=12, color="#4c72b0", edgecolor="white")
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set_title("Per-target MAE gain (no_graph - graph)")
    axes[1].set_xlabel("positive = graph variant better")
    jaccards = overlap["jaccard"].to_numpy()
    axes[2].hist(jaccards, bins=np.linspace(0, 1, 11), color="#dd8452", edgecolor="white")
    axes[2].axvline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_title("Top-3 Jaccard (graph vs no graph)")
    axes[2].set_xlabel("Jaccard")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def run(arguments) -> dict:
    output: Path = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    official = load_official_module()
    config = official.load_config(arguments.config)
    source = resolve_source(arguments, official)

    parameters = {
        "ridge_alpha": float(config.get("ridge_alpha", 1000.0)),
        "graph_threshold": float(config.get("graph_threshold", 0.5)),
        "neighbour_weight": float(config.get("neighbour_weight", 0.5)),
        "ablation_weight": ablation.ABLATION_WEIGHT,
        "bootstrap_repeats": int(config.get("bootstrap", DEFAULT_BOOTSTRAP)),
        "bootstrap_seed": int(config.get("seed", DEFAULT_SEED)),
        "excluded_features": list(config.get("excluded_features", DEFAULT_EXCLUDED)),
        "predictive_folds": ablation.PREDICTIVE_FOLDS,
        "predictive_seed": ablation.PREDICTIVE_SEED,
        "small_ridge_alpha": ablation.SMALL_RIDGE_ALPHA,
    }

    features = pd.read_csv(source / "1.csv", encoding="utf-8-sig")
    correlations = pd.read_csv(source / "2.csv", index_col=0)
    sensory = pd.read_csv(source / "4.csv", encoding="utf-8-sig")
    fields = [column for column in features.columns if column not in META_COLUMNS]
    targets = [column for column in sensory.columns if column not in META_COLUMNS]
    node_names = [str(name) for name in correlations.index]
    corr_matrix = correlations.to_numpy(dtype=float)

    dictionary = pd.read_csv(FEATURE_DICTIONARY) if FEATURE_DICTIONARY.exists() else pd.DataFrame()
    if not dictionary.empty:
        node_to_field, aliases = official.build_alias_map(dictionary, node_names, fields)
        names = dict(zip(dictionary["field_name_R"], dictionary["chemical_name"]))
    else:
        node_to_field, aliases = {}, {}
        names = {}
    cross_values, _, _, _ = cross_correlation_lookup(source, targets, aliases, fields)

    def display_name(field: str) -> str:
        value = names.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return as_compound_label(field)
        return str(value)

    candidates = candidate_features(fields, excluded=parameters["excluded_features"])
    candidate_index = [fields.index(name) for name in candidates]
    adjacency, _ = adjacency_from_corr(corr_matrix, parameters["graph_threshold"])
    candidate_adjacency = adjacency_subset(adjacency, candidate_index)

    X = features[candidates].to_numpy(dtype=float)
    Y = sensory[targets].to_numpy(dtype=float)
    standardised, centred = preprocess(X, Y)
    beta = ridge_coefficients(standardised, centred, parameters["ridge_alpha"])

    scores = {
        "graph": neighbourhood_scores(beta, candidate_adjacency, parameters["neighbour_weight"]),
        "no_graph": neighbourhood_scores(beta, candidate_adjacency, ablation.ABLATION_WEIGHT),
    }
    selected = {variant: top_k_fields(scores[variant], candidates, 3) for variant in scores}
    replicates = {
        variant: bootstrap_top_k(
            X,
            Y,
            candidates,
            candidate_adjacency,
            alpha=parameters["ridge_alpha"],
            n_bootstrap=parameters["bootstrap_repeats"],
            seed=parameters["bootstrap_seed"],
            weight=parameters["neighbour_weight"] if variant == "graph" else ablation.ABLATION_WEIGHT,
        )
        for variant in scores
    }

    commit = official.git_commit()
    inputs = {
        name: sha256(source / name) for name in ("1.csv", "2.csv", "4.csv", "6.csv")
    }
    context = {"commit": commit, "inputs": inputs, "parameters": parameters}
    write_json(output / "protocol.json", build_protocol(context))

    rows = []
    slot_frequencies = {variant: [] for variant in scores}
    bootstrap_means = {variant: [] for variant in scores}
    tiers = {variant: [dict() for _ in targets] for variant in scores}
    frequencies = {variant: [] for variant in scores}
    for target_index, target in enumerate(targets):
        per_target = {}
        for variant in ("graph", "no_graph"):
            chosen = selected[variant][target_index]
            freq = selection_frequencies(
                replicates[variant][target_index], chosen, parameters["bootstrap_repeats"]
            )
            boot = bootstrap_jaccard(replicates[variant][target_index], chosen)
            frequencies[variant].append(freq)
            slot_frequencies[variant].extend(freq[field] for field in chosen)
            bootstrap_means[variant].append(boot["mean"])
            grades = {}
            for rank, field in enumerate(chosen, start=1):
                coefficient = float(beta[candidates.index(field), target_index])
                value = cross_values.get((target, field))
                grade = evidence_grade(
                    target,
                    field,
                    coefficient,
                    value,
                    freq[field],
                    display_name=display_name(field),
                )
                tiers[variant][target_index][field] = TIER_NUMBER[grade["chemical_evidence"]]
                rows.append(
                    {
                        "variant": variant,
                        "target": target,
                        "rank": rank,
                        "field": field,
                        "chemical_name": display_name(field),
                        "ridge_coef": coefficient,
                        "graph_importance": float(scores[variant][candidates.index(field), target_index]),
                        "importance_rank": int(
                            np.argsort(-np.abs(scores[variant][:, target_index]), kind="stable").tolist().index(candidates.index(field))
                        )
                        + 1,
                        "bootstrap_freq": float(freq[field]),
                        "bootstrap_jaccard_vs_full": float(boot["mean"]),
                        "r6_cross_corr": float(value) if value is not None and not np.isnan(value) else np.nan,
                        "chemical_evidence": grade["chemical_evidence"],
                        "evidence_basis": grade["evidence_basis"],
                        "confidence": grade["confidence"],
                    }
                )

    top3 = pd.DataFrame(rows)

    overlap_rows = []
    for target_index, target in enumerate(targets):
        overlap_rows.append(
            ablation.overlap_row(
                target,
                selected["graph"][target_index],
                selected["no_graph"][target_index],
                scores["graph"][:, target_index],
                scores["no_graph"][:, target_index],
                frequencies["graph"][target_index],
                frequencies["no_graph"][target_index],
                candidates,
                tiers["graph"][target_index],
                tiers["no_graph"][target_index],
            )
        )
    overlap = pd.DataFrame(overlap_rows)

    stability = pd.DataFrame(
        [
            ablation.stability_row(
                variant,
                slot_frequencies[variant],
                bootstrap_means[variant],
                len(targets),
            )
            for variant in ("graph", "no_graph")
        ]
    )

    folds = ablation.grouped_folds(len(features), ablation.PREDICTIVE_FOLDS, ablation.PREDICTIVE_SEED)
    predictions_graph, selections_graph = ablation.grouped_cv_predictions(
        X, Y, parameters["neighbour_weight"], folds, alpha=parameters["ridge_alpha"]
    )
    predictions_no_graph, _ = ablation.grouped_cv_predictions(
        X, Y, ablation.ABLATION_WEIGHT, folds, alpha=parameters["ridge_alpha"]
    )
    metrics_graph = ablation.predictive_metrics(Y, predictions_graph)
    metrics_no_graph = ablation.predictive_metrics(Y, predictions_no_graph)

    mae_graph = np.asarray(metrics_graph["mae"])
    mae_no_graph = np.asarray(metrics_no_graph["mae"])
    rho_graph = np.asarray(metrics_graph["spearman"])
    rho_no_graph = np.asarray(metrics_no_graph["spearman"])

    mae_test = ablation.paired_test(mae_no_graph, mae_graph)
    spearman_test = ablation.paired_test(rho_graph, rho_no_graph)
    mae_stats = {
        "gain_mean": mae_test["mean_difference"],
        "p_value": mae_test["p_value"],
        "ci95_low": mae_test["ci95_low"],
        "ci95_high": mae_test["ci95_high"],
        "graph_non_worse_fraction": float(np.mean(mae_graph <= mae_no_graph)),
        "n_graph_better": int(np.sum(mae_graph < mae_no_graph)),
        "n_no_graph_better": int(np.sum(mae_no_graph < mae_graph)),
        "n_ties": int(np.sum(mae_graph == mae_no_graph)),
    }
    spearman_stats = {
        "gain_mean": spearman_test["mean_difference"],
        "p_value": spearman_test["p_value"],
        "ci95_low": spearman_test["ci95_low"],
        "ci95_high": spearman_test["ci95_high"],
        "graph_non_worse_fraction": float(np.mean(rho_graph >= rho_no_graph)),
        "n_graph_better": int(np.sum(rho_graph > rho_no_graph)),
        "n_no_graph_better": int(np.sum(rho_no_graph > rho_graph)),
        "n_ties": int(np.sum(rho_graph == rho_no_graph)),
    }
    stability_stats = {
        "graph_jaccard_mean": float(stability.loc[stability["variant"] == "graph", "bootstrap_jaccard_mean"].iloc[0]),
        "no_graph_jaccard_mean": float(stability.loc[stability["variant"] == "no_graph", "bootstrap_jaccard_mean"].iloc[0]),
        "graph_freq_mean": float(stability.loc[stability["variant"] == "graph", "freq_mean"].iloc[0]),
        "no_graph_freq_mean": float(stability.loc[stability["variant"] == "no_graph", "freq_mean"].iloc[0]),
        "graph_jaccard_min": float(stability.loc[stability["variant"] == "graph", "bootstrap_jaccard_min"].iloc[0]),
        "no_graph_jaccard_min": float(stability.loc[stability["variant"] == "no_graph", "bootstrap_jaccard_min"].iloc[0]),
    }
    verdict = ablation.ablation_verdict(mae_stats, spearman_stats, stability_stats)

    predictive = pd.DataFrame(
        {
            "target": targets,
            "mae_graph": mae_graph,
            "mae_no_graph": mae_no_graph,
            "mae_gain": mae_no_graph - mae_graph,
            "spearman_graph": rho_graph,
            "spearman_no_graph": rho_no_graph,
            "spearman_gain": rho_graph - rho_no_graph,
            "graph_better_mae": mae_graph < mae_no_graph,
            "graph_better_spearman": rho_graph > rho_no_graph,
            "top3_jaccard": overlap["jaccard"].to_numpy(),
        }
    )

    main_fields = {}
    if MAIN_TABLE.exists():
        delivered = pd.read_csv(MAIN_TABLE)
        for record in delivered.itertuples(index=False):
            main_fields[(str(record.target), int(record.rank))] = str(record.field)
    graph_rows = top3[top3["variant"] == "graph"].sort_values(["target", "rank"])
    mismatch = [
        {"target": record.target, "rank": int(record.rank), "field": record.field}
        for record in graph_rows.itertuples(index=False)
        if main_fields.get((str(record.target), int(record.rank))) != str(record.field)
    ]
    no_graph_rows = top3[top3["variant"] == "no_graph"]
    importance_error = float(
        np.max(np.abs(no_graph_rows["graph_importance"].to_numpy() - np.abs(no_graph_rows["ridge_coef"].to_numpy())))
    )
    rerun = bootstrap_top_k(
        X,
        Y,
        candidates,
        candidate_adjacency,
        alpha=parameters["ridge_alpha"],
        n_bootstrap=parameters["bootstrap_repeats"],
        seed=parameters["bootstrap_seed"],
        weight=ablation.ABLATION_WEIGHT,
    )
    rerun_identical = all(
        list(rerun[target_index]) == list(replicates["no_graph"][target_index])
        for target_index in range(len(targets))
    )
    checks = {
        "overlap_rows_50": len(overlap) == 50,
        "stability_rows_2": len(stability) == 2,
        "predictive_rows_50": len(predictive) == 50,
        "top3_rows_300": len(top3) == 300,
        "targets_50": len(set(top3["target"])) == 50,
        "three_slots_unique": all(
            len(set(group["field"])) == 3 for _, group in top3.groupby(["variant", "target"])
        ),
        "graph_variant_equals_main_table": bool(main_fields) and not mismatch,
        "no_graph_importance_equals_abs_ridge_coef": importance_error <= 1e-12,
        "bootstrap_rerun_identical": rerun_identical,
        "predictive_folds_5": int(np.unique(folds).size) == ablation.PREDICTIVE_FOLDS,
        "no_missing_core_values": bool(
            not top3[["graph_importance", "bootstrap_freq", "bootstrap_jaccard_vs_full"]].isna().any().any()
        ),
    }
    verification = {
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "detail": {
            "graph_vs_main_mismatch": mismatch[:10],
            "no_graph_importance_max_abs_error": importance_error,
            "predictive_fold_sizes": np.bincount(folds).tolist(),
            "seed": parameters["bootstrap_seed"],
        },
    }

    overlap_summary = {
        "jaccard_mean": float(overlap["jaccard"].mean()),
        "jaccard_median": float(overlap["jaccard"].median()),
        "jaccard_min": float(overlap["jaccard"].min()),
        "overlap_n_mean": float(overlap["overlap_n"].mean()),
        "n_targets_identical": int(np.sum(overlap["overlap_n"] == 3)),
        "n_targets_jaccard_ge_0_5": int(np.sum(overlap["jaccard"] >= 0.5)),
        "rank_spearman_mean": float(overlap["rank_spearman"].mean()),
        "rank_spearman_min": float(overlap["rank_spearman"].min()),
        "top3_rank_spearman_mean": float(overlap["top3_rank_spearman"].mean()),
        "freq_difference_mean": float(overlap["freq_difference"].mean()),
        "evidence_A_graph": int(overlap["graph_A"].sum()),
        "evidence_A_no_graph": int(overlap["no_graph_A"].sum()),
        "evidence_B_graph": int(overlap["graph_B"].sum()),
        "evidence_B_no_graph": int(overlap["no_graph_B"].sum()),
        "evidence_C_graph": int(overlap["graph_C"].sum()),
        "evidence_C_no_graph": int(overlap["no_graph_C"].sum()),
    }

    summary = {
        "protocol_id": "q2_ablation_20261028",
        "baseline_commit": commit,
        "purpose": "graph-neighbourhood ablation (w=0.5 vs w=0.0) for the Q2 Top-3 table",
        "targets": len(targets),
        "candidates": len(candidates),
        "weights": {"graph": parameters["neighbour_weight"], "no_graph": ablation.ABLATION_WEIGHT},
        "overlap_summary": overlap_summary,
        "stability": stability.to_dict(orient="records"),
        "predictive": {
            "mae": mae_stats,
            "spearman": spearman_stats,
            "folds": ablation.PREDICTIVE_FOLDS,
            "fold_seed": ablation.PREDICTIVE_SEED,
            "small_ridge_alpha": ablation.SMALL_RIDGE_ALPHA,
            "graph_rebuilt_per_fold": True,
        },
        "verdict": verdict,
        "boundaries": [
            "50 targets x 3 compounds; 219 candidate compounds; alpha=1000; graph threshold 0.5",
            "bootstrap: 200 beer-level resamples, seed 20261027, identical draws for both variants",
            "predictive: 5 beer-level folds, seed 20261028, graph rebuilt from training beers only",
            "descriptive Top-3 table keeps the full-sample 2.csv graph (as delivered)",
            "Jaccard is a similarity measure only; held-out predictive utility is not causal evidence",
        ],
        "verification": verification,
        "inputs_sha256": inputs,
    }

    save_csv(output, "q2__ablation_overlap.csv", overlap)
    save_csv(output, "q2__ablation_stability.csv", stability)
    save_csv(output, "q2__ablation_predictive.csv", predictive)
    save_csv(output, "q2__ablation_top3.csv", top3)
    write_json(output / "summary.json", summary)
    write_json(output / "verification.json", verification)
    figure(output, stability, predictive, overlap)

    print(
        "[q2-ablation] "
        f"overlap_jaccard_mean={overlap_summary['jaccard_mean']:.3f} "
        f"identical_targets={overlap_summary['n_targets_identical']} "
        f"freq_mean graph={stability_stats['graph_freq_mean']:.4f} no_graph={stability_stats['no_graph_freq_mean']:.4f} "
        f"boot_jaccard graph={stability_stats['graph_jaccard_mean']:.4f} no_graph={stability_stats['no_graph_jaccard_mean']:.4f} "
        f"mae_gain={mae_stats['gain_mean']:.4f} (p={mae_stats['p_value']:.3g}) "
        f"rho_gain={spearman_stats['gain_mean']:.4f} (p={spearman_stats['p_value']:.3g}) "
        f"verdict={verdict['verdict']} "
        f"all_pass={verification['all_pass']}"
    )
    return summary


def main(argv=None) -> int:
    arguments = parse_args(argv)
    summary = run(arguments)
    return 0 if summary["verification"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
