"""Q3 cross-style flavour migration (problem-statement semantics).

What this module does
---------------------
Migrates a source beer's attachment-1 compound spectrum toward a target style,
then certifies that the **listed defect compounds** did not get worse, using the
wording of the problem statement: *"保持缺陷化合物不恶化 ... 注：缺陷化合物以所列
DMS/Diacetyl/4VG/trans-2-nonenal/Stale hops 等超阈值项为准"*.  Because the note says
"超阈值项为准", the migration is certified under three explicit readings and all three
are reported rather than only the strictest one:

``A`` (set form, literal note)  the set of over-threshold defects does not expand and
                                the severity of each over-threshold defect does not rise.
``B`` (per-defect severity)     ``e_after <= e_before`` for every defect, with
                                ``e(c, T) = max(c / T - 1, 0)``.
``C`` (aggregated severity)     ``sum_d e_after <= sum_d e_before``.

Criteria are evaluated under both the ``median`` and the ``mean`` sensory threshold
scenario from ``8.xlsx``; ``median`` is the primary scenario (it is the column used by
``scripts/threshold_scenarios.py`` in this repository).

Coordinate system and units
---------------------------
Input is ``1.csv`` exactly as supplied: 250 beers x 231 compound coordinates plus
``beer`` / ``beer_id`` / ``tasting_category_fine``.  Most coordinates are already
log10-transformed by the data authors; four studied defects are:

===============  ==========================  ========  ==========================
defect           column                       units      threshold (8.xlsx row)
===============  ==========================  ========  ==========================
DMS              ``DMS``                      ug/L      ``DMS``
Diacetyl         ``X2.3.butanedione``         mg/L      ``2,3-butanedione``
4VG              ``X4VG``                     mg/L      ``X4VG``
trans-2-nonenal  ``X.E..2.nonenal``           mg/L      *(no threshold row)*
===============  ==========================  ========  ==========================

``DMS`` and ``X4VG`` are log10 coordinates; diacetyl and trans-2-nonenal are raw
concentrations.  The per-column log10/raw convention is inferred from the
``Calibrated chemical dataset`` sheet of ``8.xlsx`` (Pearson correlation of
``10**coordinate`` vs the calibrated value, compared with the raw correlation), and the
four defect columns are then pinned to the convention that the calibrated sheet and the
domain units agree on.  trans-2-nonenal has no threshold row in ``8.xlsx``, so it is
only checked for "does not get worse"; the same holds for the six ``Sensory thresholds``
Stale-hops *proxy* oxidation coordinates listed in :data:`STALE_HOP_PROXIES` (the
dataset has no "stale hops" coordinate and no stale-hops threshold).

Style model
-----------
``tasting_category_fine`` is *not* chemically separable at this resolution: the module
reports leave-one-out nearest-centroid accuracy in a whitened 8-D global-PCA space as a
diagnostic (about 0.13 on the supplied data, versus a 0.12 majority-class baseline), so
the "migrated spectrum is now a Tripel" claim must be read as "closer to the Tripel
centroid", not as a reliable style classifier.  The same geometry is used to choose the
migration strength.

Migration and projection
------------------------
``z(lam) = (1 - lam) * z_source + lam * mean_target`` on the standardised coordinates
(``lam`` on a 0.05 grid, 20 candidates per (source, target) pair, 660 in total), followed
by a hard safety projection::

    threshold defect      ->  coordinate <= min(source, 0.9 * threshold)
    no-threshold defect   ->  coordinate <= source
    stale-hop proxies     ->  coordinate <= source

``lam*`` is the largest ``lam`` whose migrated spectrum is still nearest to the target
centroid, falling back to the ``lam`` with the smallest target-centroid distance.  The
0.9 headroom is a modelling choice, not a measured safety limit.

Provenance and honesty
----------------------
All numbers come from the supplied attachments; nothing is invented.  A migrated
spectrum is a *statistical destination* in the supplied coordinate space, not a brewing
recipe and not a measured beer: the projection certifies only the listed defect
coordinates, and it does not certify that the result is physically brewable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

META_COLUMNS = ("beer", "beer_id", "tasting_category_fine")
STYLE_COLUMN = "tasting_category_fine"
BEER_COLUMN = "beer"
SOURCE_STYLE = "Pils/Lager"
TARGET_STYLES = ("Tripel", "Kriek", "Lambic")
LAMBDA_GRID = tuple(round(0.05 * step, 3) for step in range(1, 21))  # 0.05..1.00
N_COMPONENTS = 8
DEFECT_HEADROOM = 0.9
THRESHOLD_SCENARIOS = ("median", "mean")
PRIMARY_SCENARIO = "median"
MIN_NON_MISSING_FRACTION = 0.75

# The official Q1/Q3 52-target input uses 224 of the 231 attachment-1 coordinates; the seven
# excluded columns are exactly ``excluded_features`` of
# ``configs/r2_raw_panel_protocol_20261026.json`` (summary/"derived bookkeeping" columns).
OFFICIAL_EXCLUDED_FEATURES = ("GFS", "unfermentables", "kcalperc", "esters_sum",
                              "aroma_hops_sum", "acids_sum", "sulfur_sum")
CASE_ID_TEMPLATE = "S{beer:03d}_TO_{style}"
MIGRATED_METHOD = "q3_migration_projected"
DEFECT_STATUSES = ("improved", "unchanged", "worse", "unknown")
CLAIM_LEVELS = ("model_supported", "conditional", "undecidable")


def case_id(beer: int, target_style: str) -> str:
    """Stable identifier shared by the spectra, the sensory profile and the evidence tables."""
    return CASE_ID_TEMPLATE.format(beer=int(beer), style=target_style)


def defect_status(before: float | None, after: float | None,
                  threshold: float | None) -> str:
    """Severity-based before/after status of one listed defect.

    ``e(c, T) = max(c / T - 1, 0)`` is the same severity the A/B/C criteria use.  A defect
    without a threshold (trans-2-nonenal) or without a coordinate (Stale hops) is
    ``unknown`` rather than "unchanged": the note ties the claim to over-threshold items and
    those two items cannot be evaluated against a threshold at all.
    """
    if threshold is None or before is None or after is None:
        return "unknown"
    if not np.isfinite(before) or not np.isfinite(after):
        return "unknown"
    e_before = max(float(before) / float(threshold) - 1.0, 0.0)
    e_after = max(float(after) / float(threshold) - 1.0, 0.0)
    if e_after < e_before - 1e-12:
        return "improved"
    if e_after > e_before + 1e-12:
        return "worse"
    return "unchanged"


def _number(value) -> float:
    """``float`` or NaN, so graded tables can carry missing source values honestly."""
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)



@dataclass(frozen=True)
class DefectSpec:
    """One listed defect compound: its attachment column, transform, unit, threshold row."""

    name: str
    column: str
    transform: str  # "log10" | "identity"
    unit: str
    threshold_field: str | None


DEFECT_SPECS = (
    DefectSpec("DMS", "DMS", "log10", "ug/L", "DMS"),
    DefectSpec("Diacetyl", "X2.3.butanedione", "identity", "mg/L", "2,3-butanedione"),
    DefectSpec("4VG", "X4VG", "log10", "mg/L", "X4VG"),
    DefectSpec("t2n", "X.E..2.nonenal", "identity", "mg/L", None),
)
DEFECT_COLUMNS = {spec.name: spec.column for spec in DEFECT_SPECS}

# The attachments contain no "stale hops" coordinate and 8.xlsx has no stale-hops
# threshold; these hop-oxidation products are the closest measured proxies, so they are
# constrained only by "do not get worse" and reported separately.
STALE_HOP_PROXIES = (
    "humulene.epoxide.II",
    "caryophyllene.oxide",
    "dihydroisocaryophyllene.epoxide",
    "X4.8.alpha.epoxy.caryophyllane",
    "X14.hydroxy.alpha.humulene",
    "X14.hydroxy.9.epi..E..caryophyllene",
)

CRITERIA = {
    "A": ("set form of the problem note: the set of over-threshold defects does not expand "
          "and the severity of each over-threshold defect does not increase"),
    "B": "per-defect severity e_after <= e_before with e(c, T) = max(c / T - 1, 0)",
    "C": "aggregated severity sum_d e_after <= sum_d e_before",
}

HONESTY = (
    "A migrated spectrum is a statistical destination inside the supplied coordinate "
    "space, not a brewing recipe and not a measured beer; the projection certifies only "
    "the listed defect coordinates at the 0.9 x threshold headroom and does not certify "
    "physical brewability or sensory acceptance."
)


def concentration(coord, transform: str) -> np.ndarray:
    """Coordinate -> physical concentration for the given per-column transform."""
    values = np.asarray(coord, dtype=float)
    return np.power(10.0, values) if transform == "log10" else values


def coordinate(value, transform: str):
    """Physical concentration -> coordinate (inverse of :func:`concentration`)."""
    values = np.asarray(value, dtype=float)
    return np.log10(values) if transform == "log10" else values


def excess_severity(measured, threshold):
    """``e(c, T) = max(c / T - 1, 0)``; NaN concentration -> NaN severity."""
    measured = np.asarray(measured, dtype=float)
    return np.where(np.isfinite(measured), np.maximum(measured / threshold - 1.0, 0.0), np.nan)


def load_thresholds(reference: Path) -> dict[str, dict[str, float | None]]:
    """Read the DMS / diacetyl / 4VG thresholds from ``8.xlsx`` ``Sensory thresholds``.

    Returns ``{defect name: {"median": value, "mean": value}}``; trans-2-nonenal has no
    row in the sheet and therefore maps to ``{"median": None, "mean": None}``.
    """
    sheet = pd.read_excel(reference, sheet_name="Sensory thresholds")
    fields = sheet["field_name_python"].astype(str).str.strip()

    def value(field: str | None, column: str) -> float | None:
        if field is None:
            return None
        match = sheet.loc[fields == field, column]
        if match.empty or pd.isna(match.iloc[0]):
            return None
        return float(match.iloc[0])

    return {spec.name: {scenario: value(spec.threshold_field, "threshold_" + scenario)
                        for scenario in THRESHOLD_SCENARIOS}
            for spec in DEFECT_SPECS}


def infer_transforms(raw: pd.DataFrame, calibrated: pd.DataFrame) -> dict[str, str]:
    """Infer the per-column log10/identity convention from ``Calibrated chemical dataset``.

    The calibrated sheet holds the de-transformed values; the transform whose back-transform
    correlates best with it wins (a 0.02 margin avoids coin flips on flat columns).  The four
    listed defects are pinned, because the inferred winner is unstable for sparse columns and
    the units in the problem statement settle it.
    """
    merged = raw.merge(calibrated, on="beer_id", how="inner", suffixes=("", "_calibrated"))
    transforms: dict[str, str] = {}
    for column in [c for c in calibrated.columns
                   if c not in META_COLUMNS and c in raw.columns]:
        if column not in merged.columns or column + "_calibrated" not in merged.columns:
            continue
        pair = merged[[column, column + "_calibrated"]].dropna()
        if len(pair) < 40:
            continue
        log_correlation = np.corrcoef(np.power(10.0, np.clip(
                                          pair[column].to_numpy(float), -150.0, 150.0)),
                                      pair[column + "_calibrated"].to_numpy(float))[0, 1]
        raw_correlation = np.corrcoef(pair[column].to_numpy(float),
                                      pair[column + "_calibrated"].to_numpy(float))[0, 1]
        if not np.isfinite(log_correlation):
            transforms[column] = "identity"
        elif abs(log_correlation) > abs(raw_correlation) + 0.02:
            transforms[column] = "log10"
        else:
            transforms[column] = "identity"
    transforms.update({spec.column: spec.transform for spec in DEFECT_SPECS})
    return transforms


def load_profile_frame(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``1.csv`` and ``8.xlsx`` from the attachment directory."""
    source = Path(source)
    profiles = pd.read_csv(source / "1.csv", low_memory=False)
    reference = pd.read_excel(source / "8.xlsx", sheet_name=None)
    return profiles, reference


@dataclass
class MigrationData:
    """Standardised attachment-1 spectra plus the style geometry needed for migration.

    Build with :meth:`load`.  All arrays are aligned: row ``i`` is ``ids[i]`` with style
    ``styles[i]``; ``features`` names the coordinate columns in order.
    """

    ids: np.ndarray
    styles: np.ndarray
    features: list[str]
    coordinates: np.ndarray          # raw attachment-1 coordinates (imputed)
    standardised: np.ndarray         # (x - median) / robust scale
    center: np.ndarray               # per-column median used for standardisation
    scale: np.ndarray                # per-column robust scale (IQR / 1.349, sd fallback)
    missing_mask: np.ndarray         # (n, p) True where the attachment value was missing
    transforms: dict[str, str]
    thresholds: dict[str, dict[str, float | None]]
    style_list: list[str]
    components: np.ndarray           # (p, k) PCA loadings
    component_scale: np.ndarray      # per-component std used for whitening
    style_centroids: dict[str, np.ndarray]
    source_directory: Path
    raw_provenance: dict | None = None
    # full 231-column attachment-1 panel (imputed) and the official 224-feature subset, needed
    # to export complete spectra for the Q1 sensory pipeline.  ``features`` may be a subset of
    # ``full_features`` (the NaN-ratio filter), so the two are kept side by side.
    full_features: list[str] = field(default_factory=list)
    full_coordinates: np.ndarray | None = None
    official_features: list[str] = field(default_factory=list)
    _feature_index: dict[str, int] = None
    _full_index: dict[str, int] = None

    # ---------------------------------------------------------------- geometry
    def whitened(self, standardised: np.ndarray) -> np.ndarray:
        """Project standardised coordinates into the whitened 8-D style space."""
        offset = self.center  # keep signature explicit; standardised already centred
        del offset
        projected = (standardised - self.standardised.mean(0)) @ self.components
        return projected / self.component_scale

    def centroid_distances(self, standardised: np.ndarray) -> np.ndarray:
        """Distance from one standardised spectrum to every style centroid (whitened space)."""
        point = self.whitened(np.atleast_2d(standardised))[0]
        return np.array([np.linalg.norm(point - self.style_centroids[style])
                         for style in self.style_list])

    def nearest_style(self, standardised: np.ndarray) -> tuple[str, int]:
        distances = self.centroid_distances(standardised)
        index = int(np.argmin(distances))
        return self.style_list[index], index

    def style_accuracy(self) -> dict[str, float]:
        """Leave-one-out style-separability diagnostics (nearest centroid + majority)."""
        whitened = self.whitened(self.standardised)
        hits = 0
        for row, style in enumerate(self.styles):
            best, best_distance = None, np.inf
            for candidate in self.style_list:
                members = np.flatnonzero(self.styles == candidate)
                members = members[members != row]
                if len(members) < 2:
                    continue
                distance = float(np.linalg.norm(whitened[row] - whitened[members].mean(0)))
                if distance < best_distance:
                    best, best_distance = candidate, distance
            hits += int(best == style)
        majority = float(np.bincount(pd.factorize(self.styles)[0]).max() / len(self.styles))
        return {"nearest_centroid_loo_accuracy": hits / len(self.styles),
                "majority_class_baseline": majority,
                "n_classes": int(len(self.style_list))}

    # ------------------------------------------------------------------ defect IO
    def _index(self, column: str) -> int:
        if self._feature_index is None:
            self._feature_index = {name: i for i, name in enumerate(self.features)}
        return self._feature_index[column]

    def defect_concentration(self, standardised: np.ndarray, defect: str) -> float:
        """Physical concentration of one defect from a standardised spectrum."""
        spec = next(s for s in DEFECT_SPECS if s.name == defect)
        index = self._index(spec.column)
        raw = standardised[index] * self.scale[index] + self.center[index]
        return float(concentration(raw, self.transforms.get(spec.column, spec.transform)))

    def defect_coordinate(self, concentration_value: float, defect: str) -> float:
        """Raw coordinate for a physical concentration of one defect."""
        spec = next(s for s in DEFECT_SPECS if s.name == defect)
        return float(coordinate(concentration_value, self.transforms.get(spec.column, spec.transform)))

    def standardise(self, coordinates: np.ndarray) -> np.ndarray:
        return (np.asarray(coordinates, dtype=float) - self.center) / self.scale

    def destandardise(self, standardised: np.ndarray) -> np.ndarray:
        return np.asarray(standardised, dtype=float) * self.scale + self.center

    def physical_cap(self, defect: str, scenario: str = PRIMARY_SCENARIO) -> float | None:
        threshold = self.thresholds[defect][scenario]
        return None if threshold is None else DEFECT_HEADROOM * threshold

    # ---------------------------------------------------------------- projection
    def project(self, standardised: np.ndarray, source_index: int,
                target_style: str) -> tuple[np.ndarray, dict[str, str]]:
        """Hard defect projection: listed defects never above the source, threshold defects
        additionally below ``0.9 * threshold``.

        Missing source values (the supplied data has two missing diacetyl coordinates in
        the Pils/Lager group) are handled conditionally: the cap becomes
        ``min(target-style median, 0.9 * threshold)`` and the case is flagged.
        """
        projected = np.array(standardised, dtype=float, copy=True)
        notes: dict[str, str] = {}
        for spec in DEFECT_SPECS:
            index = self._index(spec.column)
            source_coordinate = self.coordinates[source_index, index]
            cap = self.physical_cap(spec.name)
            if self.missing_mask[source_index, index]:
                target_median = float(np.nanmedian(
                    self.coordinates[self.styles == target_style, index]))
                cap_value = min(target_median, cap) if cap is not None else target_median
                projected[index] = min(projected[index],
                                       self.standardise_value(spec.column, cap_value))
                notes[spec.name] = (f"source missing -> cap=min(target median, {DEFECT_HEADROOM}*T)"
                                    f"={concentration(cap_value, self.transforms[spec.column]):.5g} "
                                    f"{spec.unit} (conditional)")
            else:
                projected[index] = min(projected[index], self.standardise_value(
                    spec.column, source_coordinate))
                if cap is not None:
                    projected[index] = min(projected[index], self.standardise_value(
                        spec.column, self.defect_coordinate(cap, spec.name)))
                notes[spec.name] = (f"cap=min(source, {DEFECT_HEADROOM}*T)"
                                    if cap is not None else "cap=source (no threshold)")
        for proxy in STALE_HOP_PROXIES:
            if proxy not in self.features:
                continue
            index = self._index(proxy)
            source_coordinate = self.coordinates[source_index, index]
            if np.isfinite(source_coordinate):
                projected[index] = min(projected[index], self.standardise_value(
                    proxy, source_coordinate))
        return projected, notes

    def standardise_value(self, column: str, value: float) -> float:
        index = self._index(column)
        return float((value - self.center[index]) / self.scale[index])

    # -------------------------------------------------------------- full spectra
    def official_spectrum(self, row: int,
                          standardised: np.ndarray | None = None) -> np.ndarray:
        """The complete official 224-feature spectrum of one beer.

        With ``standardised`` given, the migrated (destandardised) coordinates replace the
        source coordinates for every migration feature and the other official features keep
        their source value; without it the source spectrum is returned (``role="before"``).
        """
        if not self.official_features or self.full_coordinates is None:
            raise ValueError("MigrationData was not built by load(); no full spectra available")
        full = np.array(self.full_coordinates[row], dtype=float, copy=True)
        if standardised is not None:
            migrated = self.destandardise(np.asarray(standardised, dtype=float))
            if migrated.shape[0] != len(self.features):
                raise ValueError("standardised spectrum length does not match migration features")
            for position, name in enumerate(self.features):
                full[self._full_index[name]] = migrated[position]
        return np.array([full[self._full_index[name]] for name in self.official_features],
                        dtype=float)

    def migration_spectra(self, targets=TARGET_STYLES) -> dict[str, pd.DataFrame]:
        """Complete before/after spectra and the 33-row Q1 input table for every case.

        Returns ``{"long": ..., "wide": ..., "after_input": ...}``:

        ``long``       66 rows (33 cases x before/after) with ``case_id``, ``source_beer``,
                       ``target_style``, ``lambda``, ``role`` and the 224 official features
        ``wide``       33 rows, ``..._before`` / ``..._after`` side by side (448 feature columns)
        ``after_input``33 rows in the official Q1 input layout
                       (``generated_id``, ``style``, ``method`` + 224 features, official order)
        """
        long_rows, wide_rows, after_rows = [], [], []
        for target in targets:
            for source_index in np.flatnonzero(self.styles == SOURCE_STYLE):
                migration = self.migrate(int(source_index), target)
                beer = int(self.ids[source_index])
                identifier = case_id(beer, target)
                before = self.official_spectrum(int(source_index))
                after = self.official_spectrum(int(source_index), migration["standardised"])
                meta = {"case_id": identifier, "source_beer": beer, "target_style": target,
                        "lambda": float(migration["lambda"])}
                long_rows.append({**meta, "role": "before",
                                  **dict(zip(self.official_features, before.tolist()))})
                long_rows.append({**meta, "role": "after",
                                  **dict(zip(self.official_features, after.tolist()))})
                wide_rows.append({**meta,
                                  "dist_reduction_pct": float(migration["distance_reduction_pct"]),
                                  "style_before": migration["style_before"],
                                  "style_after": migration["style_after"],
                                  **{f"{name}_before": float(before[position])
                                     for position, name in enumerate(self.official_features)},
                                  **{f"{name}_after": float(after[position])
                                     for position, name in enumerate(self.official_features)}})
                after_rows.append({"generated_id": identifier, "style": target,
                                   "method": MIGRATED_METHOD,
                                   **dict(zip(self.official_features, after.tolist()))})
        long_columns = ["case_id", "source_beer", "target_style", "lambda", "role",
                        *self.official_features]
        after_columns = ["generated_id", "style", "method", *self.official_features]
        return {"long": pd.DataFrame(long_rows, columns=long_columns),
                "wide": pd.DataFrame(wide_rows),
                "after_input": pd.DataFrame(after_rows, columns=after_columns)}

    # ------------------------------------------------------------ evidence grading
    def defect_statuses(self, row) -> dict[str, str]:
        """Five-defect before/after status for one result row (severity based)."""
        statuses = {}
        for spec in DEFECT_SPECS:
            before = row[f"{spec.name}_before"]
            after = row[f"{spec.name}_after"]
            statuses[spec.name] = defect_status(
                None if pd.isna(before) else float(before),
                None if pd.isna(after) else float(after),
                self.thresholds[spec.name][PRIMARY_SCENARIO])
        # the dataset has no stale-hops coordinate (only hop-oxidation proxies), so the
        # listed "Stale hops" item cannot be evaluated against a threshold at all
        statuses["Stale hops"] = "unknown"
        return statuses

    def evidence_table(self, results: pd.DataFrame | None = None) -> pd.DataFrame:
        """Per-case evidence grading: chemical criteria, style diagnostic, defect status, claim.

        ``claim_level`` is assigned by the rules written into ``protocol.json`` before the run:

        ``model_supported``  A/B/C pass, the migrated spectrum is nearest to the target
                             centroid, the target style's own diagnostic supports the claim
                             (all 11 cases nearest to that centroid), no evaluated defect got
                             worse, and no conditional cap was used.
        ``conditional``      A/B/C pass but at least one of (style diagnostic unsupported,
                             an evaluated defect worsened, a conditional diacetyl cap was
                             used); the migration claim then rests on the chemical criteria.
        ``undecidable``      A/B/C do not all pass.

        trans-2-nonenal (no threshold in 8.xlsx) and Stale hops (no coordinate) are
        ``unknown`` and are outside the pre-registered A/B/C scope; this is stated in the
        table and in the paper-facing caveats.
        """
        if results is None:
            results, _ = self.run()
        accuracy = self.style_accuracy()
        support = {style: int(results.loc[results["target_style"] == style,
                                         "style_is_target"].sum()) for style in TARGET_STYLES}
        total = int((self.styles == SOURCE_STYLE).sum())
        rows = []
        for _, row in results.iterrows():
            target = row["target_style"]
            beer = int(row["source_beer"])
            statuses = self.defect_statuses(row)
            style_after = row["style_after"]
            is_target = bool(row["style_is_target"])
            if is_target:
                nearest = "target"
            elif style_after == SOURCE_STYLE:
                nearest = "source_baseline"
            else:
                nearest = "other"
            n_support = support.get(target, 0)
            supported = bool(is_target and n_support == total)
            chemical = bool(row["criteria_A"] and row["criteria_B"] and row["criteria_C"])
            worsened = [name for name in DEFECT_SPECS
                        if statuses[name.name] == "worse"]
            conditional = bool(row["conditional_" + PRIMARY_SCENARIO])
            if not chemical:
                level = "undecidable"
                note = "A/B/C 化学判据未全部通过，迁移不可判定"
            elif supported and not worsened and not conditional:
                level = "model_supported"
                note = ("A/B/C 通过；最近质心为目标风格且该风格诊断 %d/%d；已评估缺陷无恶化；"
                        "注意全库风格可辨识度弱（LOO %.3f ≈ 多数类 %.3f），该支持仅指目标风格"
                        "质心诊断" % (n_support, total,
                                       accuracy["nearest_centroid_loo_accuracy"],
                                       accuracy["majority_class_baseline"]))
            else:
                reasons = []
                if not supported:
                    if not is_target:
                        reasons.append("迁移后最近质心为 %s 而非目标风格" % style_after)
                    reasons.append("目标风格最近质心仅 %d/%d，全库最近质心 LOO %.3f ≈ 多数类 "
                                   "%.3f，风格诊断未支持" % (
                                       n_support, total,
                                       accuracy["nearest_centroid_loo_accuracy"],
                                       accuracy["majority_class_baseline"]))
                if worsened:
                    reasons.append("已评估缺陷恶化: " + ",".join(name.name for name in worsened))
                if conditional:
                    reasons.append("源双乙酰缺失，采用 min(目标中位, 0.9T) 条件上限")
                level = "conditional"
                note = "；".join(reasons)
            rows.append({
                "case_id": case_id(beer, target),
                "source_beer": beer,
                "target_style": target,
                "lambda": float(row["lambda"]),
                "chemical_pass": chemical,
                "chemical_pass_A": bool(row["criteria_A"]),
                "chemical_pass_B": bool(row["criteria_B"]),
                "chemical_pass_C": bool(row["criteria_C"]),
                "chemical_pass_A_mean": bool(row["A_mean"]),
                "chemical_pass_B_mean": bool(row["B_mean"]),
                "chemical_pass_C_mean": bool(row["C_mean"]),
                "dist_reduction_pct": float(row["dist_reduction_pct"]),
                "style_nearest_centroid": nearest,
                "style_nearest_style": style_after,
                "style_is_target": is_target,
                "style_diagnostic_supported": supported,
                "style_diagnostic_cases": n_support,
                "style_diagnostic_total": total,
                "style_diagnostic_note": ("目标风格最近质心 %d/%d；全库最近质心 LOO %.3f vs "
                                          "多数类 %.3f（%d 类）" % (
                                              n_support, total,
                                              accuracy["nearest_centroid_loo_accuracy"],
                                              accuracy["majority_class_baseline"],
                                              accuracy["n_classes"])),
                "defect_status_DMS": statuses["DMS"],
                "defect_status_Diacetyl": statuses["Diacetyl"],
                "defect_status_4VG": statuses["4VG"],
                "defect_status_t2n": statuses["t2n"],
                "defect_status_Stale_hops": statuses["Stale hops"],
                "DMS_ugL_before": _number(row["DMS_before"]),
                "DMS_ugL_after": _number(row["DMS_after"]),
                "Diacetyl_mgL_before": _number(row["Diacetyl_before"]),
                "Diacetyl_mgL_after": _number(row["Diacetyl_after"]),
                "4VG_mgL_before": _number(row["4VG_before"]),
                "4VG_mgL_after": _number(row["4VG_after"]),
                "t2n_mgL_before": _number(row["t2n_before"]),
                "t2n_mgL_after": _number(row["t2n_after"]),
                "defects_worse_count": int(len(worsened)),
                "conditional": conditional,
                "claim_level": level,
                "claim_note": note,
            })
        columns = ["case_id", "source_beer", "target_style", "lambda",
                   "chemical_pass", "chemical_pass_A", "chemical_pass_B", "chemical_pass_C",
                   "chemical_pass_A_mean", "chemical_pass_B_mean", "chemical_pass_C_mean",
                   "dist_reduction_pct", "style_nearest_centroid", "style_nearest_style",
                   "style_is_target", "style_diagnostic_supported", "style_diagnostic_cases",
                   "style_diagnostic_total", "style_diagnostic_note",
                   "defect_status_DMS", "defect_status_Diacetyl", "defect_status_4VG",
                   "defect_status_t2n", "defect_status_Stale_hops",
                   "DMS_ugL_before", "DMS_ugL_after", "Diacetyl_mgL_before", "Diacetyl_mgL_after",
                   "4VG_mgL_before", "4VG_mgL_after", "t2n_mgL_before", "t2n_mgL_after",
                   "defects_worse_count", "conditional", "claim_level", "claim_note"]
        return pd.DataFrame(rows, columns=columns)

    # ------------------------------------------------------------------ criteria
    def defect_table(self, source_index: int, standardised: np.ndarray) -> dict[str, dict]:
        """Before/after physical concentrations for every listed defect."""
        table = {}
        for spec in DEFECT_SPECS:
            index = self._index(spec.column)
            before_coordinate = self.coordinates[source_index, index]
            missing = bool(self.missing_mask[source_index, index])
            after = self.defect_concentration(standardised, spec.name)
            table[spec.name] = {
                "before": None if missing else float(
                    concentration(before_coordinate, self.transforms[spec.column])),
                "after": after,
                "missing_source": missing,
            }
        return table

    def criteria(self, source_index: int, standardised: np.ndarray,
                 scenario: str = PRIMARY_SCENARIO) -> dict:
        """Evaluate criteria A/B/C for one migrated spectrum under one threshold scenario."""
        table = self.defect_table(source_index, standardised)
        passed_a = passed_b = True
        severity_before = severity_after = 0.0
        conditional = False
        per_defect = {}
        for spec in DEFECT_SPECS:
            threshold = self.thresholds[spec.name][scenario]
            if threshold is None:
                per_defect[spec.name] = {"threshold": None, "e_before": None, "e_after": None,
                                         "pass_a": None, "pass_b": None}
                continue
            after = table[spec.name]["after"]
            e_after = float(max(after / threshold - 1.0, 0.0))
            if table[spec.name]["missing_source"]:
                conditional = True
                e_before = np.nan
                passed_a &= e_after == 0.0
                passed_b &= e_after == 0.0
                severity_after += e_after
                per_defect[spec.name] = {"threshold": threshold, "e_before": None,
                                          "e_after": e_after, "pass_a": e_after == 0.0,
                                          "pass_b": e_after == 0.0, "conditional": True}
                continue
            before = table[spec.name]["before"]
            e_before = float(max(before / threshold - 1.0, 0.0))
            if e_before == 0.0:
                passed_a &= e_after == 0.0
            else:
                passed_a &= e_after <= e_before + 1e-12
            passed_b &= e_after <= e_before + 1e-12
            severity_before += e_before
            severity_after += e_after
            per_defect[spec.name] = {"threshold": threshold, "e_before": e_before,
                                      "e_after": e_after, "pass_a": bool(passed_a),
                                      "pass_b": bool(e_after <= e_before + 1e-12)}
        return {
            "A": bool(passed_a),
            "B": bool(passed_b),
            "C": bool(severity_after <= severity_before + 1e-12),
            "severity_before": severity_before,
            "severity_after": severity_after,
            "conditional": conditional,
            "per_defect": per_defect,
            "table": table,
        }

    # ---------------------------------------------------------------- migration
    def candidates(self, source_index: int, target_style: str) -> list[dict]:
        """All ``lam`` candidates for one (source, target) pair, with criteria results."""
        if target_style not in self.style_list:
            raise ValueError(f"unknown target style {target_style!r}; expected one of {self.style_list}")
        target_mean = self.standardised[self.styles == target_style].mean(0)
        target_index = self.style_list.index(target_style)
        candidates = []
        for lam in LAMBDA_GRID:
            blended = (1.0 - lam) * self.standardised[source_index] + lam * target_mean
            projected, notes = self.project(blended, source_index, target_style)
            distances = self.centroid_distances(projected)
            candidates.append({
                "lambda": lam,
                "standardised": projected,
                "distances": distances,
                "nearest_index": int(np.argmin(distances)),
                "target_distance": float(distances[target_index]),
                "notes": notes,
            })
        return candidates

    def migrate(self, source_index: int, target_style: str) -> dict:
        """Pick ``lam*`` and certify the migrated spectrum under both scenarios."""
        candidates = self.candidates(source_index, target_style)
        target_index = self.style_list.index(target_style)
        target_nearest = [c for c in candidates if c["nearest_index"] == target_index]
        if target_nearest:
            chosen = max(target_nearest, key=lambda c: c["lambda"])
        else:
            chosen = min(candidates, key=lambda c: c["target_distance"])
        before = self.centroid_distances(self.standardised[source_index])
        result = {
            "source_beer": int(self.ids[source_index]),
            "target_style": target_style,
            "lambda": float(chosen["lambda"]),
            "standardised": chosen["standardised"],
            "d_target_before": float(before[target_index]),
            "d_target_after": float(chosen["target_distance"]),
            "distance_reduction_pct": float(100.0 * (1.0 - chosen["target_distance"] / before[target_index])),
            "style_before": self.style_list[int(np.argmin(before))],
            "style_after": self.style_list[chosen["nearest_index"]],
            "style_is_target": bool(chosen["nearest_index"] == target_index),
            "notes": chosen["notes"],
            "scenarios": {},
        }
        for scenario in THRESHOLD_SCENARIOS:
            result["scenarios"][scenario] = self.criteria(source_index, chosen["standardised"], scenario)
        return result

    def run(self, targets=TARGET_STYLES) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Migrate every Pils/Lager source to every target; return (results, candidate counts)."""
        source_rows = np.flatnonzero(self.styles == SOURCE_STYLE)
        results, counts = [], []
        for target in targets:
            target_index = self.style_list.index(target)
            for source_index in source_rows:
                migration = self.migrate(int(source_index), target)
                table = migration["scenarios"][PRIMARY_SCENARIO]
                row = {
                    "source_beer": migration["source_beer"],
                    "target_style": target,
                    "lambda": migration["lambda"],
                    "lambda_star": migration["lambda"],
                    "d_target_before": migration["d_target_before"],
                    "d_target_after": migration["d_target_after"],
                    "dist_reduction_pct": migration["distance_reduction_pct"],
                    "style_before": migration["style_before"],
                    "style_after": migration["style_after"],
                    "style_is_target": migration["style_is_target"],
                    "notes": json.dumps(migration["notes"], ensure_ascii=False),
                }
                for scenario in THRESHOLD_SCENARIOS:
                    block = migration["scenarios"][scenario]
                    for criterion in ("A", "B", "C"):
                        row[f"{criterion}_{scenario}"] = block[criterion]
                    row[f"Esum_{scenario}_before"] = block["severity_before"]
                    row[f"Esum_{scenario}_after"] = block["severity_after"]
                    row[f"conditional_{scenario}"] = block["conditional"]
                row["criteria_A"] = row[f"A_{PRIMARY_SCENARIO}"]
                row["criteria_B"] = row[f"B_{PRIMARY_SCENARIO}"]
                row["criteria_C"] = row[f"C_{PRIMARY_SCENARIO}"]
                for spec in DEFECT_SPECS:
                    before = table["table"][spec.name]["before"]
                    after = table["table"][spec.name]["after"]
                    row[f"{spec.name}_before"] = np.nan if before is None else float(before)
                    row[f"{spec.name}_after"] = float(after)
                    row[f"{spec.name}_after_{spec.unit.replace('/', '_')}"] = float(after)
                row["pass_reason"] = _pass_reason(migration["scenarios"])
                results.append(row)

                all_candidates = self.candidates(int(source_index), target)
                count = {"target_style": target, "source_beer": migration["source_beer"],
                         "n_candidates": len(all_candidates),
                         "n_target_nearest": sum(c["nearest_index"] == target_index
                                                 for c in all_candidates)}
                for scenario in THRESHOLD_SCENARIOS:
                    for criterion in ("A", "B", "C"):
                        count[f"n_pass_{criterion}_{scenario}"] = int(sum(
                            self.criteria(int(source_index), c["standardised"], scenario)[criterion]
                            for c in all_candidates))
                counts.append(count)
        return pd.DataFrame(results), pd.DataFrame(counts)


def _pass_reason(scenarios: dict[str, dict]) -> str:
    median = scenarios[PRIMARY_SCENARIO]
    mean = scenarios["mean"]
    parts = [f"A={median['A']}(median)/{mean['A']}(mean)",
             f"B={median['B']}(median)/{mean['B']}(mean)",
             f"C={median['C']}(median)/{mean['C']}(mean)",
             f"Esum {median['severity_before']:.4f}->{median['severity_after']:.4f} (median)"]
    if median["conditional"]:
        parts.append("conditional: source diacetyl missing -> target-median cap")
    return "; ".join(parts)


def style_defect_table(data: MigrationData) -> pd.DataFrame:
    """Per-style defect baseline in physical units (median / p90 / over-threshold counts)."""
    rows = []
    for style in sorted(set(data.styles)):
        sub = data.coordinates[data.styles == style]
        row = {"style": style, "n": int(len(sub))}
        for spec in DEFECT_SPECS:
            index = data._index(spec.column)
            values = concentration(sub[:, index], data.transforms[spec.column])
            row[f"{spec.name}_unit"] = spec.unit
            row[f"{spec.name}_p50"] = float(np.nanmedian(values))
            finite = values[np.isfinite(values)]
            row[f"{spec.name}_p90"] = float(np.percentile(finite, 90)) if finite.size else np.nan
            row[f"{spec.name}_n_missing"] = int(np.isnan(values).sum())
            for scenario in THRESHOLD_SCENARIOS:
                threshold = data.thresholds[spec.name][scenario]
                if threshold is None:
                    continue
                row[f"{spec.name}_T_{scenario}"] = threshold
                row[f"{spec.name}_exc_{scenario}"] = int((values > threshold).sum())
                row[f"{spec.name}_frac_exc_{scenario}"] = float((values > threshold).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def source_defect_table(data: MigrationData, style: str = SOURCE_STYLE) -> pd.DataFrame:
    """Defect coordinates, concentrations, thresholds and severity for every source beer."""
    rows = []
    for index in np.flatnonzero(data.styles == style):
        row = {"beer": int(data.ids[index]), "style": data.styles[index]}
        for spec in DEFECT_SPECS:
            coordinate_index = data._index(spec.column)
            raw = data.coordinates[index, coordinate_index]
            row[f"{spec.name}_column"] = spec.column
            row[f"{spec.name}_coord"] = float(raw) if np.isfinite(raw) else np.nan
            value = np.nan if not np.isfinite(raw) else float(
                concentration(raw, data.transforms[spec.column]))
            row[f"{spec.name}_concentration"] = value
            for scenario in THRESHOLD_SCENARIOS:
                threshold = data.thresholds[spec.name][scenario]
                row[f"{spec.name}_threshold_{scenario}"] = np.nan if threshold is None else threshold
                row[f"{spec.name}_exceeds_{scenario}"] = (
                    np.nan if (threshold is None or not np.isfinite(value)) else int(value > threshold))
                row[f"{spec.name}_e_{scenario}"] = (
                    np.nan if (threshold is None or not np.isfinite(value))
                    else float(max(value / threshold - 1.0, 0.0)))
        for proxy in STALE_HOP_PROXIES:
            if proxy in data.features:
                row[f"stale_{proxy}"] = float(data.coordinates[index, data._index(proxy)])
        rows.append(row)
    return pd.DataFrame(rows)


def load(source: Path, reference_path: Path | None = None,
         raw_path: Path | None = None) -> MigrationData:
    """Load the attachments into a :class:`MigrationData`.

    ``source`` must contain ``1.csv`` and ``8.xlsx`` (the supplied attachment directory).
    ``reference_path`` defaults to ``source / "8.xlsx"``.  ``raw_path`` is optional external
    sensory data (``data/reference/zenodo23624915``) recorded as provenance only; the
    migration itself uses the compound coordinates and the style labels.
    """
    source = Path(source)
    profiles, workbook = load_profile_frame(source)
    calibrated = workbook["Calibrated chemical dataset"]
    thresholds = load_thresholds(reference_path or (source / "8.xlsx"))
    transforms = infer_transforms(profiles, calibrated)

    features = [column for column in profiles.columns if column not in META_COLUMNS]
    keep = [column for column in features
            if profiles[column].isna().mean() <= 1.0 - MIN_NON_MISSING_FRACTION]
    frame = profiles[keep].copy()
    styles = profiles[STYLE_COLUMN].to_numpy()
    missing_mask = frame.isna().to_numpy()
    for column in keep:
        if frame[column].isna().any():
            style_medians = profiles.groupby(STYLE_COLUMN)[column].median()
            frame[column] = (frame[column].fillna(profiles[STYLE_COLUMN].map(style_medians))
                             .fillna(frame[column].median()))
    coordinates = frame.to_numpy(float)
    full_frame = profiles[features].copy()
    for column in features:
        if full_frame[column].isna().any():
            style_medians = profiles.groupby(STYLE_COLUMN)[column].median()
            full_frame[column] = (full_frame[column].fillna(
                profiles[STYLE_COLUMN].map(style_medians)).fillna(full_frame[column].median()))
    full_coordinates = full_frame.to_numpy(float)
    official_features = [column for column in features
                         if column not in OFFICIAL_EXCLUDED_FEATURES]
    center = np.median(coordinates, axis=0)
    iqr = np.percentile(coordinates, 75, axis=0) - np.percentile(coordinates, 25, axis=0)
    sd = coordinates.std(axis=0)
    scale = np.where(iqr > 1e-9, iqr / 1.349, np.where(sd > 0, sd, 1.0))
    standardised = (coordinates - center) / scale

    centred = standardised - standardised.mean(0)
    _, _, loadings = np.linalg.svd(centred, full_matrices=False)
    components = loadings[:min(N_COMPONENTS, centred.shape[1] - 1)].T
    scores = centred @ components
    component_scale = scores.std(0)
    whitened = scores / component_scale
    style_list = list(np.unique(styles))
    style_centroids = {style: whitened[styles == style].mean(0) for style in style_list}

    provenance = None
    if raw_path is not None and Path(raw_path).exists():
        try:
            raw = pd.read_csv(raw_path, low_memory=False)
        except UnicodeDecodeError:  # the supplied panel file is not UTF-8
            raw = pd.read_csv(raw_path, low_memory=False, encoding="latin-1")
        provenance = {"path_kind": "zenodo23624915_raw_sensory", "rows": int(len(raw)),
                      "columns": int(raw.shape[1]),
                      "note": "external sensory panel data, provenance only; not used by the migration math"}

    data = MigrationData(
        ids=profiles[BEER_COLUMN].to_numpy(),
        styles=styles,
        features=keep,
        coordinates=coordinates,
        standardised=standardised,
        center=center,
        scale=scale,
        missing_mask=missing_mask,
        transforms=transforms,
        thresholds=thresholds,
        style_list=style_list,
        components=components,
        component_scale=component_scale,
        style_centroids=style_centroids,
        source_directory=source,
        raw_provenance=provenance,
        full_features=list(features),
        full_coordinates=full_coordinates,
        official_features=official_features,
    )
    data._feature_index = {name: i for i, name in enumerate(keep)}
    data._full_index = {name: i for i, name in enumerate(features)}
    return data
