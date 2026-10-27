"""Tests for the formalised Q3 cross-style migration module.

The migration needs the official attachment directory (``--source``).  Set the
``Q3_SOURCE`` environment variable to the directory holding ``1.csv`` and
``8.xlsx`` to enable the data-driven tests; they are skipped otherwise so the
suite stays runnable without the competition attachments.

Run with::

    Q3_SOURCE="<...>/附件" .venv/Scripts/python.exe -m unittest tests.test_q3_migration -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beer_models import q3_migration as qm  # noqa: E402

SOURCE = os.environ.get("Q3_SOURCE")
HAS_SOURCE = bool(SOURCE) and Path(SOURCE).exists()

_DATA = None


def data():
    global _DATA
    if _DATA is None:
        _DATA = qm.load(SOURCE)
    return _DATA


def source_index(beer: int) -> int:
    return int(np.flatnonzero(data().ids == beer)[0])


class ThresholdTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_thresholds_use_attachment_reference(self):
        thresholds = qm.load_thresholds(Path(SOURCE) / "8.xlsx")
        self.assertAlmostEqual(thresholds["DMS"]["median"], 19.0)
        self.assertAlmostEqual(thresholds["DMS"]["mean"], 17.25)
        self.assertAlmostEqual(thresholds["Diacetyl"]["median"], 0.09)
        self.assertAlmostEqual(thresholds["Diacetyl"]["mean"], 0.29925)
        self.assertAlmostEqual(thresholds["4VG"]["median"], 0.20)
        self.assertAlmostEqual(thresholds["4VG"]["mean"], 0.206333333, places=8)
        self.assertIsNone(thresholds["t2n"]["median"])
        self.assertIsNone(thresholds["t2n"]["mean"])

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_defect_unit_and_transform_convention(self):
        units = {spec.name: spec.unit for spec in qm.DEFECT_SPECS}
        self.assertEqual(units["DMS"], "ug/L")
        self.assertEqual(units["Diacetyl"], "mg/L")
        self.assertEqual(units["4VG"], "mg/L")
        self.assertEqual(units["t2n"], "mg/L")

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_transforms_follow_calibrated_sheet(self):
        profiles = pd.read_csv(Path(SOURCE) / "1.csv", low_memory=False)
        calibrated = pd.read_excel(Path(SOURCE) / "8.xlsx",
                                   sheet_name="Calibrated chemical dataset")
        transforms = qm.infer_transforms(profiles, calibrated)
        self.assertEqual(transforms["DMS"], "log10")
        self.assertEqual(transforms["X4VG"], "log10")
        self.assertEqual(transforms["X2.3.butanedione"], "identity")
        self.assertEqual(transforms["X.E..2.nonenal"], "identity")


class DataTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_panel_shape_and_styles(self):
        d = data()
        self.assertEqual(d.coordinates.shape[0], 250)
        self.assertEqual(len(d.features), d.coordinates.shape[1])
        self.assertGreaterEqual(len(d.style_list), 20)
        self.assertEqual(int((d.styles == "Pils/Lager").sum()), 11)
        for style in qm.TARGET_STYLES:
            self.assertGreaterEqual(int((d.styles == style).sum()), 10)

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_missing_mask_records_conditional_sources(self):
        d = data()
        index = d._index("X2.3.butanedione")
        missing = {int(beer) for beer in d.ids[d.missing_mask[:, index]]}
        self.assertIn(119, missing)
        self.assertIn(28, missing)

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_standardisation_roundtrip(self):
        d = data()
        np.testing.assert_allclose(d.destandardise(d.standardised), d.coordinates,
                                   rtol=0, atol=1e-9)

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_defect_concentration_inverts_coordinates(self):
        d = data()
        for spec in qm.DEFECT_SPECS:
            coordinate = d.coordinates[0, d._index(spec.column)]
            expected = float(qm.concentration(coordinate, d.transforms[spec.column]))
            self.assertAlmostEqual(d.defect_concentration(d.standardised[0], spec.name),
                                   expected)

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_physical_cap_uses_headroom(self):
        d = data()
        self.assertAlmostEqual(d.physical_cap("DMS"), 0.9 * 19.0)
        self.assertAlmostEqual(d.physical_cap("Diacetyl"), 0.9 * 0.09)
        self.assertAlmostEqual(d.physical_cap("4VG"), 0.9 * 0.20)
        self.assertIsNone(d.physical_cap("t2n"))


class ProjectionTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_projection_never_increases_any_listed_defect(self):
        d = data()
        for index in range(d.coordinates.shape[0]):
            for target in qm.TARGET_STYLES:
                projected, _ = d.project(d.standardised[index], index, target)
                before = d.defect_table(index, d.standardised[index])
                after = d.defect_table(index, projected)
                for spec in qm.DEFECT_SPECS:
                    if before[spec.name]["before"] is None:
                        continue
                    self.assertLessEqual(after[spec.name]["after"],
                                         before[spec.name]["before"] + 1e-12)
                for spec in qm.DEFECT_SPECS:
                    cap = d.physical_cap(spec.name)
                    if cap is not None:
                        self.assertLessEqual(after[spec.name]["after"], cap * (1 + 1e-12))

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_conditional_cap_applies_when_source_is_missing(self):
        d = data()
        results, _ = d.run()
        row = results[(results["source_beer"] == 119)
                      & (results["target_style"] == "Tripel")].iloc[0]
        self.assertTrue(np.isnan(row["Diacetyl_before"]))
        self.assertAlmostEqual(row["Diacetyl_after"], d.physical_cap("Diacetyl"))

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_copy_is_a_fixed_point(self):
        d = data()
        index = source_index(119)
        projected, _ = d.project(d.standardised[index], index, "Tripel")
        np.testing.assert_allclose(projected, d.standardised[index], atol=1e-12)


class CriteriaTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_criteria_hold_for_the_unchanged_spectrum(self):
        d = data()
        for beer in (124, 119, 96):
            index = source_index(beer)
            for scenario in qm.THRESHOLD_SCENARIOS:
                verdict = d.criteria(index, d.standardised[index], scenario)
                self.assertTrue(verdict["A"])
                self.assertTrue(verdict["B"])
                self.assertTrue(verdict["C"])

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_criteria_detect_a_worsened_defect(self):
        d = data()
        index = source_index(124)
        worsened = np.array(d.standardised[index], copy=True)
        cap = d.physical_cap("DMS")
        worsened[d._index("DMS")] = d.standardise_value(
            "DMS", d.defect_coordinate(cap * 1.5, "DMS"))
        verdict = d.criteria(index, worsened)
        self.assertFalse(verdict["A"])
        self.assertFalse(verdict["B"])
        self.assertFalse(verdict["C"])

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_candidate_grid_size(self):
        d = data()
        index = source_index(119)
        for target in qm.TARGET_STYLES:
            self.assertEqual(len(d.candidates(index, target)), len(qm.LAMBDA_GRID))
        self.assertEqual(len(qm.LAMBDA_GRID), 20)

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_invalid_style_is_rejected(self):
        d = data()
        index = source_index(119)
        with self.assertRaises(ValueError):
            d.migrate(index, "Not a style")
        with self.assertRaises(ValueError):
            d.migrate(index, "pils/lager")


class RunTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_full_migration_is_deterministic_and_complete(self):
        d = data()
        results, counts = d.run()
        again, other = d.run()
        self.assertEqual(len(results), 33)
        self.assertEqual(results["target_style"].value_counts().to_dict(),
                         {"Tripel": 11, "Kriek": 11, "Lambic": 11})
        self.assertEqual(int(counts["n_candidates"].sum()), 660)
        for scenario in qm.THRESHOLD_SCENARIOS:
            for criterion in ("A", "B", "C"):
                self.assertEqual(int(results[f"{criterion}_{scenario}"].sum()), 33)
                self.assertEqual(int(counts[f"n_pass_{criterion}_{scenario}"].sum()), 660)
        np.testing.assert_allclose(results["d_target_after"], again["d_target_after"])
        np.testing.assert_allclose(results["4VG_after"], again["4VG_after"])

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_119_tripel_case(self):
        d = data()
        results, _ = d.run()
        row = results[(results["source_beer"] == 119)
                      & (results["target_style"] == "Tripel")].iloc[0]
        self.assertAlmostEqual(row["4VG_before"], 0.143755, places=6)
        self.assertAlmostEqual(row["4VG_after"], 0.0405075942163172, places=9)
        self.assertTrue(np.isnan(row["Diacetyl_before"]))
        self.assertAlmostEqual(row["Diacetyl_after"], 0.081, places=9)
        self.assertAlmostEqual(row["DMS_after"], row["DMS_before"], places=9)
        self.assertAlmostEqual(row["lambda"], 0.55, places=9)

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_style_defect_table_covers_every_style(self):
        d = data()
        table = qm.style_defect_table(d)
        self.assertEqual(len(table), len(set(d.styles)))
        self.assertEqual(set(table["style"]), set(d.styles))


class CaseIdAndStatusTests(unittest.TestCase):
    """Pure helpers: case identifiers and the per-defect status rule."""

    def test_case_id_is_zero_padded(self):
        self.assertEqual(qm.case_id(119, "Tripel"), "S119_TO_Tripel")
        self.assertEqual(qm.case_id(7, "Kriek"), "S007_TO_Kriek")
        self.assertEqual(qm.case_id(250, "Lambic"), "S250_TO_Lambic")

    def test_defect_status_uses_excess_over_threshold(self):
        self.assertEqual(qm.defect_status(0.25, 0.15, 0.20), "improved")
        self.assertEqual(qm.defect_status(0.15, 0.04, 0.20), "unchanged")
        self.assertEqual(qm.defect_status(0.04, 0.04, 0.20), "unchanged")
        self.assertEqual(qm.defect_status(0.15, 0.25, 0.20), "worse")
        # 0.143755 -> 0.040508 is a large concentration drop but both are below
        # the threshold: the pre-registered "超阈值项" grade is unchanged.
        self.assertEqual(qm.defect_status(0.143755, 0.0405076, 0.20), "unchanged")

    def test_defect_status_is_unknown_without_threshold_or_before(self):
        self.assertEqual(qm.defect_status(0.01, 0.02, None), "unknown")
        self.assertEqual(qm.defect_status(float("nan"), 0.02, 0.20), "unknown")
        self.assertEqual(qm.defect_status(0.01, float("nan"), 0.20), "unknown")


class FullSpectraTests(unittest.TestCase):
    """The migration now exports the complete spectra, not only distances."""

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_official_feature_set_is_224(self):
        d = data()
        self.assertEqual(len(d.features), 231)
        self.assertEqual(len(d.official_features), 224)
        self.assertEqual(set(d.official_features), set(d.features) - set(qm.OFFICIAL_EXCLUDED_FEATURES))
        self.assertEqual(d.full_coordinates.shape, (len(d.ids), len(d.features)))

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_spectra_exports_are_paired_and_complete(self):
        d = data()
        spectra = d.migration_spectra()
        long, wide, after_input = spectra["long"], spectra["wide"], spectra["after_input"]
        self.assertEqual(long.shape, (66, 5 + 224))
        self.assertEqual(wide.shape[0], 33)
        self.assertEqual(after_input.shape, (33, 227))
        self.assertEqual(after_input.columns[:3].tolist(), ["generated_id", "style", "method"])
        self.assertEqual(long["case_id"].nunique(), 33)
        self.assertEqual(sorted(long["role"].unique()), ["after", "before"])
        counts = long.groupby(["case_id", "role"]).size()
        self.assertEqual(int(counts.min()), 1)
        self.assertEqual(int(long.iloc[:, 5:].isna().sum().sum()), 0)
        self.assertEqual(long.loc[long["role"] == "after", "case_id"].nunique(), 33)

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_after_input_matches_migrated_official_spectrum(self):
        d = data()
        index = source_index(119)
        migration = d.migrate(index, "Tripel")
        official = d.official_spectrum(index, migration["standardised"])
        after_input = d.migration_spectra()["after_input"]
        row = after_input[after_input["generated_id"] == qm.case_id(119, "Tripel")].iloc[0]
        np.testing.assert_allclose(row[d.official_features].to_numpy(dtype=float), official,
                                   rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(
            d.defect_concentration(migration["standardised"], "4VG"),
            0.0405075942, places=9)

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_before_and_after_differ_only_where_migration_moved_spectrum(self):
        d = data()
        long = d.migration_spectra()["long"]
        before = long[long["role"] == "before"].set_index("case_id")
        after = long[long["role"] == "after"].set_index("case_id")
        features = d.official_features
        delta = (after[features] - before[features]).abs().to_numpy()
        self.assertGreater(float(delta.max()), 0.0)
        changed = int((delta > 1e-12).any(axis=1).sum())
        self.assertEqual(changed, 33)


class EvidenceTests(unittest.TestCase):
    """Per-case evidence grading (chemical pass A/B/C, style, defect statuses)."""

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_evidence_table_marks_style_diagnostics_frankly(self):
        d = data()
        results, _ = d.run()
        evidence = d.evidence_table(results)
        self.assertEqual(len(evidence), 33)
        self.assertTrue(set(evidence["claim_level"]) <= set(qm.CLAIM_LEVELS))
        self.assertEqual(int((evidence["claim_level"] == "undecidable").sum()), 0)
        diagnostics = {style: int(evidence.loc[evidence["target_style"] == style,
                                               "style_diagnostic_cases"].iloc[0])
                       for style in qm.TARGET_STYLES}
        self.assertEqual(diagnostics, {"Tripel": 0, "Kriek": 11, "Lambic": 2})
        self.assertEqual(set(evidence["defect_status_t2n"]), {"unknown"})
        self.assertEqual(set(evidence["defect_status_Stale_hops"]), {"unknown"})
        self.assertEqual(int(evidence["defects_worse_count"].sum()), 0)
        case = evidence[evidence["case_id"] == qm.case_id(119, "Tripel")].iloc[0]
        self.assertAlmostEqual(case["4VG_mgL_before"], 0.143755, places=6)
        self.assertAlmostEqual(case["4VG_mgL_after"], 0.0405076, places=7)
        self.assertTrue(np.isnan(case["Diacetyl_mgL_before"]))
        self.assertAlmostEqual(case["Diacetyl_mgL_after"], 0.081, places=9)
        self.assertIn(case["claim_level"], {"conditional", "model_supported"})
        self.assertEqual(case["defect_status_DMS"], "unchanged")

    @unittest.skipUnless(HAS_SOURCE, "Q3_SOURCE not set")
    def test_model_supported_requires_full_style_diagnostics(self):
        d = data()
        results, _ = d.run()
        evidence = d.evidence_table(results)
        supported = evidence[evidence["claim_level"] == "model_supported"]
        self.assertGreater(len(supported), 0)
        for _, row in supported.iterrows():
            self.assertTrue(row["chemical_pass_A"])
            self.assertTrue(row["chemical_pass_B"])
            self.assertTrue(row["chemical_pass_C"])
            self.assertEqual(row["style_nearest_centroid"], "target")
            self.assertEqual(int(row["style_diagnostic_cases"]), 11)
            self.assertTrue(row["style_diagnostic_supported"])
            self.assertEqual(int(row["defects_worse_count"]), 0)
            self.assertFalse(bool(row["conditional"]))


class DeliveredTablesTests(unittest.TestCase):
    """Checks on the delivered ``outputs/q3_migration_20261028`` tables (skipped if absent)."""

    OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "q3_migration_20261028"

    def table(self, name):
        path = self.OUTPUT / name
        if not path.exists():
            self.skipTest(f"{name} not generated")
        return pd.read_csv(path)

    def test_spectra_long_has_66_paired_rows(self):
        long = self.table("s2c_migration_spectra_224_long.csv")
        self.assertEqual(len(long), 66)
        self.assertEqual(long["case_id"].nunique(), 33)
        self.assertEqual(sorted(long["role"].unique()), ["after", "before"])
        self.assertTrue((long.groupby("case_id")["role"].nunique() == 2).all())
        meta = ["case_id", "source_beer", "target_style", "lambda", "role"]
        features = [column for column in long.columns if column not in meta]
        self.assertEqual(len(features), 224)
        self.assertEqual(int(long[features].isna().sum().sum()), 0)

    def test_spectra_wide_and_after_input(self):
        wide = self.table("s2c_migration_spectra_224_wide.csv")
        self.assertEqual(len(wide), 33)
        self.assertEqual(len(wide.columns), 455)
        self.assertEqual(sum(column.endswith("_after") for column in wide.columns), 225)
        self.assertEqual(sum(column.endswith("_before") for column in wide.columns), 225)
        after_input = self.table("s2c_migration_after_input_33.csv")
        self.assertEqual(after_input.shape, (33, 227))
        self.assertEqual(after_input.columns[:3].tolist(), ["generated_id", "style", "method"])
        self.assertEqual(after_input["style"].value_counts().to_dict(),
                         {"Tripel": 11, "Kriek": 11, "Lambic": 11})

    def test_sensory_table_is_66_by_52(self):
        sensory = self.table("s2c_sensory_before_after_52.csv")
        self.assertEqual(len(sensory), 66 * 52)
        self.assertEqual(sensory["case_id"].nunique(), 33)
        self.assertEqual(sorted(sensory["role"].unique()), ["after", "before"])
        self.assertEqual(sensory["target_52"].nunique(), 52)
        for column in ("grade_raw", "prob_ge5", "prob_6"):
            self.assertIn(column, sensory.columns)
        self.assertEqual(int(sensory.groupby(["case_id", "role"]).size().min()), 52)
        missing = sorted(sensory.loc[sensory["grade_raw"].isna(), "target_52"].dropna().unique())
        self.assertEqual(missing, ["A_acetaldehyde", "F_acetaldehyde"])

    def test_evidence_table_covers_every_case(self):
        evidence = self.table("s2c_evidence_by_case.csv")
        self.assertEqual(len(evidence), 33)
        self.assertEqual(evidence["case_id"].nunique(), 33)
        self.assertTrue(set(evidence["claim_level"]) <= set(qm.CLAIM_LEVELS))
        self.assertEqual(int((evidence["claim_level"] == "undecidable").sum()), 0)
        diagnostics = {style: int(evidence.loc[evidence["target_style"] == style,
                                               "style_diagnostic_cases"].iloc[0])
                       for style in qm.TARGET_STYLES}
        self.assertEqual(diagnostics, {"Tripel": 0, "Kriek": 11, "Lambic": 2})
        self.assertEqual(set(evidence["defect_status_Stale_hops"]), {"unknown"})
        self.assertEqual(set(evidence["defect_status_t2n"]), {"unknown"})
        self.assertEqual(int(evidence["defects_worse_count"].sum()), 0)
        case = evidence[evidence["case_id"] == qm.case_id(119, "Tripel")].iloc[0]
        self.assertAlmostEqual(case["4VG_mgL_before"], 0.143755, places=6)
        self.assertAlmostEqual(case["4VG_mgL_after"], 0.0405076, places=7)
        self.assertAlmostEqual(case["Diacetyl_mgL_after"], 0.081, places=9)
        self.assertTrue(np.isnan(case["Diacetyl_mgL_before"]))

    def test_verification_all_checks_pass(self):
        path = self.OUTPUT / "verification.json"
        if not path.exists():
            self.skipTest("verification.json not generated")
        import json

        verification = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(verification["status"], "passed")
        failed = [item["name"] for item in verification["checks"] if not item["passed"]]
        self.assertEqual(failed, [])
        self.assertEqual(verification["counts"]["spectra_long"], 66)
        self.assertEqual(verification["counts"]["after_input"], [33, 227])
        self.assertEqual(verification["counts"]["sensory"], 66 * 52)
        self.assertEqual(verification["claim_level_counts"].get("undecidable", 0), 0)


if __name__ == "__main__":
    unittest.main()
