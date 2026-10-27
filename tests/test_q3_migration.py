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


if __name__ == "__main__":
    unittest.main()
