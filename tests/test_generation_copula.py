"""Tests for the Q3 copula generator (beer_models.generation_copula)."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from beer_models.generation_copula import (
    DEFECT_TRANSFORMS,
    HYBRID_METHOD,
    CopulaQuantileGenerator,
    DonorResidualGenerator,
    HybridSupportGenerator,
    _empirical_quantile_map,
    defect_caps,
    physical_values,
)

FEATURES = ["DMS", "X2.3.butanedione", "X4VG", "X.E..2.nonenal", "linalool", "ethyl acetate"]
THRESHOLDS = {"DMS": 19.0, "X2.3.butanedione": 0.09, "X4VG": 0.20}


def synthetic(seed=7, per_style=12):
    rng = np.random.default_rng(seed)
    styles = np.repeat(["Tripel", "Kriek", "Lambic", "Pils/Lager"], per_style)
    base = np.array([1.5, 0.30, -1.0, 0.012, 0.03, 0.4])
    spread = np.array([0.4, 0.25, 0.5, 0.006, 0.01, 0.15])
    offsets = {"Tripel": np.array([0.3, 0.1, 0.0, 0.0, 0.004, 0.05]),
               "Kriek": np.array([-0.4, 0.2, 0.0, 0.001, -0.008, 0.02]),
               "Lambic": np.array([0.1, 0.05, 0.02, 0.002, 0.0, -0.03]),
               "Pils/Lager": np.array([-0.6, -0.9, 0.35, -0.004, -0.01, -0.1])}
    x = np.vstack([base + offsets[s] + rng.normal(scale=spread) for s in styles])
    x[:, 1] = np.clip(x[:, 1], 0.0, None)  # diacetyl is a raw concentration
    return x, styles


class EmpiricalQuantileMapTest(unittest.TestCase):
    def test_monotone_and_bounded_by_marginal(self):
        values = np.array([-3.0, 0.5, 0.5, 1.2, 9.0])
        marginal = np.array([-1.0, 0.0, 2.0, 4.0, 8.0])
        mapped = _empirical_quantile_map(values, marginal)
        self.assertEqual(mapped.size, values.size)
        self.assertTrue(np.all(np.diff(mapped) >= -1e-12), "quantile map must be monotone")
        self.assertGreaterEqual(mapped.min(), marginal.min())
        self.assertLessEqual(mapped.max(), marginal.max())
        self.assertEqual(mapped[0], marginal[0])
        self.assertEqual(mapped[-1], marginal[-1])

    def test_preserves_order(self):
        rng = np.random.default_rng(3)
        values = rng.normal(size=50)
        mapped = _empirical_quantile_map(values, np.sort(rng.normal(size=9)))
        order = np.argsort(values, kind="stable")
        self.assertTrue(np.all(np.diff(mapped[order]) >= -1e-12))
        for left, right in zip(order[:-1], order[1:]):
            if values[left] < values[right]:
                self.assertLessEqual(mapped[left], mapped[right])


class DefectCapConventionTest(unittest.TestCase):
    def test_log10_and_identity_caps(self):
        caps = defect_caps(THRESHOLDS, cap_fraction=0.9)
        self.assertAlmostEqual(caps["DMS"], np.log10(0.9 * 19.0))
        self.assertAlmostEqual(caps["X2.3.butanedione"], 0.9 * 0.09)
        self.assertAlmostEqual(caps["X4VG"], np.log10(0.9 * 0.20))
        self.assertEqual(DEFECT_TRANSFORMS["DMS"], "log10")
        self.assertEqual(DEFECT_TRANSFORMS["X2.3.butanedione"], "identity")

    def test_physical_values_inverse(self):
        coordinate = np.log10(4.0)
        self.assertAlmostEqual(float(physical_values(coordinate, "DMS")), 4.0)
        self.assertEqual(float(physical_values(0.5, "X2.3.butanedione")), 0.5)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            defect_caps({"DMS": 0.0})
        with self.assertRaises(ValueError):
            defect_caps({"unknown": 1.0})


class GeneratorTest(unittest.TestCase):
    def setUp(self):
        self.x, self.styles = synthetic()

    def make(self, **kwargs):
        options = dict(features=FEATURES, components=4, defect_thresholds=THRESHOLDS)
        options.update(kwargs)
        return CopulaQuantileGenerator(**options).fit(self.x, self.styles)

    def test_fit_records_styles_and_components(self):
        generator = self.make()
        self.assertEqual(list(generator.styles_), ["Kriek", "Lambic", "Pils/Lager", "Tripel"])
        self.assertLessEqual(generator.n_components_, 4)
        protocol = generator.protocol()
        self.assertEqual(protocol["cap_fraction"], 0.9)
        self.assertEqual(protocol["defect_thresholds"], THRESHOLDS)

    def test_empirical_marginals_are_restored(self):
        generator = self.make()
        generated, metadata = generator.sample("Tripel", 400, 11)
        self.assertEqual(metadata["method"], "copula_quantile_projected")
        real = self.x[self.styles == "Tripel"]
        for j, field in enumerate(FEATURES):
            if field in THRESHOLDS:
                cap = 0.9 * THRESHOLDS[field]
                physical = physical_values(generated[:, j], field)
                self.assertLessEqual(physical.max(), cap * (1 + 1e-12), "cap violated for " + field)
                continue
            self.assertAlmostEqual(generated[:, j].min(), real[:, j].min(), places=9)
            self.assertAlmostEqual(generated[:, j].max(), real[:, j].max(), places=9)
            scale = real[:, j].std() or 1.0
            self.assertLess(abs(generated[:, j].mean() - real[:, j].mean()) / scale, 0.2,
                            "quantile mapping should restore the marginal mean for " + field)

    def test_projection_metadata(self):
        generator = self.make()
        _, metadata = generator.sample("Kriek", 100, 5)
        self.assertEqual(set(metadata["caps"]), set(THRESHOLDS))
        expected = defect_caps(THRESHOLDS)
        for field, cap in metadata["caps"].items():
            self.assertAlmostEqual(cap, expected[field])
            self.assertIn(field, metadata["projection"])
        self.assertEqual(len(metadata["defect_physical_range"]), len(THRESHOLDS))

    def test_deterministic_for_same_seed(self):
        generator = self.make()
        first, _ = generator.sample("Tripel", 20, 2026)
        second, _ = generator.sample("Tripel", 20, 2026)
        self.assertTrue(np.array_equal(first, second))
        third, _ = generator.sample("Tripel", 20, 2027)
        self.assertFalse(np.allclose(first, third))

    def test_donor_quantile_proposal(self):
        generator = self.make()
        generated, metadata = generator.sample("Lambic", 30, 3, proposal="donor_quantile")
        self.assertEqual(metadata["method"], "interp_quantile_projected")
        self.assertEqual(generated.shape, (30, len(FEATURES)))
        self.assertTrue(np.isfinite(generated).all())

    def test_no_projection_without_thresholds(self):
        x = self.x.copy()
        x[:, 0] = 3.0  # far above any DMS cap
        generator = CopulaQuantileGenerator(features=FEATURES, components=3).fit(x, self.styles)
        generated, metadata = generator.sample("Tripel", 40, 1)
        self.assertEqual(metadata["caps"], {})
        self.assertLessEqual(physical_values(generated[:, 0], "DMS").max(), 10 ** 3.0 + 1e-9)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            CopulaQuantileGenerator(features=FEATURES).fit(self.x[:, :3], self.styles)
        with self.assertRaises(ValueError):
            CopulaQuantileGenerator(features=FEATURES[:5]).fit(self.x, self.styles)
        with self.assertRaises(ValueError):
            CopulaQuantileGenerator(features=FEATURES).fit(self.x[:2], self.styles[:2])
        bad = self.x.copy()
        bad[:, 2] = np.nan
        with self.assertRaises(ValueError):
            CopulaQuantileGenerator(features=FEATURES).fit(bad, self.styles)
        generator = self.make()
        with self.assertRaises(ValueError):
            generator.sample("unknown", 5, 1)
        with self.assertRaises(ValueError):
            generator.sample("Tripel", 0, 1)
        with self.assertRaises(ValueError):
            generator.sample("Tripel", 5, 1, proposal="nope")
        with self.assertRaises(ValueError):
            generator.sample("Tripel", 5, 1.5)
        with self.assertRaises(ValueError):
            CopulaQuantileGenerator(features=FEATURES).sample("Tripel", 5, 1)


class DonorResidualGeneratorTest(unittest.TestCase):
    """v2 generator: whole donor residual reuse + support projection + defect caps."""

    def setUp(self):
        self.x, self.styles = synthetic()
        self.generator = DonorResidualGenerator(
            features=FEATURES, components=4, defect_thresholds=THRESHOLDS).fit(self.x, self.styles)

    def test_fit_records_donor_pool_and_support_radius(self):
        protocol = self.generator.protocol()
        self.assertEqual(protocol["component"], "DonorResidualGenerator")
        self.assertEqual(protocol["method"], "donor_residual_projected")
        self.assertEqual(protocol["support_quantile"], 0.95)
        for style in ("Tripel", "Kriek", "Lambic"):
            self.assertEqual(protocol["donor_pool_size"][style], 12)
            self.assertGreater(protocol["support_radius"][style], 0.0)

    def test_repeated_donors_become_distinct_spectra(self):
        generated, metadata = self.generator.sample("Kriek", 60, 11)
        self.assertEqual(metadata["method"], "donor_residual_projected")
        unique = np.unique(np.round(generated, 12), axis=0)
        self.assertEqual(len(unique), 60, "jittered quantile map must produce distinct spectra")
        self.assertEqual(generated.shape, (60, len(FEATURES)))

    def test_caps_still_apply(self):
        generated, metadata = self.generator.sample("Tripel", 200, 4)
        for field in THRESHOLDS:
            j = FEATURES.index(field)
            cap = 0.9 * THRESHOLDS[field]
            self.assertLessEqual(physical_values(generated[:, j], field).max(), cap * (1 + 1e-12))
            self.assertIn(field, metadata["projection"])

    def test_support_projection_metadata(self):
        _, metadata = self.generator.sample("Lambic", 40, 8)
        self.assertGreaterEqual(metadata["n_support_projected"], 0)
        self.assertGreater(metadata["support_radius"], 0.0)
        self.assertEqual(metadata["jitter"], True)

    def test_deterministic_and_seed_sensitive(self):
        first, _ = self.generator.sample("Tripel", 20, 2026)
        second, _ = self.generator.sample("Tripel", 20, 2026)
        self.assertTrue(np.array_equal(first, second))
        third, _ = self.generator.sample("Tripel", 20, 2027)
        self.assertFalse(np.allclose(first, third))

    def test_v1_refuses_v2_proposal(self):
        v1 = CopulaQuantileGenerator(features=FEATURES, components=4).fit(self.x, self.styles)
        with self.assertRaises(ValueError):
            v1.sample("Tripel", 5, 1, proposal="donor_residual")

    def test_v2_input_validation(self):
        with self.assertRaises(ValueError):
            self.generator.sample("unknown", 5, 1)
        with self.assertRaises(ValueError):
            self.generator.sample("Tripel", 0, 1)
        with self.assertRaises(ValueError):
            self.generator.sample("Tripel", 5, 1, proposal="donor_quantile")
        with self.assertRaises(ValueError):
            DonorResidualGenerator(features=FEATURES).sample("Tripel", 5, 1)
        no_caps = DonorResidualGenerator(features=FEATURES, components=4).fit(self.x, self.styles)
        _, metadata = no_caps.sample("Tripel", 10, 2)
        self.assertEqual(metadata["caps"], {})


class HybridSupportGeneratorTest(unittest.TestCase):
    """v3 generator: pre-registered interior/donor mixture + caps applied last."""

    def setUp(self):
        self.x, self.styles = synthetic()
        self.generator = HybridSupportGenerator(
            features=FEATURES, components=4, defect_thresholds=THRESHOLDS,
            interior_fraction=0.5).fit(self.x, self.styles)

    def test_protocol_records_pre_registered_mixture(self):
        protocol = self.generator.protocol()
        self.assertEqual(protocol["component"], "HybridSupportGenerator")
        self.assertEqual(protocol["method"], HYBRID_METHOD)
        self.assertEqual(HYBRID_METHOD, "hybrid_support_projected")
        self.assertEqual(protocol["interior_fraction"], 0.5)
        self.assertEqual(protocol["interior_seed_offset"], 7919)
        self.assertEqual(protocol["outer_component"], "donor_residual_projected")
        self.assertIn("StyleGenerator", protocol["interior_component"])
        for style in ("Tripel", "Kriek", "Lambic"):
            self.assertEqual(protocol["donor_pool_size"][style], 12)
            self.assertGreater(protocol["support_radius"][style], 0.0)

    def test_mixture_counts_in_metadata(self):
        generated, metadata = self.generator.sample("Kriek", 20, 7)
        self.assertEqual(generated.shape, (20, len(FEATURES)))
        self.assertEqual(metadata["method"], HYBRID_METHOD)
        self.assertEqual(metadata["n_interior"] + metadata["n_donor"], 20)
        self.assertEqual(metadata["n_interior"], 10)
        self.assertEqual(metadata["n_donor"], 10)
        _, smaller = self.generator.sample("Lambic", 7, 7)
        self.assertEqual(smaller["n_interior"] + smaller["n_donor"], 7)
        unique = np.unique(np.round(generated, 12), axis=0)
        self.assertEqual(len(unique), 20, "mixture spectra must be distinct")

    def test_caps_apply_and_deterministic(self):
        first, metadata = self.generator.sample("Tripel", 200, 4)
        second, _ = self.generator.sample("Tripel", 200, 4)
        self.assertTrue(np.array_equal(first, second))
        for field in THRESHOLDS:
            j = FEATURES.index(field)
            cap = 0.9 * THRESHOLDS[field]
            self.assertLessEqual(physical_values(first[:, j], field).max(), cap * (1 + 1e-12))
            self.assertIn(field, metadata["projection"])
        third, _ = self.generator.sample("Tripel", 20, 5)
        self.assertFalse(np.allclose(first[:20], third))

    def test_interior_fraction_validation_and_input_checks(self):
        with self.assertRaises(ValueError):
            HybridSupportGenerator(features=FEATURES, interior_fraction=1.5)
        with self.assertRaises(ValueError):
            HybridSupportGenerator(features=FEATURES, interior_fraction=-0.1)
        with self.assertRaises(ValueError):
            HybridSupportGenerator(features=FEATURES, components=4).sample("Tripel", 5, 1)
        with self.assertRaises(ValueError):
            self.generator.sample("unknown style", 5, 1)
        with self.assertRaises(ValueError):
            self.generator.sample("Tripel", 0, 1)

    def test_hybrid_leaves_lower_versions_untouched(self):
        v1 = CopulaQuantileGenerator(features=FEATURES, components=4).fit(self.x, self.styles)
        v2 = DonorResidualGenerator(features=FEATURES, components=4).fit(self.x, self.styles)
        _, metadata_v1 = v1.sample("Tripel", 10, 3)
        _, metadata_v2 = v2.sample("Tripel", 10, 3)
        self.assertEqual(metadata_v1["method"], "copula_quantile_projected")
        self.assertEqual(metadata_v2["method"], "donor_residual_projected")
        self.assertEqual(v2.protocol()["method"], "donor_residual_projected")
        self.assertEqual(self.generator.protocol()["method"], HYBRID_METHOD)


    def test_profiles_contract_matches_official_columns(self):
        root = Path(__file__).resolve().parents[1]
        official = root / "outputs" / "task4_raw_q3_20261026" / "generated_profiles.csv"
        if not official.is_file():
            self.skipTest("official generated_profiles.csv not available")
        import pandas as pd
        reference = list(pd.read_csv(official, nrows=0).columns)
        self.assertEqual(len(reference), 227)
        candidate = root / "outputs" / "q3_copula_v3_20261028" / "generated_profiles.csv"
        self.assertTrue(candidate.is_file(), "v3 delivered spectra must exist")
        self.assertEqual(list(pd.read_csv(candidate, nrows=0).columns), reference)

    def test_gate_rules_reused_not_redefined(self):
        script = (Path(__file__).resolve().parents[1] / "scripts"
                  / "q3_official_generation_20261028.py").read_text(encoding="utf-8")
        self.assertIn("def build_gate", script, "gate must be defined once and reused by the gate checks")
        self.assertEqual(script.count("def build_gate"), 1)
        self.assertIn("build_gate(validation, candidate=primary)", script)
        self.assertNotIn("V3_GATE", script, "v3 must not introduce a parallel gate definition")


if __name__ == "__main__":
    unittest.main()
