"""Tests for the Q3 T4/T6 evidence layer (``beer_models.q3_evidence``).

Run with::

    .venv\\Scripts\\python.exe -m unittest tests.test_q3_evidence -v

The artifact tests only run when ``outputs/q3_evidence_20261028/`` exists (it is produced by
``scripts/q3_evidence_20261028.py``); the unit tests never write anywhere.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from beer_models import q3_evidence as qe  # noqa: E402

EVIDENCE = ROOT / "outputs" / "q3_evidence_20261028"
STYLES = ("Tripel", "Kriek", "Lambic")


def _toy_folds(n: int, folds: int = 2):
    splits = np.array_split(np.arange(n), folds)
    return [(np.setdiff1d(np.arange(n), part), part) for part in splits]


class ClassifierMatrixTest(unittest.TestCase):
    def test_reorders_into_bundle_order(self):
        official = ["a", "b", "c"]
        coordinates = np.array([[1.0, 2.0, 3.0]])
        matrix = qe.classifier_matrix(coordinates, official, ["c", "a"])
        np.testing.assert_array_equal(matrix, [[3.0, 1.0]])

    def test_missing_feature_raises(self):
        with self.assertRaises(ValueError):
            qe.classifier_matrix(np.zeros((1, 2)), ["a", "b"], ["a", "missing"])


class OofPredictionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        blobs = [np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([0.0, 10.0])]
        self.x = np.vstack([blob + rng.normal(scale=0.1, size=(10, 2)) for blob in blobs])
        self.styles = np.array([style for style in STYLES for _ in range(10)], dtype=object)
        self.folds = _toy_folds(len(self.x), folds=5)
        self.config = {"seed": 7}

    def test_centroid_oof_assigns_every_row(self):
        predictions = qe.centroid_oof_predictions(self.x, self.styles, self.folds)
        self.assertFalse(any(value is None for value in predictions))
        np.testing.assert_array_equal(predictions, self.styles)

    def test_classifier_oof_fills_every_row(self):
        class FakeModel:
            def fit(self, x, y):
                self.classes_ = np.unique(y)
                return self

            def predict(self, x):
                return np.array([self.classes_[0]] * len(x), dtype=object)

        predictions, records = qe.oof_style_predictions(self.x, self.styles, self.config,
                                                         self.folds,
                                                         lambda name, config, seed: FakeModel())
        for name in ("logistic", "extra_trees"):
            self.assertFalse(any(value is None for value in predictions[name]))
        self.assertEqual(len(records), 2 * len(self.folds))
        # determinism: the same fold plan/factory gives identical predictions
        again, _ = qe.oof_style_predictions(self.x, self.styles, self.config, self.folds,
                                            lambda name, config, seed: FakeModel())
        for name in predictions:
            np.testing.assert_array_equal(predictions[name], again[name])

    def test_metrics_confusion_and_focus(self):
        metrics = qe.classification_metrics(self.styles, self.styles)
        self.assertEqual(metrics["n"], len(self.styles))
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_recall"], 1.0)
        self.assertEqual(len(metrics["confusion"]), 3)
        for style in STYLES:
            self.assertEqual(metrics["focus"][style]["recall"], 1.0)

    def test_fold_denominators_cover_all_rows(self):
        table = qe.fold_denominators(self.styles, self.folds)
        self.assertEqual(len(table), len(self.folds) * len(STYLES))
        totals = table.groupby("style")["n_heldout_style"].sum().to_dict()
        for style in STYLES:
            self.assertEqual(totals[style], 10)
        self.assertTrue((table["n_train_style"] + table["n_heldout_style"] == 10).all())


class AblationTest(unittest.TestCase):
    def _frame(self, generator: str) -> pd.DataFrame:
        rows = []
        for fold in range(2):
            for style in STYLES:
                row = {"outer_fold": fold, "style": style, "method": generator}
                row.update({name: float(fold + index) for index, name in
                            enumerate(qe.ABLATION_METRICS)})
                rows.append(row)
        return pd.DataFrame(rows)

    def _denominators(self) -> pd.DataFrame:
        return pd.DataFrame([{"outer_fold": fold, "style": style, "n_train_style": 10,
                              "n_heldout_style": 2 + fold}
                             for fold in range(2) for style in STYLES])

    def test_table_merges_denominators(self):
        table = qe.ablation_table({"interpolate": self._frame("interpolate"),
                                   "donor_residual_projected":
                                       self._frame("donor_residual_projected")},
                                  self._denominators())
        self.assertEqual(len(table), 12)
        self.assertTrue(table["n_heldout_style"].between(2, 3).all())
        self.assertEqual(table["gate_version"].unique().tolist(), ["v1", "v2"])
        self.assertEqual(int(table["primary_comparison"].sum()), 12)

    def test_missing_denominator_raises(self):
        with self.assertRaises(ValueError):
            qe.ablation_table({"interpolate": self._frame("interpolate")},
                              self._denominators().iloc[:-1])

    def test_missing_metric_raises(self):
        frame = self._frame("interpolate").drop(columns=["energy"])
        with self.assertRaises(ValueError):
            qe.ablation_table({"interpolate": frame}, self._denominators())

    def test_summary_scopes_and_denominators(self):
        table = qe.ablation_table({"interpolate": self._frame("interpolate")},
                                  self._denominators())
        summary = qe.ablation_summary(table)
        self.assertEqual(sorted(summary["scope"].unique().tolist()),
                         ["all_styles", "by_style"])
        overall = summary.loc[summary["scope"] == "all_styles"].iloc[0]
        self.assertEqual(overall["heldout_total"], 3 * (2 + 3))
        self.assertEqual(overall["heldout_per_fold"], "6/9")
        self.assertIn("energy", summary.columns)


class GateBreakdownTest(unittest.TestCase):
    def _gate(self, passed: list[bool]) -> pd.DataFrame:
        return pd.DataFrame({"rule": [f"rule_{index}" for index in range(len(passed))],
                             "passed": passed})

    def test_thirteen_substantive_plus_one_aggregate(self):
        breakdown = qe.gate_breakdown(self._gate([True] * 13))
        self.assertEqual(qe.SUBSTANTIVE_GATE_RULES, 13)
        self.assertEqual(qe.AGGREGATE_GATE_ITEMS, 1)
        self.assertEqual(breakdown["substantive_rules"], 13)
        self.assertEqual(breakdown["substantive_rules_passed"], 13)
        self.assertEqual(breakdown["aggregate_items"], 1)
        self.assertEqual(breakdown["aggregate_items_passed"], 1)
        self.assertEqual(breakdown["total_items"], 14)
        kinds = [row["kind"] for row in breakdown["rules"]]
        self.assertEqual(kinds.count("substantive"), 13)
        self.assertEqual(kinds.count("aggregate"), 1)

    def test_failed_substantive_rule_fails_aggregate(self):
        breakdown = qe.gate_breakdown(self._gate([True] * 12 + [False]))
        self.assertEqual(breakdown["substantive_rules_passed"], 12)
        self.assertEqual(breakdown["aggregate_items_passed"], 0)


class _StubMigrationData:
    """Minimal stand-in exposing the APIs ``defect_safe_point`` needs."""

    features = ["other", "DMS", "X2.3.butanedione", "X4VG", "X.E..2.nonenal"]
    caps = {"DMS": 17.1, "Diacetyl": 0.081, "4VG": 0.18, "t2n": None}

    def physical_cap(self, defect, scenario="median"):
        return self.caps[defect]

    def defect_coordinate(self, concentration, defect):
        return float(concentration)

    def standardise_value(self, column, value):
        return float(value)

    def defect_concentration(self, standardised, defect):
        column = {"DMS": "DMS", "Diacetyl": "X2.3.butanedione", "4VG": "X4VG"}[defect]
        return float(standardised[self.features.index(column)])


class DefectSafePointTest(unittest.TestCase):
    def test_clamps_threshold_defects_only(self):
        point = np.array([0.0, 30.0, 0.5, 0.05, 0.001])
        result = qe.defect_safe_point(_StubMigrationData(), point, "median")
        self.assertAlmostEqual(result["defects"]["DMS"], 17.1)
        self.assertAlmostEqual(result["defects"]["Diacetyl"], 0.081)
        self.assertAlmostEqual(result["defects"]["4VG"], 0.05)
        self.assertAlmostEqual(result["point"][4], 0.001)  # t2n untouched
        self.assertEqual(result["caps"], {"DMS": 17.1, "Diacetyl": 0.081, "4VG": 0.18})

    def test_does_not_raise_below_caps(self):
        point = np.array([0.0, 5.0, 0.01, 0.01, 0.001])
        result = qe.defect_safe_point(_StubMigrationData(), point, "median")
        np.testing.assert_allclose(result["point"], point)


@unittest.skipUnless((EVIDENCE / "verification.json").is_file(),
                     "run scripts/q3_evidence_20261028.py first")
class EvidenceArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verification = json.loads((EVIDENCE / "verification.json").read_text(encoding="utf-8"))
        cls.summary = json.loads((EVIDENCE / "summary.json").read_text(encoding="utf-8"))

    def test_verification_passed(self):
        self.assertEqual(self.verification["status"], "passed")
        self.assertEqual([check["name"] for check in self.verification["checks"]
                          if not check["passed"]], [])
        self.assertGreaterEqual(len(self.verification["checks"]), 40)

    def test_style_oof_table(self):
        table = pd.read_csv(EVIDENCE / "style_classifier_oof.csv")
        self.assertEqual(len(table), 3 * (1 + 22))
        overall = table.loc[table["scope"] == "overall"]
        self.assertEqual(sorted(overall["method"]), ["extra_trees", "logistic", "nearest_centroid_oof"])
        for style in STYLES:
            for method in overall["method"]:
                support = int(table.loc[(table["method"] == method)
                                        & (table["style"] == style), "support"].iloc[0])
                self.assertGreater(support, 0)

    def test_tripel_conflict_table(self):
        table = pd.read_csv(EVIDENCE / "tripel_conflict.csv")
        self.assertEqual(len(table), 11)
        self.assertTrue(table["case_id"].str.startswith("S").all())
        self.assertTrue((table["DMS_after_ugL"] <= table["DMS_cap_median"] + 1e-12).all())
        self.assertTrue((table["d_safe_after"] > 0).all())
        summary = json.loads((EVIDENCE / "tripel_conflict_summary.json")
                             .read_text(encoding="utf-8"))
        self.assertEqual(summary["n_style_samples"], 28)
        self.assertEqual(summary["safe_reference_kind"], "projected_style_centroid_under_caps")
        self.assertEqual(summary["n_defect_safe_samples_median"], 0)

    def test_migrated_classifier_table(self):
        table = pd.read_csv(EVIDENCE / "migrated_style_classifier.csv")
        self.assertEqual(len(table), 33)
        self.assertEqual(sorted(table["requested_style"].unique().tolist()), sorted(STYLES))
        counts = self.summary["migrated_style_classifier_counts_by_method"]
        self.assertEqual(counts["logistic"]["Tripel"], 11)
        self.assertEqual(counts["nearest_centroid"]["Tripel"], 0)

    def test_ablation_table(self):
        table = pd.read_csv(EVIDENCE / "generator_ablation.csv")
        self.assertEqual(len(table), 5 * 9)
        self.assertEqual(sorted(table["generator"].unique().tolist()),
                         sorted(qe.ABLATION_GENERATORS))
        self.assertTrue(table["n_heldout_style"].notna().all())
        styles = {"Tripel": 28, "Kriek": 10, "Lambic": 11}
        for style, total in styles.items():
            self.assertEqual(int(table.loc[(table["generator"] == "interpolate")
                                           & (table["style"] == style), "n_heldout_style"].sum()),
                             total)

    def test_gate_rules_file(self):
        rules = pd.read_csv(EVIDENCE / "generator_gate_rules.csv")
        self.assertEqual(len(rules), 14)
        self.assertEqual(int((rules["kind"] == "substantive").sum()), 13)
        self.assertEqual(int((rules["kind"] == "aggregate").sum()), 1)
        self.assertTrue(rules.loc[rules["kind"] == "substantive", "passed"].all())
        breakdown = json.loads((EVIDENCE / "generator_gate_breakdown.json")
                               .read_text(encoding="utf-8"))
        self.assertEqual(breakdown["v3"]["substantive_rules"], 13)
        self.assertEqual(breakdown["v3"]["aggregate_items"], 1)
        self.assertEqual(breakdown["v3"]["total_items"], 14)
        self.assertEqual(breakdown["v3"]["substantive_rules_passed"], 13)

    def test_disclosure_document(self):
        text = (EVIDENCE / "dev_use_disclosure.md").read_text(encoding="utf-8")
        for phrase in ("固定验证折", "实质规则", "总汇总", "近重复", "不是独立泛化证明",
                       "不是实测酒"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
