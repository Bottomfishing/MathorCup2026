import json
from pathlib import Path
import unittest
import pandas as pd
from beer_models.evidence import resolve_compound, annotate_relation


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((Path(__file__).resolve().parents[1] / 'configs/q2_evidence.json').read_text(encoding='utf-8'))

    def test_explicit_synonym_keeps_positional_isomers_separate(self):
        ref = pd.DataFrame({'Compound': ['ethyl isovalerate', 'ethyl 2-methylbutyrate']})
        hit, method = resolve_compound('ethyl_isovalerate', 'Ethyl 3-methylbutyrate', ref, self.config['aliases'])
        self.assertEqual(hit.Compound, 'ethyl isovalerate')
        self.assertEqual(method, 'explicit_alias')
        with self.assertRaises(ValueError):
            resolve_compound('ethyl_isovalerate', 'Ethyl 2-methylbutyrate', ref, self.config['aliases'])

    def test_beta_and_alpha_not_conflated(self):
        ref = pd.DataFrame({'Compound': ['α-phellandrene', 'β-phellandrene']})
        hit, _ = resolve_compound('beta.phellandrene', 'Beta-phellandrene', ref, self.config['aliases'])
        self.assertEqual(hit.Compound, 'β-phellandrene')

    def test_ambiguous_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_compound('iron', 'Iron', pd.DataFrame({'Compound': ['Iron', 'iron ']}), {})

    def test_negative_coefficient_and_descriptor_overlap_are_distinct(self):
        relation, terms, reason = annotate_relation('A_malt_cara', 'furfural', 'caramel, bread', 'Malt', -.1, self.config)
        self.assertEqual(relation, 'reference_descriptor_overlap')
        self.assertEqual(terms, 'caramel')
        self.assertIn('条件系数为负', reason)
        self.assertIn('未证明', reason)

    def test_lacking_target_evidence_remains_explicit(self):
        relation, _, _ = annotate_relation('bitternes', 'iron', 'metallic', 'Malt', .1, self.config)
        self.assertEqual(relation, 'target_specific_evidence_insufficient')
        relation, _, _ = annotate_relation('metallic', 'iron', 'metallic', 'Malt', .1, self.config)
        self.assertEqual(relation, 'named_marker_matches_target')


if __name__ == '__main__':
    unittest.main()
