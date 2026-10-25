"""Protect identity linkage and grouping before external labels can enter a model."""
import unittest

import numpy as np
import pandas as pd

from beer_audit.raw_sensory import (normalize_panel, link_beers,
                                    validate_scores, group_fold_assignments)


class RawSensoryTests(unittest.TestCase):
    def test_integer_scores_keep_real_zeros_and_reject_missing_or_fractional(self):
        validate_scores(pd.DataFrame({'score': [0, 3, 6]}))
        for bad in [np.nan, np.inf, -1, 7, 2.5]:
            with self.subTest(score=bad), self.assertRaises(ValueError):
                validate_scores(pd.DataFrame({'score': [bad]}))

    def test_zero_variance_is_undefined_in_normalized_space_but_raw_zero_is_preserved(self):
        raw = pd.DataFrame({'taster': [1, 1, 2, 2], 'a': [0, 2, 1, 3], 'b': [0, 0, 0, 2]})
        z, valid, constant = normalize_panel(raw, ['a', 'b'])
        self.assertEqual(valid, ['a'])
        self.assertTrue(z.loc[:1, 'b'].isna().all())
        self.assertEqual(constant[['taster', 'target', 'constant_raw_score']].values.tolist(), [[1, 'b', 0]])
        self.assertEqual(raw.b.tolist(), [0, 0, 0, 2])
        np.testing.assert_allclose(z.a, [-1 / np.sqrt(2), 1 / np.sqrt(2)] * 2)

    def test_identity_uses_values_even_when_rows_and_ids_disagree(self):
        left = pd.DataFrame({'a': [-1., 0., 1.], 'b': [0., 2., -2.]}, index=[8, 2, 5])
        right = pd.DataFrame({'beer': [200, 300, 100], 'a': [0., 1., -1.], 'b': [2., -2., 0.]})
        result = link_beers(left, right, ['a', 'b'], 1e-7)
        self.assertEqual(result.beer.tolist(), [100, 200, 300])
        self.assertEqual(result.panel_row_1based.tolist(), [3, 1, 2])

    def test_ambiguous_and_unmatched_beers_are_rejected(self):
        left = pd.DataFrame({'a': [-1., 1.]}, index=[1, 2])
        for values in [[-1., -1.], [-1., 2.]]:
            right = pd.DataFrame({'beer': [10, 20], 'a': values})
            with self.subTest(values=values), self.assertRaises(ValueError):
                link_beers(left, right, ['a'], 1e-7)

    def test_all_repeated_ratings_receive_their_beer_fold(self):
        raw = pd.DataFrame({'beer_name': [4, 9, 4, 4, 9]})
        mapping = pd.DataFrame({'raw_beer_name': [4, 9], 'beer': [50, 60]})
        folds = pd.DataFrame({'beer': [60, 50], 'outer_fold': [2, 0]})
        result = group_fold_assignments(raw, mapping, folds)
        self.assertEqual(result.outer_fold.tolist(), [0, 2, 0, 0, 2])
        self.assertEqual(result.raw_record_1based.tolist(), [1, 2, 3, 4, 5])


if __name__ == '__main__':
    unittest.main()
