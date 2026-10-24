"""Critical failure cases. AI-assisted: OpenAI Codex (GPT-6), 2026-10-23."""
import unittest
import numpy as np
import pandas as pd
from beer_audit.checks import align_samples, compare_scale, numeric_frame, require_original_score_calibration, square_matrix_checks
from beer_audit.checks import pairwise_spearman, repair_utf8_mojibake


class ChecksTest(unittest.TestCase):
    def setUp(self):
        self.a = pd.DataFrame({'beer':['1','2'], 'beer_id':['9007199254740993','2'],
                               'tasting_category_fine':['A','B'], 'x':[1,2]})

    def test_reordered_rows_align_without_numeric_barcode_conversion(self):
        b=align_samples(self.a,self.a.iloc[::-1])
        self.assertEqual(b['x'].tolist(),[1,2])
        self.assertEqual(b.beer_id[0],'9007199254740993')

    def test_duplicate_rejected(self):
        b=self.a.copy(); b.loc[1,'beer']='1'
        with self.assertRaises(ValueError): align_samples(self.a,b)

    def test_lost_sample_rejected(self):
        with self.assertRaises(ValueError): align_samples(self.a,self.a.iloc[:1])

    def test_style_mismatch_rejected(self):
        b=self.a.copy();b.loc[0,'tasting_category_fine']='B'
        with self.assertRaises(ValueError):align_samples(self.a,b)

    def test_malformed_and_infinite_are_not_imputed(self):
        for x in ['typo',float('inf')]:
            with self.assertRaises(ValueError):numeric_frame(pd.DataFrame({'x':[x]}))

    def test_log10_and_ln_discriminated(self):
        b=np.array([.01,.2,4.,31.])
        for label,f in [('log10',np.log10),('ln',np.log)]:
            r=compare_scale(f(b),b)
            self.assertEqual(r['candidate'],label)
            self.assertEqual(r['status'],'positive_pairs_match')

    def test_zero_reference_does_not_disappear(self):
        r=compare_scale([-5,-2,0,1],[0,.01,1,10])
        self.assertEqual(r['n_reference_zero'],1)
        self.assertEqual(r['candidate'],'log10')

    def test_constant_pairs_cannot_identify_transform(self):
        self.assertEqual(compare_scale([1,1,1],[1,1,1])['candidate'],'undetermined')

    def test_unknown_scale_blocks_inverse(self):
        with self.assertRaises(ValueError):require_original_score_calibration({})

    def test_matrix_axis_order_is_not_assumed(self):
        a=pd.DataFrame([[1,.5],[.5,1]],index=['a','b'],columns=['b','a'])
        self.assertFalse(square_matrix_checks(a)['same_ordered_axes'])

    def test_out_of_bounds_correlation_reported(self):
        a=pd.DataFrame([[1,1.5],[1.5,1]],index=['a','b'],columns=['a','b'])
        self.assertEqual(square_matrix_checks(a)['out_of_bounds'],2)

    def test_pairwise_ranking_after_dropping_missing(self):
        data=pd.DataFrame({'a':[1.,2.,3.,4.,5.], 'b':[1.,2.,np.nan,3.,4.]})
        self.assertAlmostEqual(pairwise_spearman(data).loc['a','b'],1.0)
        self.assertLess(data.rank().corr().loc['a','b'],1.0)

    def test_encoding_repair_preserves_normal_text(self):
        self.assertEqual(repair_utf8_mojibake('α-phellandrene'),'α-phellandrene')
        self.assertEqual(repair_utf8_mojibake('α-phellandrene'.encode('utf-8').decode('latin1')),'α-phellandrene')

if __name__=='__main__':unittest.main()
