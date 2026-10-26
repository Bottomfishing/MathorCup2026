"""Beer, rather than repeated predictions, is the uncertainty resampling unit."""
import unittest
import numpy as np
import pandas as pd
from beer_models.repeated_validation import beer_averaged_interval,repeat_summary,overlap


class RepeatedValidationTests(unittest.TestCase):
    def test_copying_a_repeat_does_not_artificially_narrow_the_interval(self):
        differences=np.array([[0.,.1,.2,.3,.5,.8]])
        styles=np.array(['a']*3+['b']*3)
        single=beer_averaged_interval(differences,styles,300,7,[.025,.975])
        tripled=beer_averaged_interval(np.repeat(differences,3,axis=0),styles,300,7,[.025,.975])
        np.testing.assert_allclose(single,tripled,atol=1e-15)

    def test_repeat_summary_reports_sample_sd_not_a_fake_confidence_interval(self):
        table=pd.DataFrame(dict(method=['x']*3,value=[1.,2.,3.]))
        result=repeat_summary(table,['method'],['value']).iloc[0]
        self.assertEqual(result.n_repeats,3)
        self.assertEqual(result['mean'],2.)
        self.assertEqual(result.sample_sd,1.)
        self.assertEqual(result.minimum,1.);self.assertEqual(result.maximum,3.)

    def test_two_empty_rankings_are_not_perfect_identification_agreement(self):
        self.assertTrue(np.isnan(overlap([],[])))
        self.assertEqual(overlap([1,2],[2,3]),1/3)


if __name__=='__main__':unittest.main()
