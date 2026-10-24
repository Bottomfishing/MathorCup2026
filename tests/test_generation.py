"""Critical generator/constraint tests. AI-assisted: Codex, 2026-10-24."""
import unittest
import numpy as np
from beer_models.generation import StyleGenerator,conservative_screen,excess_severity


class GenerationTest(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[1.,2.,3.],[2.,3.,5.],[3.,4.,7.],[8.,9.,10.],[9.,10.,12.],[10.,11.,13.]])
        self.g=StyleGenerator(2).fit(self.x,['a']*3+['b']*3)

    def test_reproducible_style_bounds(self):
        for method in ['resample','interpolate','latent_gaussian']:
            x,meta=self.g.sample('a',20,method,6)
            np.testing.assert_array_equal(x,self.g.sample('a',20,method,6)[0])
            self.assertTrue((x>=self.x[:3].min(axis=0)-1e-12).all())
            self.assertTrue((x<=self.x[:3].max(axis=0)+1e-12).all())

    def test_interpolation_preserves_linear_identity(self):
        x,meta=self.g.sample('a',20,'interpolate',8)
        np.testing.assert_allclose(x[:,2],2*x[:,0]+1)
        self.assertTrue((meta['parents'][:,0]!=meta['parents'][:,1]).all())

    def test_reject_unknown_style_or_method(self):
        with self.assertRaises(ValueError):self.g.sample('z',1,'resample',0)
        with self.assertRaises(ValueError):self.g.sample('a',1,'unknown',0)

    def test_nonincrease_screen_rejects_each_violation_and_nonfinite(self):
        candidates=np.array([[.5,.4],[1.1,.2],[.2,.2],[np.nan,.2]])
        y=np.array([[.2],[.2],[1.1],[.2]])
        ok,_,_=conservative_screen(candidates,np.array([1.,1.]),y,np.array([1.]),[0,1],[0])
        np.testing.assert_array_equal(ok,[True,False,False,False])

    def test_training_imputation_not_test_data(self):
        x=self.x.copy();x[0,0]=np.nan;g=StyleGenerator(2).fit(x,['a']*3+['b']*3)
        self.assertEqual(g.impute_.statistics_[0],np.nanmedian(x[:,0]))

    def test_exceedance_allows_below_threshold_increase(self):
        np.testing.assert_allclose(excess_severity([.1,.9,1.,1.5],1.),[0,0,0,.5])
        self.assertLessEqual(excess_severity([1.4],1.)[0],excess_severity([1.5],1.)[0])
        with self.assertRaises(ValueError):excess_severity([1.],0)
        with self.assertRaises(ValueError):excess_severity([-1.],1)


if __name__=='__main__':unittest.main()
