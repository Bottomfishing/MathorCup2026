import unittest
import numpy as np
from beer_models.stale_risk import StaleNonzeroModel


class StaleRiskTests(unittest.TestCase):
    def test_prior_and_training_only_aging_subset(self):
        x=np.array([[1.,100.],[2.,-100.],[3.,50.],[4.,0.]])
        y=np.array([0.,.1,.3,.4]);query=np.array([[np.nan,99999.],[2.,-99999.]])
        prior=StaleNonzeroModel('training_prior',[0],1).fit(x,y)
        np.testing.assert_allclose(prior.predict(query),.2)
        model=StaleNonzeroModel('age_logit_C0.1',[0],1).fit(x,y)
        np.testing.assert_allclose(model.imputer.statistics_,[2.5])
        same=np.array([[2.,9.],[2.,-9.]])
        np.testing.assert_array_equal(model.predict(same)[0],model.predict(same)[1])
        self.assertTrue(((model.predict(query)>=0)&(model.predict(query)<=1)).all())

    def test_forest_uses_unique_beer_rows_and_preserves_probability_bounds(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(25,3));y=np.linspace(0,.5,25)
        model=StaleNonzeroModel('all_et_leaf10',[0],1).fit(x,y)
        np.testing.assert_allclose(model.imputer.statistics_,np.median(x,axis=0))
        self.assertTrue(((model.predict(x)>=0)&(model.predict(x)<=1)).all())


if __name__ == '__main__':unittest.main()
