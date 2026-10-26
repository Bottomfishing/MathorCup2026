"""Generation support, duplicate resampling and complete probability guards."""
import unittest
import numpy as np
from beer_models.q3_raw import (ShrinkageCopula,defect_guard,resampled_records,
    spectrum_diagnostics,discrete_w1)


class RawQ3Tests(unittest.TestCase):
    def test_copula_fixed_seed_stays_in_training_empirical_marginals(self):
        x=np.column_stack([np.arange(12,dtype=float),np.arange(12,dtype=float)**2,np.ones(12)*5])
        x[0,0]=np.nan
        model=ShrinkageCopula().fit(x,np.array(['a']*6+['b']*6))
        generated=model.sample('a',40,3)
        np.testing.assert_array_equal(generated,model.sample('a',40,3))
        self.assertTrue((generated>=model.marginals['a'].min(0)-1e-12).all())
        self.assertTrue((generated<=model.marginals['a'].max(0)+1e-12).all())
        np.testing.assert_allclose(generated[:,2],5)
        self.assertGreaterEqual(np.linalg.eigvalsh(model.correlations['a']).min(),-1e-12)

    def test_zero_grades_cannot_hide_increased_uncalibrated_defect_risk(self):
        def result(risk):
            p=np.zeros((1,1,7));p[0,0,0]=1-risk;p[0,0,1]=risk
            return p
        before=dict(grade=np.zeros((1,1),int),calibrated=result(.1),uncalibrated=result(.1))
        after=dict(grade=np.zeros((1,1),int),calibrated=result(.05),uncalibrated=result(.2))
        passed,_,parts=defect_guard(before,after,[0],1e-10)
        self.assertTrue(parts['grade'].item());self.assertTrue(parts['calibrated'].item())
        self.assertFalse(parts['uncalibrated'].item());self.assertFalse(passed.item())

    def test_bootstrap_repeated_beer_keeps_every_occurrence_unit_weight(self):
        data=dict(positions=np.array([0,0,1]),scores=np.array([[0],[6],[2]]),weights=np.array([.5,.5,1.]))
        positions,scores,weights=resampled_records(data,np.array([1,1,0]))
        np.testing.assert_allclose(np.bincount(positions,weights=weights),np.ones(3))
        np.testing.assert_array_equal(scores.ravel(),[2,2,0,6])

    def test_copy_and_near_duplicate_diagnostics_do_not_call_resampling_novel(self):
        train=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
        generated=np.repeat(train[:1],4,axis=0)
        protocol=dict(coverage_radius_quantile=.95,near_copy_fraction_of_training_median_LOO_radius=.05)
        metrics=spectrum_diagnostics(train,train,generated,protocol)
        self.assertEqual(metrics['exact_training_copy_rate'],1.)
        self.assertEqual(metrics['internal_near_duplicate_rate'],1.)
        self.assertEqual(metrics['variance_ratio'],0.)

    def test_discrete_wasserstein_retains_rating_distance(self):
        zero=np.eye(7)[0];six=np.eye(7)[6]
        self.assertAlmostEqual(float(discrete_w1(zero,six)),6.)


if __name__=='__main__':unittest.main()
