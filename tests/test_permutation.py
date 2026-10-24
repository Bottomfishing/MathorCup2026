"""Tests for held-out permutation. AI-assisted: OpenAI Codex, 2026-10-24."""
import unittest
import numpy as np
from beer_models.permutation import permuted_copy,permutation_deltas,correlation_groups,ranked_indices,validation_bootstrap

class KnownModel:
    def predict(self,x):return x[:,[0]]

class PermutationTest(unittest.TestCase):
    def test_joint_permutation_preserves_group_relationship(self):
        x=np.array([[1.,2.,9.],[2.,4.,8.],[3.,6.,7.]])
        out=permuted_copy(x,[0,1],[2,0,1])
        np.testing.assert_equal(out[:,1],out[:,0]*2)
        np.testing.assert_equal(out[:,2],x[:,2])
        np.testing.assert_equal(x[:,0],[1,2,3])

    def test_unused_feature_exactly_zero_and_used_positive(self):
        x=np.arange(40.).reshape(20,2);y=x[:,[0]]
        d,r,_=permutation_deltas(KnownModel(),x,y,[[0],[1]],3,9,2)
        self.assertGreater(d[:,0,0].mean(),0)
        np.testing.assert_equal(d[:,1,0],np.zeros(20))
        np.testing.assert_allclose(d.mean(axis=0),r.mean(axis=0))

    def test_negative_importance_is_not_promoted_by_absolute_value(self):
        np.testing.assert_equal(ranked_indices([-10,.3,.1,.2]),[1,3,2])

    def test_complete_link_groups_only_training_correlations(self):
        rng=np.random.default_rng(4);a=rng.normal(size=60)
        x=np.column_stack([a,-a,rng.normal(size=60)])
        groups,rho=correlation_groups(x,.9)
        self.assertIn([0,1],groups);self.assertIn([2],groups)
        self.assertAlmostEqual(rho[0,1],-1)

    def test_validation_bootstrap_counts_top3_per_target(self):
        rng=np.random.default_rng(5);d=rng.normal(size=(10,5,2))
        means,counts,draws=validation_bootstrap(d,['a']*3+['b']*7,10,8)
        np.testing.assert_equal(counts.sum(axis=0),[30,30])
        self.assertEqual(means.shape,(10,5,2));self.assertEqual(draws.shape,(10,10))

    def test_within_style_cannot_perturb_style_constant_feature(self):
        x=np.array([[1.,2.],[1.,3.],[8.,4.],[8.,9.],[99.,1.]])
        styles=np.array(['a','a','b','b','single'])
        d,_,orders=permutation_deltas(KnownModel(),x,x[:,[0]],[[0],[1]],5,12,strata=styles)
        np.testing.assert_array_equal(d,np.zeros_like(d))
        for order in orders:
            np.testing.assert_array_equal(styles[order],styles)
            self.assertEqual(order[-1],4)

if __name__=='__main__':unittest.main()
