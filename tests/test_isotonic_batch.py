import itertools
import unittest
import numpy as np
from sklearn.isotonic import isotonic_regression
from beer_models.ordered_models import cumulative_probabilities


class IsotonicBatchTests(unittest.TestCase):
    def check_reference(self,q):
        fixed=np.asarray([isotonic_regression(row,increasing=False) for row in q.reshape(-1,6)]).reshape(q.shape)
        edges=np.concatenate([np.ones(q.shape[:2]+(1,)),fixed,np.zeros(q.shape[:2]+(1,))],axis=2)
        actual=cumulative_probabilities(q)
        np.testing.assert_allclose(actual,edges[:,:,:-1]-edges[:,:,1:],rtol=0,atol=2e-15)
        self.assertGreaterEqual(actual.min(),0)
        np.testing.assert_allclose(actual.sum(2),1,atol=2e-15)

    def test_all_ternary_sequences_including_constant_and_crossed(self):
        self.check_reference(np.array(list(itertools.product([0.,.5,1.],repeat=6)))[:,None,:])

    def test_random_and_almost_tied(self):
        rng=np.random.default_rng(26)
        self.check_reference(rng.random((100,5,6)))
        self.check_reference(.5+rng.uniform(-1e-14,1e-14,(20,3,6)))

    def test_empty_batch(self):
        self.assertEqual(cumulative_probabilities(np.empty((0,50,6))).shape,(0,50,7))
