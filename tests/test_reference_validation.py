import unittest
import numpy as np
from beer_models.reference_validation import nearest_reference,anchored_prediction


class ReferenceValidationTests(unittest.TestCase):
    def test_matching_uses_only_declared_columns(self):
        i,d=nearest_reference([1,999],[[0,999],[1,-999]],[0])
        self.assertEqual(i,1);self.assertEqual(d,0)

    def test_tie_is_deterministic_and_empty_rejected(self):
        self.assertEqual(nearest_reference([0],[[1],[-1]],[0])[0],0)
        with self.assertRaises(ValueError):nearest_reference([0],np.empty((0,1)),[0])

    def test_anchor_exact_reference_and_shared_bias(self):
        self.assertEqual(anchored_prediction(2,10,10),2)
        self.assertEqual(anchored_prediction(2,11,10),3)
        with self.assertRaises(ValueError):anchored_prediction(2,np.nan,10)


if __name__=='__main__':unittest.main()
