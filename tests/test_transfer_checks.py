import unittest
import numpy as np
from beer_models.transfer_checks import coordinate_nonincrease, transfer_status


class TransferCheckTests(unittest.TestCase):
    def test_missing_cap_is_unknown_not_imputed_proof(self):
        self.assertIsNone(coordinate_nonincrease(0., np.nan))
        self.assertEqual(transfer_status([True, None]), 'undetermined_missing_constraint_evidence')

    def test_any_failed_chemical_or_proxy_rejects(self):
        self.assertFalse(coordinate_nonincrease(1.1, 1.))
        self.assertEqual(transfer_status([True, False], True), 'rejected_by_unified_model_checks')

    def test_reference_zero_is_separate_conditional_status(self):
        self.assertTrue(coordinate_nonincrease(0., 0.))
        self.assertEqual(transfer_status([True]*8, True), 'model_candidate_with_reference_zero_assumption')
        self.assertEqual(transfer_status([True]*8), 'model_candidate_under_proxy_constraints')


if __name__ == '__main__':
    unittest.main()
