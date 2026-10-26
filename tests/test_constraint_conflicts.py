import unittest
import numpy as np
from beer_models.constraint_conflicts import subset_counts,minimal_blockers,minimax_diagnostic


class ConflictTests(unittest.TestCase):
    def test_joint_conflict_even_when_every_single_group_has_witness(self):
        x=np.array([[1,0,1],[0,1,1]],bool)
        self.assertEqual(minimal_blockers(x),[(0,1)])
        self.assertEqual(subset_counts(x)[(2,)],2)

    def test_multiple_inclusion_minimal_not_only_smallest_cardinality(self):
        x=np.array([[0,1,0],[0,0,1]],bool)
        self.assertEqual(minimal_blockers(x),[(0,),(1,2)])

    def test_empty_pool_and_unrestricted_feasibility(self):
        self.assertEqual(minimal_blockers(np.empty((0,3),bool)),[()])
        self.assertEqual(minimal_blockers(np.ones((2,3),bool)),[])
        self.assertEqual(subset_counts(np.empty((4,0),bool)),{():4})

    def test_minimax_ties_and_failure_is_not_success(self):
        chosen,maximum,n=minimax_diagnostic([[.1,-.1],[.1,.05],[.1,-.2],[.2,0]],[1,0,1,0],[9,2,3,4])
        self.assertEqual(chosen,2)
        self.assertGreater(maximum[chosen],0)
        self.assertEqual(n[chosen],1)

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):subset_counts([[1,0]])
        with self.assertRaises(ValueError):subset_counts(np.ones((1,17),bool))
        with self.assertRaises(ValueError):minimax_diagnostic([[np.nan]],[0],[1])
        with self.assertRaises(ValueError):minimax_diagnostic([[0],[1]],[0,1],[1,1])
