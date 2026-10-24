"""Q2 critical checks; AI-assisted: OpenAI Codex, OpenAI, 2026-10-23."""
import unittest
import numpy as np
from beer_models.explain import top_indices,stratified_draw,network_summary,marginal_spearman

class ExplainTest(unittest.TestCase):
    def test_top3_excludes_aggregate_and_keeps_negative_association(self):
        np.testing.assert_equal(top_indices([100.,-4.,2.,3.],[1,2,3]),[1,3,2])

    def test_bootstrap_retains_style_counts(self):
        styles=np.array(['A']*3+['B']*7)
        draw=stratified_draw(styles,np.random.default_rng(5))
        self.assertEqual(int((styles[draw]=='A').sum()),3)
        self.assertEqual(int((styles[draw]=='B').sum()),7)

    def test_network_counts_signs_and_isolates_without_self_loops(self):
        a=np.array([[1,.8,0,0],[.8,1,-.7,0],[0,-.7,1,0],[0,0,0,1]])
        info,degree,_,_=network_summary(a,.6)
        self.assertEqual(info['edges'],2)
        self.assertEqual(info['positive_edges'],1)
        self.assertEqual(info['negative_edges'],1)
        self.assertEqual(info['components'],2)
        np.testing.assert_equal(degree,[1,2,1,0])

    def test_marginal_correlation_ranks_complete_pairs(self):
        x=np.array([[1.],[2.],[np.nan],[3.],[4.]])
        y=np.array([[1.,5.],[2.,4.],[3.,3.],[4.,2.],[5.,1.]])
        np.testing.assert_allclose(marginal_spearman(x,y),[[1.],[-1.]])

if __name__=='__main__':unittest.main()
