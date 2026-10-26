"""Tail-score tradeoffs and non-safety semantics of rating risk estimates."""
import unittest
import numpy as np
import pandas as pd

from beer_models.tail_risk import (soft_logit, binary_predict, binary_loss, cost_decision,
    tail_metrics, feasible, RawOrdinalModel, SelectedRiskModel, fit_risk, risk_frame, survival)


class FixedDistribution(RawOrdinalModel):
    def __init__(self, p):
        self.p = p

    def predict_distribution(self, x):
        return np.broadcast_to(self.p, (len(x),)+self.p.shape)


class TailRiskTests(unittest.TestCase):
    def test_fractional_label_calibration_preserves_beer_weight(self):
        x = np.zeros((4,1))
        model = soft_logit(x, np.array([0,.25,.5,1.]), C=1.)
        np.testing.assert_allclose(binary_predict(model,x), .4375)
        # Expected loss to raw binary ratings includes their within-beer variation.
        np.testing.assert_allclose(binary_loss(.5,.25),.25)
        self.assertNotEqual(float(binary_loss(.5,.25)),float((.5-.25)**2))

    def test_cost_decision_exposes_error_tradeoff_instead_of_relabelling_median(self):
        p = np.zeros((1,1,7));p[0,0,0]=.8;p[0,0,6]=.2
        median,_ = cost_decision(p,[1]*7,[])
        tail,_ = cost_decision(p,[1,1,1,1,1,4,8],[])
        self.assertEqual(median.item(),0);self.assertEqual(tail.item(),6)
        self.assertGreater(tail_metrics(p,tail)['raw_rating_mae'],tail_metrics(p,median)['raw_rating_mae'])

    def test_zero_high_recall_does_not_qualify_despite_low_average_error(self):
        limits=dict(maximum_raw_mae_increase=.02,minimum_each_raw_grade5_grade6_recall=.05,
            minimum_each_panel_median5_median6_recall=.05,maximum_raw_high_false_positive_rate=.02,
            maximum_panel_high_false_positive_rate=.02)
        m=dict(raw_rating_mae=.1,raw_recall_5=0.,raw_recall_6=0.,panel_recall_5=0.,panel_recall_6=0.,
               raw_high_fpr=0.,panel_high_fpr=0.)
        self.assertFalse(feasible(m,.2,limits))

    def test_risk_source_falls_back_to_training_prior_when_predictors_add_no_information(self):
        truth=np.zeros((8,1,7));truth[:,:,0]=.75;truth[:,:,1]=.25
        poor=np.zeros_like(truth);poor[:,:,0]=.2;poor[:,:,1]=.8
        protocol=dict(base_models=['extra_trees','raw_ordinal'],calibration_epsilon=1e-6,calibration_C=1.)
        models={name:FixedDistribution(poor[0]) for name in protocol['base_models']}
        risk,rows=fit_risk(models,dict(prior=truth,extra_trees=poor,raw_ordinal=poor),truth,protocol)
        self.assertEqual(risk.choices,['prior'])
        np.testing.assert_allclose(risk.predict_distribution(np.ones((3,1))),np.tile(truth[0],(3,1,1)))
        self.assertEqual([r['source'] for r in rows if r['selected']],['prior'])

    def test_calibration_output_remains_an_ordered_distribution(self):
        prior=np.zeros((1,7));prior[0,0]=.5;prior[0,6]=.5
        model=FixedDistribution(prior)
        risk=SelectedRiskModel({'extra_trees':model},['extra_trees'],[[.2,.8,.1,.1,.05,.02]],prior,
                               {'calibration_epsilon':1e-6})
        p=risk.predict_distribution(np.zeros((2,1)))
        self.assertTrue((p>=0).all());np.testing.assert_allclose(p.sum(2),1.)
        self.assertTrue((np.diff(survival(p),axis=2)<=1e-12).all())

    def test_risk_table_does_not_certify_safety_even_for_prior_probability_zero(self):
        prior=np.zeros((1,7));prior[0,0]=1
        model=FixedDistribution(prior)
        risk=SelectedRiskModel({'extra_trees':model},['prior'],[[0.]*6],prior,{'calibration_epsilon':1e-6})
        bundle=dict(features=['x'],targets=['stale_hops'],risk_panel_model=risk,risk_status=['insufficient_support'])
        result=risk_frame(pd.DataFrame({'x':[1.]}),bundle,['b'])
        self.assertFalse(result.safety_certified.any())
        self.assertEqual(result.probability_rating_ge1.iloc[0],0.)
        self.assertEqual(result.risk_validation_status.iloc[0],'insufficient_support')


if __name__=='__main__':unittest.main()
