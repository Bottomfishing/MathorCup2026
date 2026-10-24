"""Verification of validation boundaries; AI-assisted by OpenAI Codex, 2026-10-23."""
import json
from pathlib import Path
import unittest
import numpy as np
from beer_models.baseline import make_splits,make_pipeline,mask_observed,metric_vectors
from beer_models.transforms import RobustAsinh


class BaselineTest(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).resolve().parents[1]/'configs/baseline_v1.json').read_text())

    def test_each_validation_sample_once_and_all_styles_present(self):
        styles=['a']*3+['b']*4+['c']*9
        splits=make_splits(styles,3,42)
        seen=[]
        for train,test in splits:
            self.assertFalse(set(train)&set(test))
            self.assertEqual(set(np.array(styles)[test]),{'a','b','c'})
            seen.extend(test)
        self.assertEqual(sorted(seen),list(range(len(styles))))

    def test_five_folds_rejected_for_rare_style(self):
        with self.assertRaises(ValueError):make_splits(['a']*3+['b']*10,5,42)

    def test_nested_validation_does_not_touch_outer_test(self):
        styles=np.array(['a']*6+['b']*9)
        for train,test in make_splits(styles,3,7):
            for it,iv in make_splits(styles[train],2,9):
                self.assertFalse(set(train[it])&set(test))
                self.assertFalse(set(train[iv])&set(test))

    def test_imputation_and_scaling_ignore_validation_values(self):
        pipe,_=make_pipeline('ridge',self.config,7)
        pipe.fit(np.array([[0.],[2.],[np.nan]]),np.array([[0.,1.],[1.,2.],[.5,1.5]]))
        before=pipe.named_steps['impute'].statistics_.copy()
        pipe.predict(np.array([[1000000.],[np.nan]]))
        np.testing.assert_allclose(before,[1.])
        np.testing.assert_allclose(pipe.named_steps['impute'].statistics_,before)
        np.testing.assert_allclose(pipe.named_steps['scale'].mean_,[1.])

    def test_mask_does_not_modify_input_or_restore_missing(self):
        x=np.array([[1.,np.nan],[2.,3.]])
        result,mask=mask_observed(x,1.,42)
        self.assertTrue(np.isnan(result).all())
        self.assertEqual(int(mask.sum()),3)
        self.assertEqual(x[0,0],1.)
        small,_=mask_observed(x,0.,42)
        np.testing.assert_equal(small,x)

    def test_masks_are_nested(self):
        x=np.ones((100,5))
        _,m1=mask_observed(x,.05,9);_,m2=mask_observed(x,.2,9)
        self.assertTrue(np.all(~m1|m2))

    def test_metrics_known_values_and_constant_target(self):
        y=np.array([[0.,1.],[2.,1.]])
        p=np.array([[1.,0.],[1.,0.]])
        m=metric_vectors(y,p)
        np.testing.assert_allclose(m['mae'],[1.,1.])
        np.testing.assert_allclose(m['rmse'],[1.,1.])
        self.assertEqual(m['r2'][0],0.)
        self.assertTrue(np.isnan(m['r2'][1]))

    def test_nan_prediction_is_failure(self):
        with self.assertRaises(ValueError):metric_vectors(np.ones((2,2)),np.full((2,2),np.nan))

    def test_robust_compression_is_train_only_and_monotone(self):
        t=RobustAsinh().fit([[0.,1.],[1.,1.],[2.,1.]])
        center=t.center_.copy()
        result=t.transform([[3.,1.],[12596.,1.]])
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(result[1,0],result[0,0])
        self.assertLess(result[1,0],20.)
        np.testing.assert_equal(t.center_,center)
        np.testing.assert_equal(result[:,1],[0.,0.])

    def test_robust_pipeline_does_not_need_target_values_at_prediction(self):
        x=np.array([[0.,1.],[1.,1.],[2.,np.nan],[4.,1.]])
        p,_=make_pipeline('ridge_asinh',self.config,7)
        p.fit(x,np.array([[1.,2.],[2.,3.],[3.,4.],[4.,5.]]))
        self.assertTrue(np.isfinite(p.predict([[12596.,1.]])).all())

if __name__=='__main__':unittest.main()
