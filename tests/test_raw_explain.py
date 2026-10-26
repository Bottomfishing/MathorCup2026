"""Raw-scale estimands and held-out explanation invariants."""
import importlib.util
import unittest
import numpy as np
from sklearn.linear_model import Ridge
from beer_models.raw_panel import RawPanelModel
from beer_models.raw_explain import (fit_raw_ridge,explain_linear,explain_quantile,
    check_additivity,nonzero_top,training_modules,module_importance)


class RawExplainTests(unittest.TestCase):
    def test_ridge_balanced_means_equal_weighted_raw_record_fit(self):
        x=np.arange(16,dtype=float).reshape(8,2)
        q=np.zeros((8,1,7));q[:,0,0]=np.linspace(.1,.8,8);q[:,0,6]=1-q[:,0,0]
        fitted=fit_raw_ridge(x,q,10.,1)
        z=fitted[:-1].transform(x)
        expanded=np.repeat(z,2,axis=0)
        y=np.tile([0.,6.],8)[:,None]
        weights=q[:,0,[0,6]].ravel()
        reference=Ridge(alpha=10.).fit(expanded,y,sample_weight=weights)
        np.testing.assert_allclose(fitted.named_steps['model'].coef_,reference.coef_,atol=1e-12)
        np.testing.assert_allclose(fitted.predict(x),reference.predict(z),atol=1e-12)

    def test_linear_contributions_reconstruct_unprocessed_mean(self):
        x=np.arange(20,dtype=float).reshape(10,2)
        q=np.zeros((10,1,7));q[:,:,0]=.3;q[:,:,3]=.7
        model=fit_raw_ridge(x[:7],q[:7],10.,1)
        v,b,c=explain_linear(model,x[7:],x[:5])
        self.assertLess(check_additivity(v,b,c),1e-12)
        with self.assertRaises(ValueError):check_additivity(v,b+1,c)

    def test_zero_effects_and_ineligible_fields_do_not_fill_ranks(self):
        self.assertEqual(nonzero_top(np.array([9.,0.,.1,0.]),[1,2,3]).tolist(),[2])
        self.assertEqual(nonzero_top(np.zeros(4),np.arange(4)).tolist(),[])

    def test_correlated_components_keep_opposite_signed_neighbors(self):
        x=np.column_stack([np.arange(12),-np.arange(12),np.ones(12)])
        groups,edges=training_modules(x,.7)
        self.assertEqual(groups,[[0,1],[2]])
        self.assertEqual(edges,[(0,1,-1.)])

    def test_module_signed_contributions_cancel_before_absolute_importance(self):
        v=np.array([[[1.,-1.,.5]],[[2.,-2.,-.5]]])
        np.testing.assert_allclose(module_importance(v,[[0,1],[2]]),[[0.,.5]])

    @unittest.skipUnless(importlib.util.find_spec('shap'),'Install requirements-r3-lock.txt for Tree SHAP test')
    def test_exact_tree_shap_explains_quantile_before_grade_postprocessing(self):
        x=np.arange(24,dtype=float).reshape(12,2)/10
        scores=(np.arange(8)%5)[:,None]
        q=np.eye(7)[scores]
        protocol=dict(minimum_leaf_beer_weight=1.,parallel_jobs=1,
            gbr=dict(n_estimators=10,learning_rate=.1,max_features=1.,loss='quantile',alpha=.5))
        model=RawPanelModel('gbr_depth1',protocol,1,[]).fit(x[:8],np.arange(8),scores,np.ones(8),q)
        v,b,c=explain_quantile(model,x[8:],x[:8])
        self.assertEqual(v.shape,(4,1,2))
        self.assertLess(check_additivity(v,b,c),1e-6)


if __name__=='__main__':unittest.main()
