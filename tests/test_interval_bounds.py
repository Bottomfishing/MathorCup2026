import unittest
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from beer_models.interval_bounds import prediction_bounds
from beer_models.transforms import RobustAsinh


class IntervalBoundsTest(unittest.TestCase):
    def test_tree_box_crosses_split_and_point_is_exact(self):
        model=DecisionTreeRegressor(max_depth=1).fit([[-1],[1]],[-2,5])
        np.testing.assert_allclose(prediction_bounds(model,[-1],[1]),[-2,5])
        np.testing.assert_allclose(prediction_bounds(model,[1],[1]),[5,5])

    def test_linear_bounds_match_exhaustive_corners(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(30,2));model=Ridge(alpha=.1).fit(x,2*x[:,0]-3*x[:,1])
        corners=np.array([[-1,-2],[-1,3],[2,-2],[2,3]])
        actual=model.predict(corners)
        np.testing.assert_allclose(prediction_bounds(model,[-1,-2],[2,3]),[actual.min(),actual.max()])

    def test_forest_contains_sampled_predictions_after_transforms(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(60,3));y=np.column_stack([x[:,0]**2,x[:,1]-x[:,2]])
        base=make_pipeline(RobustAsinh(),StandardScaler(),ExtraTreesRegressor(n_estimators=9,random_state=1))
        model=TransformedTargetRegressor(regressor=base,transformer=StandardScaler()).fit(x,y*7+15)
        lo=np.array([-.5,-1,-.2]);hi=np.array([.3,.4,.8]);samples=rng.uniform(lo,hi,size=(300,3))
        for target in range(2):
            low,high=prediction_bounds(model,lo,hi,target);pred=model.predict(samples)[:,target]
            self.assertTrue((pred>=low-1e-10).all() and (pred<=high+1e-10).all())
            point=model.predict(lo[None,:])[0,target]
            np.testing.assert_allclose(prediction_bounds(model,lo,lo,target),[point,point],atol=1e-10)

    def test_invalid_box_and_unsupported_model_rejected(self):
        model=Ridge().fit([[0],[1]],[0,1])
        with self.assertRaises(ValueError):prediction_bounds(model,[2],[1])
        with self.assertRaises(ValueError):prediction_bounds(model,[np.nan],[1])
        with self.assertRaises(ValueError):prediction_bounds(object(),[0],[1])


if __name__=='__main__':unittest.main()
