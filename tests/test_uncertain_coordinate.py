import unittest
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from beer_models.transforms import RobustAsinh
from beer_models.uncertain_coordinate import (preserve_unknown, reachable_forest_points, logistic_margin_lower_bound,
    logistic_target_crossings, distance_improvement_lower_bound, maximum_nearest_distance)


class UncertainCoordinateTests(unittest.TestCase):
    def test_unknown_is_preserved_not_replaced_by_zero(self):
        source=np.array([1.,np.nan,3.]);target=np.array([2.,0.,5.])
        result=preserve_unknown(source,target,1)
        self.assertTrue(np.isnan(result[1]));self.assertEqual(result[0],2.)
        source[1]=.28
        self.assertEqual(preserve_unknown(source,target,1)[1],.28)

    def test_reachable_float32_states_cover_independent_dense_tree_predictions(self):
        rng=np.random.default_rng(3);x=rng.uniform(0,2,(100,3));y=(x[:,0]>.7)+x[:,1]*x[:,2]
        forest=ExtraTreesRegressor(n_estimators=7,max_depth=5,random_state=3).fit(x,y)
        fixed=np.array([0.,.7,.4]);points=reachable_forest_points(forest,fixed,0,0.,2.)
        boundaries=np.unique(np.r_[points,(points[1:]+points[:-1])/2])
        representatives=np.tile(fixed,(len(boundaries),1));representatives[:,0]=boundaries
        probe=np.tile(fixed,(1000,1));probe[:,0]=rng.uniform(0,2,1000)
        known=forest.apply(representatives);actual=forest.apply(probe)
        self.assertTrue(all(any(np.array_equal(row,state) for state in known) for row in actual))
        with self.assertRaises(ValueError):reachable_forest_points(forest,fixed,0,2.,0.)

    def test_logit_bound_is_below_dense_probability_gap(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(120,3));labels=np.where(x[:,0]+x[:,1] > .7,'Tripel',np.where(x[:,1]<0,'Pils/Lager','Other'))
        pipeline=Pipeline([('impute',SimpleImputer()),('compress',RobustAsinh()),('scale',StandardScaler()),('model',LogisticRegression(C=1))]).fit(x,labels)
        fixed=np.array([0.,2.,0.]);lo,hi=-2.,2.
        bound=logistic_margin_lower_bound(pipeline,fixed,0,lo,hi)
        grid=np.tile(fixed,(1001,1));grid[:,0]=np.linspace(lo,hi,1001)
        prob=pipeline.predict_proba(grid);index=list(pipeline.classes_).index('Tripel')
        gap=prob[:,index]-np.delete(prob,index,axis=1).max(1)
        self.assertLessEqual(bound,gap.min()+1e-12)
        crossings=logistic_target_crossings(pipeline,fixed,0,lo,hi)
        self.assertTrue(((crossings >= lo)&(crossings <= hi)).all())

    def test_geometry_certificates_include_interior_nearest_neighbor_switch(self):
        robust=RobustAsinh().fit(np.array([[-2.,0.],[2.,0.],[-1.,1.],[1.,1.]]))
        refs=robust.transform(np.array([[-1.,0.],[1.,0.]]))
        source=np.array([0.,1.]);target=np.array([0.,0.]);center=refs.mean(0)
        maximum=maximum_nearest_distance(robust,target,refs,0,-1.,1.)
        grid=np.tile(target,(10001,1));grid[:,0]=np.linspace(-1,1,len(grid))
        z=robust.transform(grid)
        actual=np.sqrt(np.square(z[:,None]-refs[None]).mean(2)).min(1).max()
        self.assertAlmostEqual(maximum,actual,places=12)
        bound=distance_improvement_lower_bound(robust,source,target,center,0,-1,1)
        before=np.tile(source,(len(grid),1));before[:,0]=grid[:,0]
        observed=1-np.sqrt(np.square(z-center).mean(1))/np.sqrt(np.square(robust.transform(before)-center).mean(1))
        self.assertLessEqual(bound,observed.min()+1e-12)


if __name__ == '__main__':unittest.main()
