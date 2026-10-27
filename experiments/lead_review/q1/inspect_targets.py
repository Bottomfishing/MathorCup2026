# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
obs = pd.read_csv(r'outputs\r2_raw_q1_20261026\observed_panel_distribution.csv')
print(obs.shape, obs.target.nunique())
print(obs.target.tolist()[:60])
print('panel median dtype/unique:', obs.panel_weighted_median.dtype, sorted(obs.panel_weighted_median.unique())[:20], 'n_unique', obs.panel_weighted_median.nunique())
print('counts of panel median>=5:', (obs.panel_weighted_median >= 5).sum(), '>=6:', (obs.panel_weighted_median >= 6).sum())
print('panel mean range', obs.panel_weighted_mean.min(), obs.panel_weighted_mean.max())
t6 = pd.read_csv(r'outputs\task6_final_20261026\paper\tables\q1__oof_scores_all52.csv')
print('t6 targets:', t6.target.nunique())
print('t6 targets not in obs:', sorted(set(t6.target.dropna()) - set(obs.target)))
print('obs targets not in t6:', sorted(set(obs.target) - set(t6.target.dropna())))
r2 = pd.read_csv(r'outputs\r2_raw_q1_20261026\oof_predictions.csv.gz')
et = r2[r2.method == 'extra_trees'][['beer', 'target', 'predicted_grade']]
m = t6.merge(et, on=['beer', 'target'], how='inner')
print('t6-vs-r2 et mismatch count:', (m.estimated_grade_0_6 != m.predicted_grade).sum(), 'of', len(m))
print(m.head())
