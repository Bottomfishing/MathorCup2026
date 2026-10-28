"""Diagnostic: why did frozen thresholds promote 380 cells to >=5?"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r'C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件')
det = ROOT/'outputs/q1_decision_det_20261028'
ev = pd.read_csv(det/'q1__evaluation_long_12500.csv')
th = pd.read_csv(det/'q1__thresholds.csv')
print('threshold rows', len(th), th.estimator_slice.value_counts().to_dict())
print('finite tau_ge5 folds rows', th.query("fold != 'all' and emits_ge5").shape[0])
print('promoted >=5 by target:')
print(ev[ev.grade_0_6 >= 5].groupby('target').size().sort_values(ascending=False).head(15))
print('targets with truth>=5:', ev[ev.panel_median_reference >= 5].groupby('target').size().to_dict())

# check one target in detail
for target in ['co2', 'body', 'overall', 'sweetness', 'bitternes']:
    sub = ev[ev.target == target]
    if sub.empty:
        continue
    print('\n==', target, 'cells', len(sub), 'truth>=5', int((sub.panel_median_reference >= 5).sum()),
          'pred>=5', int((sub.grade_0_6 >= 5).sum()))
    for fold in [0, 1, 2]:
        part = sub[sub.outer_fold == fold]
        print(' fold', fold, 'n', len(part), 'pred>=5', int((part.grade_0_6 >= 5).sum()),
              'tau_ge5', part.tau_ge5.iloc[0], 'p_ge5 range', part.probability_rating_ge5.min(), part.probability_rating_ge5.max())

# recompute exact exceed counts per fold/target using threshold table rates
tr = th[th.fold != 'all']
print('\nfold rows sample (target=co2):')
print(tr[tr.target == 'co2'][['fold', 'r_ge5', 'n_train_ge5', 'k_ge5', 'tau_ge5', 'emits_ge5']].to_string(index=False))
