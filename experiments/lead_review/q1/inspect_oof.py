# -*- coding: utf-8 -*-
"""Inspect r2_raw_q1 OOF + task6 Q1 tables."""
import os
import pandas as pd

ROOT = r'C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件'
os.chdir(ROOT)

oof = pd.read_csv(r'outputs\r2_raw_q1_20261026\oof_predictions.csv.gz')
print('== r2_raw oof', oof.shape)
print('methods:', oof.method.unique().tolist())
print('rows per method:', oof.groupby('method').size().to_dict())
print(oof.groupby('method')[['weighted_raw_absolute_loss']].mean())
print('targets:', oof.target.nunique(), 'beers:', oof.beer.nunique())
w = pd.read_csv(r'outputs\r2_raw_q1_20261026\rating_weights_and_folds.csv')
print('== rating_weights_and_folds', w.shape)
print(w.columns.tolist())
print(w.head(5).to_string())
print(w.describe().to_string())

t = r'outputs\task6_final_20261026\paper\tables'
for f in ['validation__q1_metrics_by_repeat.csv', 'validation__q1_repeat_summary.csv',
          'q1__point_comparison.csv', 'q1__point_uncertainty.csv']:
    p = os.path.join(t, f)
    print('==', f)
    df = pd.read_csv(p, encoding='utf-8-sig')
    print(df.shape)
    print(df.to_string())

# task6 oof
o = pd.read_csv(os.path.join(t, 'q1__oof_scores_all52.csv'))
print('== task6 oof', o.shape, 'nonnull grade', o.estimated_grade_0_6.notna().sum())
print('targets', o.target.nunique(), 'beers', o.beer.nunique())
print(o.groupby('target').estimated_grade_0_6.apply(lambda s: s.notna().sum()).sort_values().head(5))
