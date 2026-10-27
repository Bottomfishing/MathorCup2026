# -*- coding: utf-8 -*-
import pandas as pd
p = r'data\reference\zenodo23624915\Raw sensory data anonymized.csv'
try:
    df = pd.read_csv(p, encoding='cp1252')
except Exception as e:
    print('cp1252 fail', e)
    df = pd.read_csv(p, encoding='utf-8')
print(df.shape)
print(df.columns.tolist()[:70])
print(df.head(3).to_string()[:2000])
w = pd.read_csv(r'outputs\r2_raw_q1_20261026\rating_weights_and_folds.csv')
print('sum weights per beer describe:')
print(w.groupby('beer').weight_within_beer.sum().describe())
print('sum all weights:', w.weight_within_beer.sum())
print('outer fold per beer unique max:', w.groupby('beer').outer_fold.nunique().max())
print('taster dtype/values:', sorted(w.taster.unique())[:20])
