# -*- coding: utf-8 -*-
"""Extra checks: 4.csv extreme values; per-target high-grade AUC; panel-mean retarget Ridge."""
import os, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

ROOT = r'C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件'
SRC = os.path.join(ROOT, r'2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件')
if not os.path.isdir(SRC):
    SRC = r'C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件'
OUT = os.path.join(ROOT, 'experiments', 'lead_review', 'q1')
R2 = os.path.join(ROOT, 'outputs', 'r2_raw_q1_20261026')
T6 = os.path.join(ROOT, 'outputs', 'task6_final_20261026', 'paper', 'tables')
rep = {}

# 1) 4.csv extreme values
s4 = pd.read_csv(os.path.join(SRC, '4.csv'))
num = s4.select_dtypes(include=[np.number])
ext = {}
for c in num.columns:
    mx = float(np.nanmax(np.abs(num[c].values)))
    if mx > 10:
        ext[c] = {'min': float(num[c].min()), 'max': float(num[c].max()), 'n_abs_gt_10': int((num[c].abs() > 10).sum())}
rep['4csv_extreme_cols'] = ext
print('4.csv extreme cols:', json.dumps(ext, ensure_ascii=False))
rep['4csv_non_numeric_cols'] = [c for c in s4.columns if c not in num.columns]

# 2) per-target AUC for high grade from task6 probabilities
obs = pd.read_csv(os.path.join(R2, 'observed_panel_distribution.csv'))
t6 = pd.read_csv(os.path.join(T6, 'q1__oof_scores_all52.csv'))
t = obs.merge(t6[['beer', 'target', 'probability_rating_ge5', 'probability_rating_6', 'estimated_grade_0_6']], on=['beer', 'target'])
for g, col in [(5, 'probability_rating_ge5'), (6, 'probability_rating_6')]:
    aucs = []
    for tg, gg in t.groupby('target'):
        y = (gg['panel_weighted_median'] >= g).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aucs.append(roc_auc_score(y, gg[col]))
    aucs = np.array(aucs)
    rep[f'panel_pertarget_auc_ge{g}'] = {'n_targets': int(len(aucs)), 'mean': float(aucs.mean()),
                                        'median': float(np.median(aucs)), 'min': float(aucs.min()),
                                        'max': float(aucs.max()), 'n_gt_0.8': int((aucs > 0.8).sum())}

# 3) panel-mean retarget Ridge on chemistry (3-fold OOF, protocol folds)
chem = pd.read_csv(os.path.join(SRC, '1.csv'))
wt = pd.read_csv(os.path.join(R2, 'rating_weights_and_folds.csv'))
fold = wt.groupby('beer')['outer_fold'].first()
excluded = ['kcalperc', 'acids_sum', 'esters_sum', 'aroma_hops_sum', 'sulfur_sum', 'GFS', 'unfermentables']
feat = [c for c in chem.select_dtypes(include=[np.number]).columns if c not in excluded and c != 'beer_id']
X = chem[feat].astype(float).values
beers = chem['beer'].values
styles = pd.get_dummies(chem['tasting_category_fine'], drop_first=True).astype(float).values
folds = np.array([fold[b] for b in beers])
y_mean = t.pivot(index='beer', columns='target', values='panel_weighted_mean')
y_med = t.pivot(index='beer', columns='target', values='panel_weighted_median')
y_mean = y_mean.loc[beers]; y_med = y_med.loc[beers]
targets = y_mean.columns.tolist()
folds_u = sorted(set(folds))

def run_ridge(Xmat, ymat, alpha, name):
    pred = np.full(ymat.shape, np.nan)
    for f in folds_u:
        tr = folds != f; te = folds == f
        imp = SimpleImputer(strategy='median').fit(Xmat[tr]); sc = StandardScaler().fit(imp.transform(Xmat[tr]))
        Xtr = sc.transform(imp.transform(Xmat[tr])); Xte = sc.transform(imp.transform(Xmat[te]))
        for j in range(ymat.shape[1]):
            mdl = Ridge(alpha=alpha).fit(Xtr, ymat[tr, j])
            pred[te, j] = mdl.predict(Xte)
    mae = float(np.nanmean(np.abs(pred - ymat)))
    med = np.round(pred)
    exact = float(np.nanmean(med == y_med.values))
    w1 = float(np.nanmean(np.abs(med - y_med.values) <= 1))
    return {'model': name, 'panel_mean_mae': mae, 'panel_median_exact_rounded': exact,
            'panel_median_within_one_rounded': w1}, pred

rows = []
for alpha in [1.0, 10.0, 30.0, 100.0]:
    m, _ = run_ridge(X, y_mean.values, alpha, f'Ridge(chem) alpha={alpha} -> panel mean')
    rows.append(m)
m, _ = run_ridge(np.hstack([X, styles]), y_mean.values, 10.0, 'Ridge(chem+style) alpha=10 -> panel mean')
rows.append(m)
m, _ = run_ridge(X, y_med.values, 10.0, 'Ridge(chem) alpha=10 -> panel median (round)')
rows.append(m)
rep['D6_ridge_retarget'] = rows
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'ridge_retarget.csv'), index=False)
print(json.dumps(rows, indent=2))

with open(os.path.join(OUT, 'extra_report.json'), 'w', encoding='utf-8') as fh:
    json.dump(rep, fh, indent=2, ensure_ascii=False)
