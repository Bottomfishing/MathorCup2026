# -*- coding: utf-8 -*-
"""Q1 lead-review analysis: answer 5 questions with existing OOF artifacts.
Read-only w.r.t. official outputs; writes only under experiments/lead_review/q1/.
"""
import json, os, re
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

ROOT = r'C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件'
SRC = r'C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件'
OUT = os.path.join(ROOT, 'experiments', 'lead_review', 'q1')
os.makedirs(OUT, exist_ok=True)
RAWCSV = os.path.join(ROOT, 'data', 'reference', 'zenodo23624915', 'Raw sensory data anonymized.csv')
R2 = os.path.join(ROOT, 'outputs', 'r2_raw_q1_20261026')
T6 = os.path.join(ROOT, 'outputs', 'task6_final_20261026', 'paper', 'tables')

def jd(o):
    def conv(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return x
    return json.dumps(o, indent=2, ensure_ascii=False, default=conv)

report = {}

# ---------------- A. missingness evidence ----------------
chem = pd.read_csv(os.path.join(SRC, '1.csv'))
chem_num = chem.select_dtypes(include=[np.number])
miss_rows = []
for c in chem_num.columns:
    n = int(chem_num[c].isna().sum())
    if n: miss_rows.append({'file': '1.csv', 'column': c, 'n_missing': n, 'rate_pct': round(100*n/len(chem), 4)})
chem_miss_total = int(chem_num.isna().sum().sum())
report['A_1csv'] = {'shape': list(chem.shape), 'numeric_cols': int(chem_num.shape[1]),
                    'total_missing_cells': chem_miss_total,
                    'total_rate_pct': round(100*chem_miss_total/chem_num.size, 4),
                    'columns_with_missing': miss_rows}
# excluded derived features per config
excluded = ['kcalperc','acids_sum','esters_sum','aroma_hops_sum','sulfur_sum','GFS','unfermentables']
present_ex = [c for c in excluded if c in chem_num.columns]
inp_cols = [c for c in chem_num.columns if c not in excluded]
inp_miss = int(chem_num[inp_cols].isna().sum().sum())
report['A_1csv']['model_input_cols_after_exclusion'] = len(inp_cols)
report['A_1csv']['model_input_missing_cells'] = inp_miss
report['A_1csv']['excluded_present'] = present_ex

s4 = pd.read_csv(os.path.join(SRC, '4.csv'))
s4_num = s4.select_dtypes(include=[np.number])
report['A_4csv'] = {'shape': list(s4.shape), 'numeric_cols': int(s4_num.shape[1]),
                    'missing_cells': int(s4_num.isna().sum().sum()),
                    'value_min': float(np.nanmin(s4_num.values)), 'value_max': float(np.nanmax(s4_num.values))}
s7 = pd.read_csv(os.path.join(SRC, '7.csv'))
s7_num = s7.select_dtypes(include=[np.number])
m7 = s7_num.isna().sum()
report['A_7csv'] = {'shape': list(s7.shape), 'numeric_cols': int(s7_num.shape[1]),
                    'missing_cells': int(m7.sum()), 'rate_pct': round(100*m7.sum()/s7_num.size, 4),
                    'columns_with_missing': {c: int(v) for c, v in m7[m7 > 0].items()}}
pd.DataFrame(miss_rows).to_csv(os.path.join(OUT, 'missingness_1csv.csv'), index=False)

# ---------------- load labels / predictions ----------------
obs = pd.read_csv(os.path.join(R2, 'observed_panel_distribution.csv'))
oof = pd.read_csv(os.path.join(R2, 'oof_predictions.csv.gz'))
wt = pd.read_csv(os.path.join(R2, 'rating_weights_and_folds.csv'))
t6 = pd.read_csv(os.path.join(T6, 'q1__oof_scores_all52.csv'))
raw = pd.read_csv(RAWCSV, encoding='cp1252')
targets = obs['target'].unique().tolist()
assert len(targets) == 50

fold = wt.groupby('beer')['outer_fold'].first()

# long individual records; record id = row order in raw (0-based -> 1-based).
# raw['Beer'] is a 1..13 panel id, NOT competition beer id; competition beer comes from
# rating_weights_and_folds.csv (raw record -> beer mapping).
raw = raw.reset_index(drop=True)
raw['raw_record_1based'] = np.arange(1, len(raw) + 1)
raw = raw.rename(columns={'A_ester_fruity': 'A_esters_fruity', 'F_ester_fruity': 'F_esters_fruity'})
raw = raw.merge(wt[['raw_record_1based', 'beer', 'weight_within_beer']], on='raw_record_1based', how='left')
assert raw['beer'].notna().all() and raw['weight_within_beer'].notna().all()
raw_long = raw.melt(id_vars=['raw_record_1based', 'beer', 'weight_within_beer'],
                    value_vars=targets, var_name='target', value_name='raw')
assert raw_long['weight_within_beer'].notna().all()
assert abs(raw_long.groupby(['beer', 'target'])['weight_within_beer'].sum().sub(1).abs().max()) < 1e-9
# each target total weight = 250
print('weight mass per target (should be 250):', raw_long.groupby('target')['weight_within_beer'].sum().round(6).unique()[:5])

et = oof[oof.method == 'extra_trees'][['beer', 'outer_fold', 'target', 'predicted_grade']].copy()
tm = oof[oof.method == 'training_median'][['beer', 'outer_fold', 'target', 'predicted_grade']].copy()
_chk = et.merge(t6[t6.target.isin(et['target'].unique())][['beer', 'target', 'estimated_grade_0_6']], on=['beer', 'target'])
assert (_chk['predicted_grade'] == _chk['estimated_grade_0_6']).all()

# individual-level weighted metrics
def indiv_metrics(pred_df, name):
    d = raw_long.merge(pred_df[['beer', 'target', 'predicted_grade']], on=['beer', 'target'], how='left')
    w = d['weight_within_beer'].values
    err = (d['predicted_grade'] - d['raw']).abs().values
    return {'method': name, 'individual_weighted_mae': float(np.sum(w*err)/np.sum(w)),
            'individual_weighted_exact': float(np.sum(w*(err == 0))/np.sum(w)),
            'individual_weighted_within_one': float(np.sum(w*(err <= 1))/np.sum(w)),
            'n_weighted_cells': float(np.sum(w))}

indiv = [indiv_metrics(et, 'extra_trees'), indiv_metrics(tm, 'training_median')]
# oracle: predict observed panel median per cell
o_med = obs[['beer', 'target', 'panel_weighted_median']].rename(columns={'panel_weighted_median': 'predicted_grade'})
indiv.append(indiv_metrics(o_med, 'panel_median_oracle'))
report['B_individual'] = indiv

# ---------------- panel-level metrics ----------------
d = obs.merge(et[['beer', 'target', 'predicted_grade']], on=['beer', 'target'], how='left', suffixes=('', '_et'))
# fold-wise target mean and style-target mean baselines
chem_style = chem[['beer', 'tasting_category_fine']].copy()
d = d.merge(chem_style, on='beer', how='left')
d['outer_fold'] = d['beer'].map(fold)
# target mean baseline (fold-wise over training beers)
mean_base = np.full(len(d), np.nan); style_base = np.full(len(d), np.nan)
for t, g in d.groupby('target'):
    for f in sorted(d['outer_fold'].unique()):
        tr = g[g.outer_fold != f]; te = g.index[g.outer_fold == f]
        mean_base[d.index.get_indexer(te)] = tr['panel_weighted_mean'].mean()
        smap = tr.groupby('tasting_category_fine')['panel_weighted_mean'].mean()
        glob = tr['panel_weighted_mean'].mean()
        style_base[d.index.get_indexer(te)] = g.loc[te, 'tasting_category_fine'].map(smap).fillna(glob).values
d['mean_base'] = mean_base; d['style_base'] = style_base

def panel_metrics(pred, name):
    pred = np.asarray(pred, float)
    y_mean = d['panel_weighted_mean'].values
    y_med = d['panel_weighted_median'].values
    out = {'method': name,
           'panel_mean_mae': float(np.mean(np.abs(pred - y_mean))),
           'panel_median_mae': float(np.mean(np.abs(pred - y_med))),
           'panel_median_exact': float(np.mean(np.round(pred) == y_med)),
           'panel_median_within_one': float(np.mean(np.abs(np.round(pred) - y_med) <= 1)),
           'panel_mean_rmse': float(np.sqrt(np.mean((pred - y_mean)**2)))}
    sp = []
    for t, g in d.groupby('target'):
        if g['panel_weighted_mean'].nunique() < 3: continue
        sp.append(stats.spearmanr(g['predicted_grade'] if name.startswith('extra') else pred[g.index], g['panel_weighted_mean']).correlation)
    out['panel_spearman_macro'] = float(np.nanmean(sp))
    return out

rows = [panel_metrics(d['predicted_grade'], 'extra_trees (OOF integer median)'),
        panel_metrics(d['mean_base'], 'training target mean (fold-wise)'),
        panel_metrics(d['style_base'], 'style x target mean (fold-wise)')]
tm_merge = tm.rename(columns={'predicted_grade': 'pred_tm'})
d2 = obs.merge(tm_merge[['beer', 'target', 'pred_tm']], on=['beer', 'target'], how='left')
rows.append({'method': 'training_median (fold-wise)', 'panel_mean_mae': float(np.mean(np.abs(d2['pred_tm'] - d2['panel_weighted_mean']))),
             'panel_median_mae': float(np.mean(np.abs(d2['pred_tm'] - d2['panel_weighted_median']))),
             'panel_median_exact': float(np.mean(d2['pred_tm'] == d2['panel_weighted_median'])),
             'panel_median_within_one': float(np.mean(np.abs(d2['pred_tm'] - d2['panel_weighted_median']) <= 1)),
             'panel_mean_rmse': float(np.sqrt(np.mean((d2['pred_tm'] - d2['panel_weighted_mean'])**2))),
             'panel_spearman_macro': float(np.nanmean([stats.spearmanr(g['pred_tm'], g['panel_weighted_mean']).correlation for t, g in d2.groupby('target') if g['panel_weighted_mean'].nunique() > 2]))})
rows.append({'method': 'panel_median oracle (upper ref.)', 'panel_mean_mae': float(np.mean(np.abs(d['panel_weighted_median'] - d['panel_weighted_mean']))),
             'panel_median_mae': 0.0, 'panel_median_exact': 1.0, 'panel_median_within_one': 1.0,
             'panel_mean_rmse': float(np.sqrt(np.mean((d['panel_weighted_median'] - d['panel_weighted_mean'])**2))),
             'panel_spearman_macro': 1.0})
panel_tab = pd.DataFrame(rows)
panel_tab.to_csv(os.path.join(OUT, 'panel_level_metrics.csv'), index=False)
report['B_panel'] = rows

# panel median distribution
dist = d['panel_weighted_median'].value_counts().sort_index()
report['B_panel_median_distribution'] = {int(k): int(v) for k, v in dist.items()}
pd.DataFrame({'grade': dist.index, 'n_cells': dist.values}).to_csv(os.path.join(OUT, 'panel_median_distribution.csv'), index=False)

# confusion matrices
for nm, pred in [('extra_trees', d['predicted_grade']), ('training_median', d2['pred_tm'])]:
    cm = pd.crosstab(pd.Series(d['panel_weighted_median'].values, name='observed_panel_median'),
                     pd.Series(pred.values, name='predicted_grade'))
    cm.to_csv(os.path.join(OUT, f'confusion_panel_{nm}.csv'))
    report.setdefault('B_confusion', {})[nm] = cm.to_dict()

# high-grade ranking quality (available probability columns: ge1, ge5, 6 only)
t6m = obs.merge(t6[t6.target.isin(targets)][['beer', 'target', 'probability_rating_ge5', 'probability_rating_6']], on=['beer', 'target'])
prob_col = {5: 'probability_rating_ge5', 6: 'probability_rating_6'}
hg = {}
for g in [5, 6]:
    y = (t6m['panel_weighted_median'] >= g).astype(int).values
    p = t6m[prob_col[g]].values
    hg[f'panel_pooled_auc_ge{g}'] = float(roc_auc_score(y, p)) if y.sum() > 0 else None
    per = []
    for t, gg in t6m.groupby('target'):
        yy = (gg['panel_weighted_median'] >= g).astype(int)
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        per.append(roc_auc_score(yy, gg[prob_col[g]]))
    hg[f'panel_macro_auc_ge{g}'] = float(np.nanmean(per)) if per else None
    hg[f'n_positive_cells_ge{g}'] = int(y.sum())
    # precision@5 per target (only targets with >=1 positive among top5 count)
    prec, n_top = [], []
    for t, gg in t6m.groupby('target'):
        top = gg.nlargest(5, prob_col[g])
        prec.append(float((top['panel_weighted_median'] >= g).mean()))
    hg[f'panel_precision_at5_ge{g}'] = float(np.nanmean(prec))
# individual-level weighted AUC
il = raw_long.merge(t6[t6.target.isin(targets)][['beer', 'target', 'probability_rating_ge5', 'probability_rating_6']], on=['beer', 'target'])
for g in [5, 6]:
    y = (il['raw'] >= g).astype(int).values
    hg[f'individual_weighted_auc_ge{g}'] = float(roc_auc_score(y, il[prob_col[g]].values, sample_weight=il['weight_within_beer'].values))
    hg[f'individual_n_positive_ge{g}'] = int(y.sum())
# current model high-grade behaviour at panel level
for g in [5, 6]:
    hg[f'et_n_predicted_ge{g}_panel'] = int((d['predicted_grade'] >= g).sum())
    hg[f'et_panel_recall_ge{g}'] = float(((d['predicted_grade'] >= g) & (d['panel_weighted_median'] >= g)).sum() / max((d['panel_weighted_median'] >= g).sum(), 1))
    hg[f'et_panel_precision_ge{g}'] = float(((d['predicted_grade'] >= g) & (d['panel_weighted_median'] >= g)).sum() / max((d['predicted_grade'] >= g).sum(), 1))
report['B_high_grade'] = hg

# ---------------- E. variance decomposition / floor ----------------
# weighted ICC per target
icc = []
for t, g in raw_long.groupby('target'):
    w = g['weight_within_beer'].values; y = g['raw'].values.astype(float)
    mu = np.sum(w*y)/np.sum(w)
    bw = pd.Series(w).groupby(g['beer'].values).sum()
    bm = pd.Series(w*y).groupby(g['beer'].values).sum()/bw
    between = float(np.sum(bw*(bm - mu)**2)/np.sum(w))
    within = float(np.sum(w*(y - g['beer'].map(bm).values)**2)/np.sum(w))
    icc.append({'target': t, 'icc': between/(between+within) if between+within > 0 else np.nan,
                'between_var': between, 'within_var': within})
icc = pd.DataFrame(icc)
icc.to_csv(os.path.join(OUT, 'variance_decomposition_icc.csv'), index=False)
report['E_variance'] = {'mean_icc': float(icc['icc'].mean()), 'median_icc': float(icc['icc'].median()),
                        'mean_within_share': float((icc['within_var']/(icc['between_var']+icc['within_var'])).mean()),
                        'n_targets_icc_gt_0.5': int((icc['icc'] > 0.5).sum())}

# ---------------- C. z (4.csv) -> raw 0-6 recoverability + anchor calibration ----------------
s4l = s4.melt(id_vars=['beer'], value_vars=[c for c in s4.columns if c in targets or c == '4vg'], var_name='target', value_name='z')
rename = {}
for c in s4l['target'].unique():
    if c == '4vg': rename[c] = 'X4vg'
s4l['target'] = s4l['target'].replace(rename)
assert set(s4l.target) == set(targets), (set(s4l.target) - set(targets), set(targets) - set(s4l.target))
z = obs.merge(s4l, on=['beer', 'target'], how='left')
cal = []
rs = np.random.RandomState(20261024)
for t, g in z.groupby('target'):
    if g['z'].std() == 0 or g['panel_weighted_mean'].std() == 0:
        continue
    row = {'target': t, 'pearson_z_vs_panelmean': float(stats.pearsonr(g['z'], g['panel_weighted_mean'])[0]),
           'spearman_z_vs_panelmean': float(stats.spearmanr(g['z'], g['panel_weighted_mean'])[0]),
           'pearson_z_vs_panelmedian': float(stats.pearsonr(g['z'], g['panel_weighted_median'])[0]),
           'const_baseline_mae': float(np.mean(np.abs(g['panel_weighted_mean'] - g['panel_weighted_mean'].mean())))}
    g = g.reset_index(drop=True)
    for k in (10, 12):
        iso_maes, lin_maes, grade_exact, grade_w1 = [], [], [], []
        for _ in range(20):
            idx = rs.choice(len(g), size=k, replace=False)
            tr = g.iloc[idx]; te = g.drop(index=idx)
            iso = IsotonicRegression(out_of_bounds='clip').fit(tr['z'], tr['panel_weighted_mean'])
            pi = iso.predict(te['z'])
            iso_maes.append(np.mean(np.abs(pi - te['panel_weighted_mean'])))
            lin = LinearRegression().fit(tr[['z']], tr['panel_weighted_mean'])
            lin_maes.append(np.mean(np.abs(lin.predict(te[['z']]) - te['panel_weighted_mean'])))
            grade_exact.append(np.mean(np.round(pi) == te['panel_weighted_median']))
            grade_w1.append(np.mean(np.abs(np.round(pi) - te['panel_weighted_median']) <= 1))
        row[f'iso_k{k}_mae'] = float(np.mean(iso_maes))
        row[f'lin_k{k}_mae'] = float(np.mean(lin_maes))
        row[f'iso_k{k}_grade_exact'] = float(np.mean(grade_exact))
        row[f'iso_k{k}_grade_within1'] = float(np.mean(grade_w1))
    cal.append(row)
cal = pd.DataFrame(cal)
cal.to_csv(os.path.join(OUT, 'calibration_z_to_raw.csv'), index=False)
report['C_calibration'] = {
    'n_targets': int(len(cal)),
    'mean_pearson_z_panelmean': float(cal['pearson_z_vs_panelmean'].mean()),
    'median_pearson': float(cal['pearson_z_vs_panelmean'].median()),
    'n_pearson_gt_0.9': int((cal['pearson_z_vs_panelmean'] > 0.9).sum()),
    'n_pearson_gt_0.99': int((cal['pearson_z_vs_panelmean'] > 0.99).sum()),
    'const_baseline_mae_mean': float(cal['const_baseline_mae'].mean()),
    'iso_k10_mae_mean': float(cal['iso_k10_mae'].mean()),
    'iso_k12_mae_mean': float(cal['iso_k12_mae'].mean()),
    'lin_k10_mae_mean': float(cal['lin_k10_mae'].mean()),
    'lin_k12_mae_mean': float(cal['lin_k12_mae'].mean()),
    'iso_k10_grade_exact_mean': float(cal['iso_k10_grade_exact'].mean()),
    'iso_k10_grade_within1_mean': float(cal['iso_k10_grade_within1'].mean()),
    'iso_k12_grade_exact_mean': float(cal['iso_k12_grade_exact'].mean()),
    'iso_k12_grade_within1_mean': float(cal['iso_k12_grade_within1'].mean()),
}

# ---------------- D. improvement experiments (all fold-wise, no retraining) ----------------
dd = d.copy()

def panel_metrics_named(df, predcol, name):
    y_mean = df['panel_weighted_mean'].values
    y_med = df['panel_weighted_median'].values
    pred = df[predcol].values.astype(float)
    yb = (y_med >= 5).astype(int); pb = (pred >= 5).astype(int)
    tp = int(((yb == 1) & (pb == 1)).sum()); fp = int(((yb == 0) & (pb == 1)).sum()); fn = int(((yb == 1) & (pb == 0)).sum())
    return {'method': name, 'panel_mean_mae': float(np.mean(np.abs(pred - y_mean))),
            'panel_median_mae': float(np.mean(np.abs(pred - y_med))),
            'panel_median_exact_rounded': float(np.mean(np.round(pred) == y_med)),
            'panel_median_within_one_rounded': float(np.mean(np.abs(np.round(pred) - y_med) <= 1)),
            'panel_mean_rmse': float(np.sqrt(np.mean((pred - y_mean)**2))),
            'n_pred_ge5': int((pred >= 5).sum()),
            'recall_ge5': float(tp / max(tp + fn, 1)), 'precision_ge5': float(tp / max(tp + fp, 1))}

imp = [panel_metrics_named(dd, 'predicted_grade', 'ET integer median (current)')]
# D2: fold-wise per-target linear calibration ET integer -> panel mean
cal2 = np.full(len(dd), np.nan)
for t, g in dd.groupby('target'):
    for f in sorted(dd['outer_fold'].unique()):
        tr = g[g.outer_fold != f]; te = g.index[g.outer_fold == f]
        lr = LinearRegression().fit(tr[['predicted_grade']], tr['panel_weighted_mean'])
        cal2[dd.index.get_indexer(te)] = lr.predict(dd.loc[te, ['predicted_grade']])
dd['et_cal'] = cal2
imp.append(panel_metrics_named(dd, 'et_cal', 'ET integer + fold-wise linear calibration -> panel mean'))
# D3: fold-wise per-target isotonic calibration ET integer -> panel mean
cal3 = np.full(len(dd), np.nan)
for t, g in dd.groupby('target'):
    for f in sorted(dd['outer_fold'].unique()):
        tr = g[g.outer_fold != f]; te = g.index[g.outer_fold == f]
        if tr['predicted_grade'].nunique() < 2:
            cal3[dd.index.get_indexer(te)] = tr['panel_weighted_mean'].mean()
            continue
        iso = IsotonicRegression(out_of_bounds='clip').fit(tr['predicted_grade'], tr['panel_weighted_mean'])
        cal3[dd.index.get_indexer(te)] = iso.predict(dd.loc[te, 'predicted_grade'])
dd['et_iso'] = cal3
imp.append(panel_metrics_named(dd, 'et_iso', 'ET integer + fold-wise isotonic calibration -> panel mean'))
# D4: ET + style mean, fold-wise linear (2 features)
X = dd[['predicted_grade']].copy(); X['style_base'] = dd['style_base']
pred = np.full(len(dd), np.nan)
for t, g in dd.groupby('target'):
    for f in sorted(dd['outer_fold'].unique()):
        tr = g[g.outer_fold != f]; te = g.index[g.outer_fold == f]
        lr = LinearRegression().fit(X.loc[tr.index], tr['panel_weighted_mean'])
        pred[dd.index.get_indexer(te)] = lr.predict(X.loc[te])
dd['et_style'] = pred
imp.append(panel_metrics_named(dd, 'et_style', 'ET integer + style mean, fold-wise linear'))
# D5: marginal (quantile) calibration of predicted grade to observed panel-median distribution
# For each target/fold: rank test prediction among train predictions -> empirical train panel-median quantile.
qm = np.full(len(dd), np.nan)
for t, g in dd.groupby('target'):
    for f in sorted(dd['outer_fold'].unique()):
        tr = g[g.outer_fold != f]; te = g.index[g.outer_fold == f]
        tp = np.sort(tr['predicted_grade'].values)
        obsmed = np.sort(tr['panel_weighted_median'].values)
        p = dd.loc[te, 'predicted_grade'].values
        r = (np.searchsorted(tp, p, side='left') + np.searchsorted(tp, p, side='right')) / (2.0 * len(tp))
        q = np.quantile(obsmed, np.clip(r, 0, 1), method='lower')
        qm[dd.index.get_indexer(te)] = q
dd['et_quantile'] = qm
imp.append(panel_metrics_named(dd, 'et_quantile', 'ET integer + fold-wise marginal (quantile) calibration'))
imp = pd.DataFrame(imp)
imp.to_csv(os.path.join(OUT, 'improvement_experiments.csv'), index=False)
report['D_improvements'] = imp.to_dict(orient='records')

# candidate deliverable: calibrated panel mean + rounded grade for 50 targets
cand = dd[['beer', 'target', 'panel_weighted_mean', 'panel_weighted_median', 'predicted_grade', 'et_cal']].copy()
cand = cand.rename(columns={'panel_weighted_mean': 'observed_panel_mean_for_ref',
                            'panel_weighted_median': 'observed_panel_median_for_ref',
                            'predicted_grade': 'pred_grade_current',
                            'et_cal': 'pred_panel_mean_calibrated'})
cand['pred_grade_calibrated_round'] = np.round(cand['pred_panel_mean_calibrated']).clip(0, 6).astype(int)
cand.to_csv(os.path.join(OUT, 'candidate_panel_mean_and_grade_250x50.csv'), index=False)
report['D_candidate_cols'] = list(cand.columns)

# ---------------- write report json ----------------
with open(os.path.join(OUT, 'analysis_report.json'), 'w', encoding='utf-8') as fh:
    fh.write(jd(report))
print(jd(report))
