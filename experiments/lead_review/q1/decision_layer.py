# -*- coding: utf-8 -*-
"""Decision-layer Q1 table: threshold cumulative probabilities (p>=1, p>=5, p=6)
instead of integer conditional median. Also resolves the 0.694 vs 0.807 AUC gap."""
import os, json
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score

ROOT = r'C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件'
OUT = os.path.join(ROOT, 'experiments', 'lead_review', 'q1')
R2 = os.path.join(ROOT, 'outputs', 'r2_raw_q1_20261026')
T6 = os.path.join(ROOT, 'outputs', 'task6_final_20261026', 'paper', 'tables')
raw = pd.read_csv(os.path.join(ROOT, 'data', 'reference', 'zenodo23624915', 'Raw sensory data anonymized.csv'), encoding='cp1252')
raw = raw.rename(columns={'A_ester_fruity': 'A_esters_fruity', 'F_ester_fruity': 'F_esters_fruity'})
wt = pd.read_csv(os.path.join(R2, 'rating_weights_and_folds.csv'))
obs = pd.read_csv(os.path.join(R2, 'observed_panel_distribution.csv'))
t6 = pd.read_csv(os.path.join(T6, 'q1__oof_scores_all52.csv'))
oof = pd.read_csv(os.path.join(R2, 'oof_predictions.csv.gz'))

targets = obs['target'].unique().tolist()
raw['raw_record_1based'] = np.arange(1, len(raw) + 1)
raw = raw.merge(wt[['raw_record_1based', 'beer', 'weight_within_beer']], on='raw_record_1based', how='left')
raw_long = raw.melt(id_vars=['raw_record_1based', 'beer', 'weight_within_beer'],
                    value_vars=targets, var_name='target', value_name='raw')
fold = wt.groupby('beer')['outer_fold'].first()
et = oof[oof.method == 'extra_trees'][['beer', 'target', 'predicted_grade', 'unconstrained_grade']].copy()
et['outer_fold'] = et['beer'].map(fold)
t = obs.merge(t6[['beer', 'target', 'probability_rating_ge1', 'probability_rating_ge5', 'probability_rating_6']], on=['beer', 'target'])
t = t.merge(et[['beer', 'target', 'predicted_grade', 'unconstrained_grade']], on=['beer', 'target'])
t['outer_fold'] = t['beer'].map(fold)
assert len(t) == 12500

rep = {}

# ---------- A. AUC discrepancy resolution ----------
def macro_auc(df, score, ycol, group='target'):
    aucs = []
    for tg, g in df.groupby(group):
        y = g[ycol].astype(int)
        if y.nunique() < 2: continue
        aucs.append(roc_auc_score(y, g[score]))
    return float(np.mean(aucs)), int(len(aucs))

# individual-level long table with weights
il = raw_long.merge(t6[['beer', 'target', 'probability_rating_ge1', 'probability_rating_ge5', 'probability_rating_6']],
                    on=['beer', 'target'])
il = il.merge(et[['beer', 'target', 'predicted_grade', 'unconstrained_grade']], on=['beer', 'target'])
il['y5'] = (il['raw'] >= 5).astype(int)
il['y6'] = (il['raw'] >= 6).astype(int)
t['y5'] = (t['panel_weighted_median'] >= 5).astype(int)
t['y6'] = (t['panel_weighted_median'] >= 6).astype(int)
for name, score in [('prob_ge5', 'probability_rating_ge5'), ('prob_6', 'probability_rating_6'),
                    ('et_unconstrained', 'unconstrained_grade')]:
    for g in [5, 6]:
        ycol = f'y{g}'
        mac_p, n_p = macro_auc(t, score, ycol)
        pooled_p = roc_auc_score(t[ycol], t[score])
        mac_i, n_i = macro_auc(il, score, ycol)
        pooled_i = roc_auc_score(il[ycol], il[score], sample_weight=il['weight_within_beer'])
        rep[f'AUC_{name}_ge{g}'] = {'panel_macro': mac_p, 'panel_pooled': float(pooled_p), 'n_panel_targets': n_p,
                                    'individual_macro': mac_i, 'individual_pooled_weighted': float(pooled_i), 'n_ind_targets': n_i}

# ---------- B. decision-layer tables ----------
base = t[['beer', 'target', 'panel_weighted_mean', 'panel_weighted_median', 'outer_fold',
          'probability_rating_ge1', 'probability_rating_ge5', 'probability_rating_6',
          'predicted_grade', 'unconstrained_grade']].copy()
base = base.rename(columns={'predicted_grade': 'old_grade'})

def override(base, tau5, tau6):
    """tau5/tau6: dict (target,fold)->threshold; grade=6 if p6>=tau6, elif p5>=tau5 ->5, else old grade."""
    g = base['old_grade'].astype(int).values.copy()
    p5 = base['probability_rating_ge5'].values; p6 = base['probability_rating_6'].values
    t5 = np.array([tau5[(r.target, r.outer_fold)] for r in base.itertuples()])
    t6h = np.array([tau6[(r.target, r.outer_fold)] for r in base.itertuples()])
    hit6 = p6 >= t6h; hit5 = (p5 >= t5) & ~hit6
    g[hit5] = np.maximum(g[hit5], 5)
    g[hit6] = 6
    return g

# thresholds: (a) marginal-match per target/fold; (b) F1-optimal per target/fold
tau5_match, tau6_match, tau5_f1, tau6_f1 = {}, {}, {}, {}
for (tg, f), g in base.groupby(['target', 'outer_fold']):
    tr = g  # threshold chosen on the same fold is leaky; instead use the other folds below
for (tg, f), g in base.groupby(['target', 'outer_fold']):
    tr = base[(base.target == tg) & (base.outer_fold != f)]
    # marginal match: choose quantile so that predicted count matches observed rate in training folds
    r5 = float((tr['panel_weighted_median'] >= 5).mean()); r6 = float((tr['panel_weighted_median'] >= 6).mean())
    tau5_match[(tg, f)] = float(np.quantile(tr['probability_rating_ge5'], 1 - r5)) if r5 > 0 else np.inf
    tau6_match[(tg, f)] = float(np.quantile(tr['probability_rating_6'], 1 - r6)) if r6 > 0 else np.inf
    # F1-optimal
    best = (-1.0, np.inf)
    for th in np.unique(np.round(tr['probability_rating_ge5'].values, 4)):
        f1 = f1_score((tr['panel_weighted_median'] >= 5).astype(int), (tr['probability_rating_ge5'] >= th).astype(int), zero_division=0)
        if f1 > best[0]: best = (f1, th)
    tau5_f1[(tg, f)] = best[1] if best[0] > 0 else np.inf
    best6 = (-1.0, np.inf)
    for th in np.unique(np.round(tr['probability_rating_6'].values, 4)):
        f1 = f1_score((tr['panel_weighted_median'] >= 6).astype(int), (tr['probability_rating_6'] >= th).astype(int), zero_division=0)
        if f1 > best6[0]: best6 = (f1, th)
    tau6_f1[(tg, f)] = best6[1] if best6[0] > 0 else np.inf

base['new_grade_match'] = override(base, tau5_match, tau6_match)
base['new_grade_f1'] = override(base, tau5_f1, tau6_f1)

def eval_table(df, col, name):
    g = df[col].values.astype(float)
    med = df['panel_weighted_median'].values; mean = df['panel_weighted_mean'].values
    y5 = med >= 5; y6 = med >= 6
    p5 = g >= 5; p6 = g >= 6
    sp = []
    for tg, gg in df.groupby('target'):
        if gg['panel_weighted_mean'].nunique() > 2:
            sp.append(stats.spearmanr(gg[col], gg['panel_weighted_mean']).correlation)
    def pr(y, p):
        tp = int((y & p).sum()); return int(p.sum()), float(tp / max(p.sum(), 1)), float(tp / max(y.sum(), 1))
    n5, prec5, rec5 = pr(y5, p5); n6, prec6, rec6 = pr(y6, p6)
    return {'table': name, 'panel_median_mae': float(np.mean(np.abs(g - med))),
            'panel_mean_mae': float(np.mean(np.abs(g - mean))),
            'exact': float(np.mean(g == med)), 'within_one': float(np.mean(np.abs(g - med) <= 1)),
            'macro_spearman': float(np.nanmean(sp)),
            'n_pred_5': n5, 'precision_5': prec5, 'recall_5': rec5,
            'n_pred_6': n6, 'precision_6': prec6, 'recall_6': rec6,
            'f1_5': float(2 * prec5 * rec5 / max(prec5 + rec5, 1e-12)),
            'dist': {int(k): int(v) for k, v in zip(*np.unique(g.astype(int), return_counts=True))}}

evals = [eval_table(base, 'old_grade', 'OLD: ET integer conditional median'),
         eval_table(base, 'new_grade_match', 'NEW: p>=5/p6 threshold, marginal-match'),
         eval_table(base, 'new_grade_f1', 'NEW: p>=5/p6 threshold, F1-optimal')]
rep['B_tables'] = evals
# per-target coverage of 5/6 for the two new tables
cov = {}
for col in ['old_grade', 'new_grade_match', 'new_grade_f1']:
    c5 = base.groupby('target')[col].apply(lambda s: int((s >= 5).sum()))
    c6 = base.groupby('target')[col].apply(lambda s: int((s >= 6).sum()))
    cov[col] = {'targets_with_any_5': int((c5 > 0).sum()), 'targets_with_any_6': int((c6 > 0).sum()),
                'max_5_per_target': int(c5.max()), 'total_5': int(c5.sum()),
                'max_6_per_target': int(c6.max()), 'total_6': int(c6.sum())}
rep['B_coverage'] = cov
rep['B_per_target_5_counts'] = {'new_f1': {k: int(v) for k, v in base.groupby('target')['new_grade_f1'].apply(lambda s: (s >= 5).sum()).items()}}
rep['B_per_target_6_counts'] = {'new_f1': {k: int(v) for k, v in base.groupby('target')['new_grade_f1'].apply(lambda s: (s >= 6).sum()).items()}}

# individual-level impact of decision layer (weighted)
for col in ['old_grade', 'new_grade_match', 'new_grade_f1']:
    d = raw_long.merge(base[['beer', 'target', col]], on=['beer', 'target'])
    w = d['weight_within_beer'].values
    err = (d[col] - d['raw']).abs().values
    rep.setdefault('B_individual', []).append({'table': col, 'weighted_mae': float(np.sum(w * err) / np.sum(w)),
                                               'exact': float(np.sum(w * (err == 0)) / np.sum(w)),
                                               'within_one': float(np.sum(w * (err <= 1)) / np.sum(w))})

# candidate new delivery table 250x50
newtab = base[['beer', 'target', 'old_grade', 'new_grade_match', 'new_grade_f1',
               'probability_rating_ge1', 'probability_rating_ge5', 'probability_rating_6']].copy()
newtab.to_csv(os.path.join(OUT, 'q1_decision_layer_table_250x50.csv'), index=False)

# chosen table as 52-slot style wide format for convenience
wide = newtab.pivot(index='beer', columns='target', values='new_grade_f1').reset_index()
wide.to_csv(os.path.join(OUT, 'q1_decision_layer_grades_wide_250x50.csv'), index=False)

with open(os.path.join(OUT, 'decision_layer_report.json'), 'w', encoding='utf-8') as fh:
    json.dump(rep, fh, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
print(json.dumps(rep, indent=2, ensure_ascii=False, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o))
