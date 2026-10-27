# -*- coding: utf-8 -*-
"""Top-k cumulative-probability decision rule (per target/fold) vs integer median.
Guarantees high-grade coverage only where the panel actually contains 5/6."""
import os, json
import numpy as np, pandas as pd
from scipy import stats

ROOT = r'C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件'
OUT = os.path.join(ROOT, 'experiments', 'lead_review', 'q1')
R2 = os.path.join(ROOT, 'outputs', 'r2_raw_q1_20261026')
T6 = os.path.join(ROOT, 'outputs', 'task6_final_20261026', 'paper', 'tables')
wt = pd.read_csv(os.path.join(R2, 'rating_weights_and_folds.csv'))
obs = pd.read_csv(os.path.join(R2, 'observed_panel_distribution.csv'))
t6 = pd.read_csv(os.path.join(T6, 'q1__oof_scores_all52.csv'))
oof = pd.read_csv(os.path.join(R2, 'oof_predictions.csv.gz'))
fold = wt.groupby('beer')['outer_fold'].first()
et = oof[oof.method == 'extra_trees'][['beer', 'target', 'predicted_grade']].copy()

t = obs.merge(t6[['beer', 'target', 'probability_rating_ge1', 'probability_rating_ge5', 'probability_rating_6']], on=['beer', 'target'])
t = t.merge(et, on=['beer', 'target'])
t['outer_fold'] = t['beer'].map(fold)
assert len(t) == 12500

rep = {}
# observed per-target high-grade prevalence / max (answering "can every dimension reach 5/6?")
prev = t.groupby('target').agg(n5=('panel_weighted_median', lambda s: int((s >= 5).sum())),
                               n6=('panel_weighted_median', lambda s: int((s >= 6).sum())),
                               max_med=('panel_weighted_median', 'max'))
rep['observed_prevalence'] = {'targets_with_5': int((prev.n5 > 0).sum()), 'targets_with_6': int((prev.n6 > 0).sum()),
                              'targets_without_5': prev.index[prev.n5 == 0].tolist(),
                              'targets_without_6': prev.index[prev.n6 == 0].tolist(),
                              'total_cells_5': int(prev.n5.sum()), 'total_cells_6': int(prev.n6.sum())}

def topk_rule(force_min=False):
    g = t.copy()
    g['new_grade'] = np.minimum(g['predicted_grade'], 4).astype(int)
    for (tg, f), te_idx in g.groupby(['target', 'outer_fold']).groups.items():
        tr = g[(g.target == tg) & (g.outer_fold != f)]
        te = g.loc[te_idx]
        n = len(te)
        r5 = float((tr['panel_weighted_median'] >= 5).mean())
        r6 = float((tr['panel_weighted_median'] >= 6).mean())
        k5 = int(round(r5 * n)); k6 = int(round(r6 * n))
        if force_min:
            k5 = max(k5, 1); k6 = max(k6, 1)
        order6 = te['probability_rating_6'].sort_values(ascending=False).index[:k6]
        g.loc[order6, 'new_grade'] = 6
        rest = te.drop(index=order6)
        order5 = rest['probability_rating_ge5'].sort_values(ascending=False).index[:k5]
        g.loc[order5, 'new_grade'] = np.maximum(g.loc[order5, 'new_grade'].values, 5)
    return g

def evaluate(df, col, name):
    g = df[col].values.astype(float); med = df['panel_weighted_median'].values; mean = df['panel_weighted_mean'].values
    y5 = med >= 5; y6 = med >= 6; p5 = g >= 5; p6 = g >= 6
    def pr(y, p):
        tp = int((y & p).sum()); return int(p.sum()), float(tp / max(p.sum(), 1)), float(tp / max(y.sum(), 1))
    n5, prec5, rec5 = pr(y5, p5); n6, prec6, rec6 = pr(y6, p6)
    sp = [stats.spearmanr(gg[col], gg['panel_weighted_mean']).correlation
          for tg, gg in df.groupby('target') if gg['panel_weighted_mean'].nunique() > 2]
    return {'table': name, 'panel_median_mae': float(np.mean(np.abs(g - med))),
            'panel_mean_mae': float(np.mean(np.abs(g - mean))), 'exact': float(np.mean(g == med)),
            'within_one': float(np.mean(np.abs(g - med) <= 1)), 'macro_spearman': float(np.nanmean(sp)),
            'n_pred_5': n5, 'precision_5': prec5, 'recall_5': rec5, 'f1_5': float(2*prec5*rec5/max(prec5+rec5,1e-12)),
            'n_pred_6': n6, 'precision_6': prec6, 'recall_6': rec6,
            'dist': {int(k): int(v) for k, v in zip(*np.unique(g.astype(int), return_counts=True))}}

evals = [evaluate(t, 'predicted_grade', 'OLD: ET integer conditional median')]
for force in [False, True]:
    g = topk_rule(force_min=force)
    nm = 'NEW: top-k cumulative-prob (fold-wise rate)' + (' + force>=1' if force else '')
    evals.append(evaluate(g, 'new_grade', nm))
    if force:
        rep['chosen_table'] = nm
        rep['chosen_dist'] = evals[-1]['dist']
        g[['beer', 'target', 'predicted_grade', 'probability_rating_ge1', 'probability_rating_ge5',
           'probability_rating_6', 'new_grade']].rename(columns={'predicted_grade': 'old_grade'}).to_csv(
            os.path.join(OUT, 'q1_decision_topk_table_250x50.csv'), index=False)
        wide = g.pivot(index='beer', columns='target', values='new_grade').reset_index()
        wide.to_csv(os.path.join(OUT, 'q1_decision_topk_grades_wide_250x50.csv'), index=False)
        cov5 = g.groupby('target')['new_grade'].apply(lambda s: int((s >= 5).sum()))
        cov6 = g.groupby('target')['new_grade'].apply(lambda s: int((s >= 6).sum()))
        rep['chosen_coverage'] = {'targets_with_any_5': int((cov5 > 0).sum()), 'targets_with_any_6': int((cov6 > 0).sum()),
                                  'max_5_per_target': int(cov5.max()), 'max_6_per_target': int(cov6.max())}
rep['tables'] = evals
with open(os.path.join(OUT, 'decision_topk_report.json'), 'w', encoding='utf-8') as fh:
    json.dump(rep, fh, indent=2, ensure_ascii=False)
print(json.dumps(rep, indent=2, ensure_ascii=False))
