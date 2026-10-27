# -*- coding: utf-8 -*-
"""Build the paper hand-off package (deliverables/paper_handoff_20261028).

All numbers are recomputed or copied from experiments/lead_review/{q1,q2,q3};
nothing here retrains a model. The Q1 submission table is regenerated with the
non-forced top-k decision rule (the variant recommended in q1/FINDINGS.md) and
its metrics are asserted against decision_topk_report.json.
"""
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
LR = ROOT / 'experiments' / 'lead_review'
OUT = ROOT / 'deliverables' / 'paper_handoff_20261028'
TAB = OUT / 'tables'
FIG = OUT / 'figures'
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
checks = {}


def save(df, name):
    df.to_csv(TAB / name, index=False, encoding='utf-8')
    return name


# ---------------------------------------------------------------- Q1
R2 = ROOT / 'outputs' / 'r2_raw_q1_20261026'
T6 = ROOT / 'outputs' / 'task6_final_20261026' / 'paper' / 'tables'
wt = pd.read_csv(R2 / 'rating_weights_and_folds.csv')
obs = pd.read_csv(R2 / 'observed_panel_distribution.csv')
t6 = pd.read_csv(T6 / 'q1__oof_scores_all52.csv')
oof = pd.read_csv(R2 / 'oof_predictions.csv.gz')
fold = wt.groupby('beer')['outer_fold'].first()
et = oof[oof.method == 'extra_trees'][['beer', 'target', 'predicted_grade']]
t = obs.merge(t6[['beer', 'target', 'probability_rating_ge1', 'probability_rating_ge5', 'probability_rating_6']],
              on=['beer', 'target']).merge(et, on=['beer', 'target'])
t['outer_fold'] = t['beer'].map(fold)
assert len(t) == 12500, len(t)


def topk_rule(force_min=False):
    g = t.copy()
    g['new_grade'] = np.minimum(g['predicted_grade'], 4).astype(int)
    for (tg, f), te_idx in g.groupby(['target', 'outer_fold']).groups.items():
        tr = g[(g.target == tg) & (g.outer_fold != f)]
        te = g.loc[te_idx]
        n = len(te)
        r5 = float((tr['panel_weighted_median'] >= 5).mean())
        r6 = float((tr['panel_weighted_median'] >= 6).mean())
        k5, k6 = int(round(r5 * n)), int(round(r6 * n))
        if force_min:
            k5, k6 = max(k5, 1), max(k6, 1)
        top6 = te['probability_rating_6'].sort_values(ascending=False).index[:k6]
        g.loc[top6, 'new_grade'] = 6
        rest = te.drop(index=top6)
        top5 = rest['probability_rating_ge5'].sort_values(ascending=False).index[:k5]
        g.loc[top5, 'new_grade'] = np.maximum(g.loc[top5, 'new_grade'].values, 5)
    return g


def evaluate(g, col, name):
    x = g[col].values.astype(float)
    med, mean = g['panel_weighted_median'].values, g['panel_weighted_mean'].values
    y5, y6, p5, p6 = med >= 5, med >= 6, x >= 5, x >= 6
    def pr(y, p):
        tp = int((y & p).sum())
        return int(p.sum()), float(tp / max(p.sum(), 1)), float(tp / max(y.sum(), 1))
    n5, prec5, rec5 = pr(y5, p5)
    n6, prec6, rec6 = pr(y6, p6)
    sp = [stats.spearmanr(gg[col], gg['panel_weighted_mean']).correlation
          for _, gg in g.groupby('target') if gg['panel_weighted_mean'].nunique() > 2]
    return dict(table=name, panel_median_mae=float(np.mean(np.abs(x - med))),
                panel_mean_mae=float(np.mean(np.abs(x - mean))),
                exact=float(np.mean(x == med)), within_one=float(np.mean(np.abs(x - med) <= 1)),
                macro_spearman=float(np.nanmean(sp)),
                n_pred_5=n5, precision_5=prec5, recall_5=rec5,
                n_pred_6=n6, precision_6=prec6, recall_6=rec6,
                dist={int(k): int(v) for k, v in zip(*np.unique(x.astype(int), return_counts=True))})


g = topk_rule(force_min=False)
new = evaluate(g, 'new_grade', 'q1_topk_nonforced')
old = evaluate(g, 'predicted_grade', 'q1_et_integer_median')
rep = json.loads((LR / 'q1' / 'decision_topk_report.json').read_text(encoding='utf-8'))
ref = [e for e in rep['tables'] if e['table'].startswith('NEW: top-k cumulative-prob (fold-wise rate)')][0]
for k in ('panel_median_mae', 'panel_mean_mae', 'exact', 'within_one', 'macro_spearman',
          'n_pred_5', 'precision_5', 'recall_5', 'n_pred_6', 'precision_6', 'recall_6'):
    assert abs(new[k] - ref[k]) < 1e-9, (k, new[k], ref[k])
checks['q1_topk_matches_report'] = True

# submission-grade long table
g['target_family'] = g['target'].map(lambda s: 'Aroma' if s.startswith('A_')
                                     else ('Flavour' if s.startswith('F_') else 'Supplementary'))
sub = g[['beer', 'target', 'target_family', 'predicted_grade', 'new_grade', 'panel_weighted_mean',
         'panel_weighted_median', 'probability_rating_ge1', 'probability_rating_ge5',
         'probability_rating_6', 'outer_fold']].copy()
sub = sub.rename(columns={'predicted_grade': 'grade_integer_median_old', 'new_grade': 'grade_0_6',
                          'panel_weighted_mean': 'panel_mean_reference',
                          'panel_weighted_median': 'panel_median_reference'})
save(sub, 'q1__evaluation_long_12500.csv')
save(sub[['beer', 'target', 'target_family', 'grade_0_6',
          'probability_rating_ge5', 'probability_rating_6']],
     'q1__submission_long_12500.csv')
save(sub.pivot(index='beer', columns='target', values='grade_0_6').reset_index(),
     'q1__submission_wide_250x50.csv')

cal = pd.read_csv(LR / 'q1' / 'candidate_panel_mean_and_grade_250x50.csv')
save(cal.rename(columns={'observed_panel_mean_for_ref': 'panel_mean_reference',
                         'observed_panel_median_for_ref': 'panel_median_reference',
                         'pred_grade_current': 'grade_integer_median_old',
                         'pred_panel_mean_calibrated': 'panel_mean_calibrated',
                         'pred_grade_calibrated_round': 'grade_from_calibrated_mean_round'}),
     'q1__calibrated_panel_mean_12500.csv')

# metrics summary
imp = pd.read_csv(LR / 'q1' / 'improvement_experiments.csv')
rows = []
for e in (old, new):
    rows.append(dict(scope='beer_level', method=e['table'], panel_median_mae=e['panel_median_mae'],
                     panel_mean_mae=e['panel_mean_mae'], exact_match=e['exact'],
                     within_one=e['within_one'], macro_spearman=e['macro_spearman'],
                     n_pred_ge5=e['n_pred_5'], precision_ge5=e['precision_5'],
                     recall_ge5=e['recall_5'], n_pred_6=e['n_pred_6'],
                     precision_6=e['precision_6'], recall_6=e['recall_6']))
rows.append(dict(scope='individual_taster', method='extra_trees_integer_median',
                 panel_median_mae=0.795214, panel_mean_mae=np.nan, exact_match=0.604919,
                 within_one=0.783035, macro_spearman=np.nan))
rows.append(dict(scope='lower_bound', method='panel_median_oracle_individual_level',
                 panel_median_mae=0.663567, exact_match=0.658689, within_one=0.818681))
for _, r in imp.iterrows():
    rows.append(dict(scope='beer_level', method='calibration:' + r['method'],
                     panel_median_mae=r['panel_median_mae'], panel_mean_mae=r['panel_mean_mae'],
                     exact_match=r['panel_median_exact_rounded'],
                     within_one=r['panel_median_within_one_rounded'],
                     n_pred_ge5=r['n_pred_ge5'], precision_ge5=r['precision_ge5'],
                     recall_ge5=r['recall_ge5']))
m = pd.DataFrame(rows)
save(m, 'q1__metrics_summary.csv')

# per-target high-grade coverage (observed prevalence)
prev = sub.groupby('target').agg(n_observed_ge5=('panel_median_reference', lambda s: int((s >= 5).sum())),
                                 n_observed_6=('panel_median_reference', lambda s: int((s >= 6).sum())),
                                 n_predicted_ge5=('grade_0_6', lambda s: int((s >= 5).sum())),
                                 n_predicted_6=('grade_0_6', lambda s: int((s >= 6).sum()))).reset_index()
save(prev, 'q1__high_grade_coverage_by_target.csv')

# ---------------------------------------------------------------- Q2
q2 = LR / 'q2'
top3 = pd.read_csv(q2 / 'q2_top3_full50_graphridge.csv')
keep = ['target', 'target_family', 'rank', 'field', 'chemical_name', 'origin_paper',
        'ridge_coef', 'graph_importance', 'importance_rank', 'bootstrap_freq',
        'r6_cross_corr', 'r6_sign_agree', 'mechanism_class', 'mechanism_note',
        'target_semantic_match', 'chemical_evidence', 'confidence', 'off_flavors',
        'threshold_mean', 'degree_graph', 'community_id']
save(top3[keep], 'q2__top3_all50_150.csv')
save(top3.pivot(index='target', columns='rank', values='chemical_name').reset_index(),
     'q2__top3_wide_50.csv')
save(pd.read_csv(q2 / 'q2_top3_wide.csv'), 'q2__top3_wide_50_fields.csv')
save(pd.read_csv(q2 / 'q2_stability_bootstrap.csv'), 'q2__stability_bootstrap.csv')
save(pd.read_csv(q2 / 'q2_method_jaccard.csv'), 'q2__method_jaccard.csv')
save(pd.read_csv(q2 / 'q2_network_edges_0_5.csv'), 'q2__network_edges_0_5.csv')
save(pd.read_csv(q2 / 'q2_network_communities.csv'), 'q2__network_communities.csv')
save(top3.groupby(['chemical_evidence', 'confidence']).size().reset_index(name='n_compound_slots'),
     'q2__evidence_tier_counts.csv')
save(q2 / 'q2_summary.json', 'q2__summary.json') if False else None
shutil.copyfile(q2 / 'q2_summary.json', TAB / 'q2__summary.json')
for f in ['fig_q2_bootstrap_stability.png', 'fig_q2_method_jaccard.png', 'fig_q2_evidence_confidence.png']:
    shutil.copyfile(q2 / f, FIG / f)
checks['q2_150rows_50targets'] = (len(top3) == 150, top3['target'].nunique() == 50,
                                  int(top3['chemical_name'].isna().sum()), 0)

# ---------------------------------------------------------------- Q3
q3 = LR / 'q3'
mig = pd.read_csv(q3 / 's2c_migration_results.csv')
save(mig, 'q3__migration_33cases.csv')
cnt = pd.read_csv(q3 / 's2c_candidate_counts.csv')
save(cnt, 'q3__migration_candidate_counts.csv')
save(pd.read_csv(q3 / 's2_source_defects.csv'), 'q3__source_defects_11beers.csv')
save(pd.read_csv(q3 / 's2_style_defect_exceedance.csv'), 'q3__style_defect_exceedance.csv')
save(pd.read_csv(q3 / 's3_generation_quality.csv'), 'q3__generation_quality.csv')
checks['q3_33_of_33_pass'] = {c: int((cnt[c] == 20).sum()) for c in
                              ['n_pass_A_mean', 'n_pass_B_mean', 'n_pass_C_mean']}
checks['q3_migration_rows'] = len(mig)
checks['q3_dist_reduction_range'] = [float(mig.dist_reduction_pct.min()), float(mig.dist_reduction_pct.max())]

# ---------------------------------------------------------------- manifest
checks['q1_submission_dist'] = new['dist']
checks['q1_rows'] = len(sub)
json.dump(checks, open(OUT / 'verification.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print(json.dumps(checks, indent=2, ensure_ascii=False))
print('files:', len(list(TAB.iterdir())), 'tables +', len(list(FIG.iterdir())), 'figures')
