"""Comparison, tail precision, missingness and plots from frozen Q1 OOF predictions."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json
from beer_models.baseline import make_splits, mask_observed
from beer_models.raw_panel import rating_loss, median_grades
from beer_models.tail_risk import tail_metrics
from r2_raw_q1 import load_inputs, RAW_MANIFEST, summary_metrics
from q1_target_experiment import BASE, PROTOCOL


def grade_matrix(table, beers, targets):
    return table.pivot(index='beer', columns='target', values='predicted_grade').loc[beers, targets].to_numpy(int)


def tail_details(q, pred):
    rows = []
    empirical = median_grades(q)
    for k in [5, 6, 'ge5']:
        truth = np.clip(q[:, :, k] if k != 'ge5' else q[:, :, 5:].sum(2), 0, 1)
        yes = pred == k if k != 'ge5' else pred >= 5
        panel_yes = empirical == k if k != 'ge5' else empirical >= 5
        tp = float((truth*yes).sum())
        fp = float(((1-truth)*yes).sum())
        ptp = int((panel_yes&yes).sum())
        pfp = int((~panel_yes&yes).sum())
        rows.append(dict(event=str(k), n_predicted=int(yes.sum()), raw_truth_mass=float(truth.sum()),
            expected_raw_true_positives=tp, expected_raw_false_positives=fp,
            raw_precision=tp/yes.sum() if yes.any() else np.nan,
            raw_recall=tp/truth.sum() if truth.sum() else np.nan,
            raw_fpr=fp/(1-truth).sum(), panel_truth_count=int(panel_yes.sum()),
            panel_true_positives=ptp, panel_false_positives=pfp,
            panel_precision=ptp/yes.sum() if yes.any() else np.nan,
            panel_recall=ptp/panel_yes.sum() if panel_yes.any() else np.nan,
            panel_fpr=pfp/(~panel_yes).sum()))
    return rows


def summarize(args):
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    bp = json.loads(BASE.read_text(encoding='utf-8'))
    d = load_inputs(args.source, args.raw, bp)
    o, q = args.output, d['distribution']
    summary = json.loads((o/'summary.json').read_text(encoding='utf-8'))
    g = pd.read_csv(o/'oof_grades.csv.gz')
    summary_rows, tails, target_tails = [], [], []
    masks = dict(all50=np.ones(len(d['targets']), bool), primary45=d['primary'])
    for (repeat, method), part in g.groupby(['repeat', 'method'], sort=False):
        pred = grade_matrix(part, d['ids'].beer, d['targets'])
        for scope, mask in masks.items():
            summary_rows.append(dict(repeat=repeat, method=method, scope=scope,
                **(summary_metrics(q[:, mask], pred[:, mask]) | tail_metrics(q[:, mask], pred[:, mask]))))
            tails.extend(dict(repeat=repeat, method=method, scope=scope, **r) for r in tail_details(q[:, mask], pred[:, mask]))
        for j, target in enumerate(d['targets']):
            target_tails.extend(dict(repeat=repeat, method=method, target=target, **r)
                for r in tail_details(q[:, j:j+1], pred[:, j:j+1]))
    save_csv(o, 'scope_comparison.csv', summary_rows)
    save_csv(o, 'tail_precision.csv', tails)
    save_csv(o, 'tail_by_target.csv', target_tails)
    save_csv(o, 'observed_panel_oracle.csv', [dict(**(summary_metrics(q, median_grades(q)) |
        tail_metrics(q, median_grades(q))), role='descriptive_observed_panel_not_a_forecast')])
    styles = d['ids'].tasting_category_fine.to_numpy()
    missing_rows = []
    for repeat, seed in enumerate(p['repeat_outer_seeds']):
        for fold, (train, test) in enumerate(make_splits(styles, p['outer_folds'], seed)):
            point = joblib.load(o/'models'/f'r{repeat}_f{fold}.joblib')
            for missing_repeat in range(3):
                for rate in [.05, .1, .2]:
                    z, added = mask_observed(d['x'][test], rate, p['seed']+repeat*10000+fold*100+missing_repeat)
                    predictions = dict(shared_et=point.models['shared_et'].predict(z)['grade'], inner_selected=point.predict(z)['grade'])
                    for name, pred in predictions.items():
                        before = grade_matrix(g.query('repeat == @repeat and method == @name'), d['ids'].beer, d['targets'])[test]
                        missing_rows.append(dict(repeat=repeat, outer_fold=fold, missing_repeat=missing_repeat,
                            additional_missing_rate=rate, method=name, n_beers=len(test), added_missing_cells=int(added.sum()),
                            raw_rating_mae=float(rating_loss(q[test], pred).mean()),
                            changed_grade_fraction=float((pred != before).mean())))
            print(f'Missingness replay r{repeat}_f{fold}', flush=True)
    save_csv(o, 'missingness_by_fold.csv', missing_rows)
    frame = pd.DataFrame(missing_rows)
    save_csv(o, 'missingness_summary.csv', [dict(method=name, additional_missing_rate=rate,
        raw_rating_mae=float(np.average(part.raw_rating_mae, weights=part.n_beers)),
        changed_grade_fraction=float(np.average(part.changed_grade_fraction, weights=part.n_beers)),
        scope='Artificial MCAR masks over observed held-out cells; not validation of informative missingness')
        for (name, rate), part in frame.groupby(['method', 'additional_missing_rate'])])
    plot(o)
    write_json(o/'supplement_verification.json', dict(status='completed', tail_metric_rows=len(tails),
        target_tail_rows=len(target_tails), missingness_replays=len(missing_rows),
        original_missing_cells_preserved=True, no_refitting_on_heldout_data=True,
        tail_precision_zero_predictions='undefined blank, not zero', deployment=summary['deployed_name']))


def plot(o):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = pd.read_csv(o/'metrics_by_repeat.csv')
    by_grade = pd.read_csv(o/'metrics_by_observed_grade.csv').query("scope == 'all50'")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    names = ['training_prior', 'shared_et', 'target_leaf3', 'target_leaf1', 'inner_selected']
    labels = ['Prior', 'Shared ET', 'Target leaf 3', 'Target leaf 1', 'Inner selected']
    colors = ['#aeb6bf', '#687e91', '#a7c4b4', '#b5a4c8', '#187963']
    for i, (name, color) in enumerate(zip(names, colors)):
        values = metrics.loc[metrics.method.eq(name), 'raw_rating_mae']
        axes[0].bar(i, values.mean(), color=color)
        axes[0].scatter(np.full(len(values), i), values, color='black', s=12)
    axes[0].set(xticks=range(len(names)), xticklabels=labels, ylabel='Individual-rating MAE', ylim=(.6, .86),
                title='Three repeated nested comparisons')
    axes[0].tick_params(axis='x', labelrotation=30)
    for name, label, color in [('shared_et', 'Shared ET', colors[1]), ('inner_selected', 'Inner selected', colors[-1])]:
        part = by_grade.loc[by_grade.method.eq(name)].groupby('observed_grade')[['grade_mae', 'exact_accuracy']].mean()
        axes[1].plot(part.index, part.grade_mae, marker='o', label=label, color=color)
        axes[2].plot(part.index, part.exact_accuracy*100, marker='o', label=label, color=color)
    axes[1].set(xlabel='Observed individual grade', ylabel='Conditional MAE', xticks=range(7), title='Error by grade')
    axes[2].set(xlabel='Observed individual grade', ylabel='Exact recall (%)', xticks=range(7), title='Tail failure remains visible')
    axes[1].legend(); axes[2].legend()
    fig.savefig(o/'q1_comparison.png', dpi=170)
    fig.savefig(o/'q1_comparison.svg')
    plt.close(fig)
    svg = o/'q1_comparison.svg'
    svg.write_text('\n'.join(v.rstrip() for v in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--output', type=Path, required=True)
    with threadpool_limits(limits=1):
        summarize(parser.parse_args())
