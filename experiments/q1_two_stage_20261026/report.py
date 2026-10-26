"""Compact comparison and paired beer-level tail uncertainty for both experiments."""
import argparse
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from beer_audit.audit import save_csv
from beer_models.explain import stratified_draw
from beer_models.raw_panel import median_grades
from r2_raw_q1 import load_inputs, RAW_MANIFEST
from q1_target_experiment import BASE
from summarize_q1_target import tail_details
from run import scoped_metrics


def matrix(table, ids, targets, method):
    frame = table.loc[table.method.eq(method)]
    return np.stack([frame.loc[frame.repeat.eq(r)].pivot(index='beer', columns='target', values='predicted_grade').loc[ids.beer, targets].to_numpy(int) for r in range(3)])


def run(args):
    d = load_inputs(args.source, args.raw, json.loads(BASE.read_text(encoding='utf-8')))
    q, ids, targets = d['distribution'], d['ids'], d['targets']
    first = pd.read_csv(HERE/'results/oof_grades.csv.gz')
    second = pd.read_csv(HERE/'forest_results/oof_grades.csv.gz')
    previous = pd.read_csv(ROOT/'outputs/q1_selective_tail_20261026/oof_grades.csv.gz')
    filtered = pd.read_csv(HERE/'confidence_results/oof_grades.csv.gz')
    all_grades = dict(original_median=matrix(second, ids, targets, 'original_median'),
        previous_selective=matrix(previous, ids, targets, 'inner_selected'),
        old_same_selector=matrix(second, ids, targets, 'old_same_selector'),
        two_stage=matrix(first, ids, targets, 'new_selected'),
        joint_forest=matrix(second, ids, targets, 'new_selected'),
        confidence_filter=matrix(filtered, ids, targets, 'new_selected'))
    rows, tail_rows = [], []
    for name, grades in all_grades.items():
        for r in range(3):
            rows.extend(dict(method=name, repeat=r, **a) for a in scoped_metrics(q, grades[r], d['primary']))
            for scope, mask in [('all50', np.ones(len(targets), bool)), ('primary45', d['primary'])]:
                tail_rows.extend(dict(method=name, repeat=r, scope=scope, **a) for a in tail_details(q[:, mask], grades[r][:, mask]))
    repeats, tails = pd.DataFrame(rows), pd.DataFrame(tail_rows)
    summary = []
    for (scope, method), frame in repeats.groupby(['scope', 'method'], sort=False):
        t = tails.loc[tails.scope.eq(scope)&tails.method.eq(method)]
        row = dict(scope=scope, method=method, **{c: frame[c].mean() for c in ['raw_rating_mae', 'panel_median_mae',
            'raw_recall_5', 'raw_recall_6', 'panel_recall_5', 'panel_recall_6', 'raw_high_fpr', 'panel_high_fpr']})
        for event in ['5', '6', 'ge5']:
            a = t.loc[t.event.eq(event)]
            n = a.n_predicted.sum()
            row['n_predicted_'+event] = int(n)
            row['precision_'+event] = a.expected_raw_true_positives.sum()/n if n else np.nan
            row['panel_precision_'+event] = a.panel_true_positives.sum()/n if n else np.nan
        summary.append(row)
    save_csv(HERE, 'comparison.csv', summary)
    save_csv(HERE, 'comparison_by_repeat.csv', repeats)
    styles = ids.tasting_category_fine.to_numpy()
    intervals = []
    for scope, mask in [('all50', np.ones(len(targets), bool)), ('primary45', d['primary'])]:
        truth = q[:, mask]
        for kind, grade in [('individual5', 5), ('individual6', 6), ('panel6', 6)]:
            mass = truth[:, :, grade] if kind.startswith('individual') else (median_grades(truth) == grade).astype(float)
            denominator = mass.sum(1)
            new_hits = (mass[None]*(all_grades['joint_forest'][:, :, mask] == grade)).mean(0).sum(1)
            old_hits = (mass[None]*(all_grades['old_same_selector'][:, :, mask] == grade)).mean(0).sum(1)
            delta = new_hits-old_hits
            rng = np.random.default_rng(20261026)
            draws = []
            for _ in range(2000):
                ix = stratified_draw(styles, rng)
                if denominator[ix].sum() > 0:
                    draws.append(delta[ix].sum()/denominator[ix].sum())
            intervals.append(dict(scope=scope, metric=kind+'_recall_gain', reference='old_same_selector',
                gain=delta.sum()/denominator.sum(), lower95=np.quantile(draws, .025), upper95=np.quantile(draws, .975),
                valid_bootstrap_draws=len(draws), independent_beer_units=len(q), repeated_predictions=750,
                uncertainty='Fixed-OOF paired beer resampling; not full refit or external test'))
    save_csv(HERE, 'tail_gain_intervals.csv', intervals)
    by_target = []
    for j, target in enumerate(targets):
        observed_six = q[:, j, 6]
        panel_six = median_grades(q)[:, j] == 6
        for method in ['old_same_selector', 'joint_forest']:
            predicted = all_grades[method][:, :, j] == 6
            hits = float((predicted*observed_six).sum())
            count = int(predicted.sum())
            by_target.append(dict(method=method, target=target, primary=bool(d['primary'][j]),
                predicted_six=count, raw_true_six_mass=float(observed_six.sum()),
                raw_correct_mass=hits, raw_recall_six=hits/(3*observed_six.sum()) if observed_six.sum() else np.nan,
                raw_precision_six=hits/count if count else np.nan,
                panel_six_count=int(panel_six.sum()), panel_correct_six=int((predicted*panel_six).sum())))
    save_csv(HERE, 'six_by_target.csv', by_target)
    plot(pd.DataFrame(summary), repeats)
    print(pd.DataFrame(summary).query("scope == 'primary45'").to_string(index=False))
    print(pd.DataFrame(intervals).query("scope == 'primary45'").to_string(index=False))


def plot(summary, repeats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['original_median', 'old_same_selector', 'two_stage', 'joint_forest', 'confidence_filter']
    labels = ['Official median', 'Old + same rule', 'Two-stage boost', 'Joint tail forest', 'Precision filter']
    colors = ['#6d8795', '#aaa0b4', '#c18e70', '#317d75', '#a9b9b5']
    s = summary.query("scope == 'primary45'").set_index('method').loc[names]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    x = np.arange(len(names))
    axes[0, 0].bar(x, s.raw_rating_mae, color=colors)
    for i, name in enumerate(names):
        values = repeats.query("scope == 'primary45' and method == @name").raw_rating_mae
        axes[0, 0].scatter([i]*len(values), values, s=17, color='black')
    axes[0, 0].set(ylabel='Individual-rating MAE', ylim=(.77, .80), title='Error: forest still worse than official median')
    axes[0, 1].bar(x-.18, s.raw_recall_5*100, .36, color='#7896aa', label='Exact 5')
    axes[0, 1].bar(x+.18, s.raw_recall_6*100, .36, color='#317d75', label='Exact 6')
    axes[0, 1].set(ylabel='Individual exact recall (%)', title='Actual six-score gain with same decision rule')
    axes[0, 1].legend()
    axes[1, 0].bar(x, s.panel_recall_6*100, color=colors)
    axes[1, 0].set(ylabel='Panel median-6 recall (%)', title='Observed group representative: six-score recall')
    axes[1, 1].bar(x[1:]-.18, s.precision_ge5.iloc[1:]*100, .36, color='#7896aa', label='At least 5')
    axes[1, 1].bar(x[1:]+.18, s.precision_6.iloc[1:]*100, .36, color='#317d75', label='Exact 6')
    axes[1, 1].axhline(50, color='#7896aa', linestyle=':', label='High-score precision gate')
    axes[1, 1].axhline(25, color='#317d75', linestyle=':', label='Exact-6 precision gate')
    axes[1, 1].set(ylabel='Correct among issued predictions (%)', ylim=(0, 60), title='Precision remains insufficient')
    axes[1, 1].legend(fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks(x, labels, rotation=15)
    fig.suptitle('Primary 45 labelled targets; independent experiment, no official integration')
    fig.savefig(HERE/'comparison.png', dpi=170); fig.savefig(HERE/'comparison.svg')
    plt.close(fig)
    svg = HERE/'comparison.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    run(parser.parse_args())
