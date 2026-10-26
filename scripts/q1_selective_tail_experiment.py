"""Nested exact-high-grade decisions on already trained grouped probabilities."""
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
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.baseline import make_splits
from beer_models.raw_panel import median_grades, enforce_hierarchy, rating_loss, predict_raw_frame
from beer_models.tail_risk import tail_metrics, feasible, risk_frame
from beer_models.repeated_validation import beer_averaged_interval
from beer_models.selective_tail import selective_decision, candidates, select_policy, SelectiveTailPoint
from r2_raw_q1 import load_inputs, RAW_MANIFEST, content_hash, metric_tables, summary_metrics, write_scorecard
from q1_target_experiment import BASE, from_frame
from summarize_q1_target import tail_details

PROTOCOL = ROOT/'configs/q1_selective_tail_protocol_20261026.json'


def run(args):
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    bp = json.loads(BASE.read_text(encoding='utf-8'))
    d = load_inputs(args.source, args.raw, bp)
    o, parent = args.output, args.parent
    o.mkdir(parents=True, exist_ok=False)
    write_json(o/'protocol.json', p)
    q, ids, targets = d['distribution'], d['ids'], d['targets']
    styles = ids.tasting_category_fine.to_numpy()
    base_table = pd.read_csv(parent/'oof_probabilities.csv.gz', float_precision='round_trip').query("method == 'shared_et'")
    choices, priors, metrics, target_rows, level_rows, confusion, panel_confusion, tail_rows = [], [], [], [], [], [], [], []
    scope_rows = []
    policy_records = []
    gradients = {name: np.empty((len(p['repeat_outer_seeds']), *q.shape[:2]), int)
        for name in ['shared_et', 'inner_selected', 'training_prior']+[f'policy{i}' for i in range(len(candidates(p)))]}

    def choose(context, train, seed):
        table = pd.read_csv(parent/f'inner_{context}.csv.gz', float_precision='round_trip').query("method == 'shared_et'")
        prob = from_frame(table, ids.beer.iloc[train], targets)
        all_predictions = {i: np.empty((len(train), len(targets)), int) for i in range(len(candidates(p)))}
        for fold, (it, iv) in enumerate(make_splits(styles[train], p['inner_folds'], seed)):
            prior = q[train[it]].mean((0, 1))
            priors.append(dict(context=context, inner_fold=fold, prior5=float(prior[5]), prior6=float(prior[6]),
                               n_prior_training_beers=len(it)))
            for i, penalties in enumerate(candidates(p)):
                all_predictions[i][iv] = selective_decision(prob[iv], prior, *penalties, d['parents'])
        baseline = enforce_hierarchy(median_grades(prob), d['parents'])
        selected, rows = select_policy(all_predictions, q[train], baseline, p)
        choices.extend(dict(context=context, **r) for r in rows)
        policy_records.append(dict(context=context, selected_candidate=selected,
            has_eligible_candidate=selected is not None,
            prior5=float(q[train].mean((0, 1))[5]), prior6=float(q[train].mean((0, 1))[6])))
        return selected

    folds = np.empty((len(p['repeat_outer_seeds']), len(q)), int)
    for repeat, seed in enumerate(p['repeat_outer_seeds']):
        probability = from_frame(base_table.loc[base_table.repeat.eq(repeat)], ids.beer, targets)
        baseline = enforce_hierarchy(median_grades(probability), d['parents'])
        gradients['shared_et'][repeat] = baseline
        for fold, (train, test) in enumerate(make_splits(styles, p['outer_folds'], seed)):
            selected = choose(f'r{repeat}_f{fold}', train, seed+100+fold)
            prior = q[train].mean((0, 1))
            for i, penalties in enumerate(candidates(p)):
                gradients[f'policy{i}'][repeat, test] = selective_decision(probability[test], prior, *penalties, d['parents'])
            gradients['inner_selected'][repeat, test] = baseline[test] if selected is None else gradients[f'policy{selected}'][repeat, test]
            gradients['training_prior'][repeat, test] = enforce_hierarchy(median_grades(np.broadcast_to(q[train].mean(0), (len(test), *q.shape[1:]))), d['parents'])
            folds[repeat, test] = fold
        tables = metric_tables(d, {name: dict(grade=g[repeat]) for name, g in gradients.items()})
        scope_rows.append(tables['model_comparison.csv'].assign(repeat=repeat))
        target_rows.append(tables['metrics_by_target.csv'].assign(repeat=repeat))
        level_rows.append(tables['metrics_by_observed_grade.csv'].assign(repeat=repeat))
        confusion.append(tables['confusion.csv'].assign(repeat=repeat))
        observed = median_grades(q)
        for name, g in gradients.items():
            metrics.append(dict(repeat=repeat, method=name, **(summary_metrics(q, g[repeat]) | tail_metrics(q, g[repeat]))))
            for scope, mask in [('all50', np.ones(len(targets), bool)), ('primary45', d['primary'])]:
                tail_rows.extend(dict(repeat=repeat, method=name, scope=scope, **r) for r in tail_details(q[:, mask], g[repeat][:, mask]))
            for k in range(7):
                truth = observed == k
                for predicted in range(7):
                    n = int((truth & (g[repeat] == predicted)).sum())
                    panel_confusion.append(dict(repeat=repeat, method=name, observed_grade=k, predicted_grade=predicted,
                        count=n, row_fraction=n/truth.sum() if truth.sum() else np.nan))
    metrics = pd.DataFrame(metrics)
    delta = np.stack([rating_loss(q, gradients['inner_selected'][r]).mean(1)-rating_loss(q, gradients['shared_et'][r]).mean(1)
                     for r in range(len(folds))])
    increase, low, high = beer_averaged_interval(delta, styles, p['bootstrap_repeats'], p['bootstrap_seed'], [.025, .975])
    selected_metrics = metrics.query("method == 'inner_selected'").sort_values('repeat')
    base_metrics = metrics.query("method == 'shared_et'").sort_values('repeat')
    eligible_repeats = [feasible(a, b, p['point_limits']) for a, b in zip(selected_metrics.to_dict('records'), base_metrics.raw_rating_mae)]
    pooled = tail_metrics(np.concatenate([q]*len(folds)), gradients['inner_selected'].reshape(-1, len(targets)))
    pooled_feasible = feasible(pooled, float(base_metrics.raw_rating_mae.mean()), p['point_limits'])
    adopted = bool(all(eligible_repeats) and pooled_feasible and high <= .02)
    full_choice = choose('full_training', np.arange(len(q)), p['seed']+103)
    adopted = bool(adopted and full_choice is not None)
    bundle = joblib.load(parent/'model.joblib')
    base = bundle['raw_panel_model'].models['shared_et']
    penalties = candidates(p)[full_choice] if adopted and full_choice is not None else None
    bundle['raw_panel_model'] = SelectiveTailPoint(base, q.mean((0, 1)), penalties, d['parents'])
    bundle['experimental_point_model'] = SelectiveTailPoint(base, q.mean((0, 1)),
        candidates(p)[full_choice] if full_choice is not None else None, d['parents'])
    bundle.update(q1_selective_protocol=p, selected_model='selective_exact_tail' if adopted else 'extra_trees__median',
        grade_method='q1_selective_tail_decision' if adopted else 'r2_raw_panel_median',
        point_semantics='selective_exact_tail_utility_not_conditional_median' if adopted else 'integer_balanced_panel_conditional_median_with_parent_cap',
        score_status='selective_tail_grade_prediction' if adopted else 'raw_panel_balanced_median_prediction',
        tail_point_qualified_on_historical_folds=adopted,
        point_deployment_status='selective_tail_qualified' if adopted else 'provisional_baseline_tail_not_qualified')
    joblib.dump(bundle, o/'model.joblib', compress=3)
    frame = pd.read_csv(args.source/'1.csv', dtype=str)
    long, _, wide, _ = predict_raw_frame(frame, bundle)
    long = long.merge(risk_frame(frame, bundle, long.beer.drop_duplicates().to_numpy()), on=['beer', 'target'], how='left', validate='one_to_one')
    save_csv(o, 'full_fit_scores_all52.csv', long)
    save_csv(o, 'full_fit_grades_all52.csv', wide)
    for name, table in [('metrics_by_repeat.csv', metrics), ('metrics_by_target.csv', pd.concat(target_rows)),
        ('scope_comparison.csv', pd.concat(scope_rows)),
        ('metrics_by_observed_grade.csv', pd.concat(level_rows)), ('confusion.csv', pd.concat(confusion)),
        ('panel_confusion.csv', pd.DataFrame(panel_confusion)), ('tail_precision.csv', pd.DataFrame(tail_rows)),
        ('inner_selection.csv', pd.DataFrame(choices)), ('training_priors.csv', pd.DataFrame(priors)),
        ('selected_policies.csv', pd.DataFrame(policy_records))]:
        save_csv(o, name, table)
    pd.concat([pd.DataFrame(dict(repeat=r, method=name, beer=np.repeat(ids.beer, len(targets)), target=np.tile(targets, len(q)),
        predicted_grade=g[r].ravel())) for r in range(len(folds)) for name, g in gradients.items()]).to_csv(o/'oof_grades.csv.gz',
            index=False, compression={'method': 'gzip', 'mtime': 0})
    deployed = 'inner_selected' if adopted else 'shared_et'
    write_scorecard(o, pd.DataFrame(dict(beer=np.repeat(ids.beer, len(targets)), target=np.tile(targets, len(q)),
        predicted_grade=gradients[deployed][0].ravel())), ids, d['scope'], bp['unlabelled_card_targets'])
    (o/'experimental').mkdir()
    write_scorecard(o/'experimental', pd.DataFrame(dict(beer=np.repeat(ids.beer, len(targets)), target=np.tile(targets, len(q)),
        predicted_grade=gradients['inner_selected'][0].ravel(), experiment_status='not_adopted' if not adopted else 'adopted')),
        ids, d['scope'], bp['unlabelled_card_targets'])
    save_csv(o, 'paired_mae_increase.csv', [dict(mean=increase, lower95=low, upper95=high, n_independent_beers=len(q))])
    write_json(o/'summary.json', dict(status='completed', adopted=adopted, protocol_sha256=content_hash(p),
        eligible_repeats=eligible_repeats, pooled_feasible=bool(pooled_feasible), selected_metrics=pooled,
        mean_mae_increase=float(increase), upper95_mae_increase=float(high),
        full_training_choice=full_choice, model_sha256=sha256(o/'model.joblib'),
        parent_evidence_sha256={name: sha256(parent/name) for name in ['model.joblib', 'oof_probabilities.csv.gz', 'protocol.json']},
        n_independent_beers=len(q), n_repeated_predictions=len(q)*len(folds), q3_started=False))
    print(metrics.query("method in ['shared_et','inner_selected']").to_string(index=False), flush=True)
    print(f'ADOPTED={adopted}; eligible_repeats={eligible_repeats}; upper95_increase={high}', flush=True)
    plot_results(o, parent)


def plot_results(output, parent):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old = pd.read_csv(parent/'metrics_by_repeat.csv')
    new = pd.read_csv(output/'metrics_by_repeat.csv')
    definitions = [('Training prior', old.query("method == 'training_prior'"), 'baseline'),
        ('Shared ET median', old.query("method == 'shared_et'"), 'retained_official'),
        ('Target forest selection', old.query("method == 'inner_selected'"), 'rejected'),
        ('Selective tail', new.query("method == 'inner_selected'"), 'experimental_not_adopted')]
    summary = []
    for name, frame, status in definitions:
        for metric in ['raw_rating_mae', 'panel_median_mae', 'raw_recall_5', 'raw_recall_6',
                       'panel_recall_5', 'panel_recall_6', 'raw_high_fpr', 'panel_high_fpr']:
            summary.append(dict(method=name, status=status, metric=metric, mean=frame[metric].mean(),
                sample_sd=frame[metric].std(ddof=1), minimum=frame[metric].min(), maximum=frame[metric].max()))
    save_csv(output, 'comparison_summary.csv', summary)
    tails = pd.read_csv(output/'tail_precision.csv').query("method == 'inner_selected' and scope == 'all50'")
    precision = [tails.loc[tails.event.eq(event), 'expected_raw_true_positives'].sum()/tails.loc[tails.event.eq(event), 'n_predicted'].sum()
                 for event in ['5', '6', 'ge5']]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
    colors = ['#adb5bd', '#236b8e', '#aa9bb7', '#c27b3f']
    labels = ['Prior', 'Official median', 'Target selection', 'Selective tail']
    for i, (_, frame, _) in enumerate(definitions):
        axes[0].bar(i, frame.raw_rating_mae.mean(), color=colors[i])
        axes[0].scatter([i]*3, frame.raw_rating_mae, s=13, color='black')
    axes[0].set(xticks=range(4), xticklabels=labels, ylim=(.78, .84), ylabel='Individual-rating MAE', title='Only the median remains official')
    axes[0].tick_params(axis='x', labelrotation=25)
    x = np.arange(3)
    for offset, metric, label, color in [(-.18, 'raw_recall_5', 'Exact 5', '#4a8a9e'), (.18, 'raw_recall_6', 'Exact 6', '#c27b3f')]:
        values = [frame[metric].mean()*100 for _, frame, _ in definitions[1:]]
        axes[1].bar(x+offset, values, width=.36, label=label, color=color)
    axes[1].set(xticks=x, xticklabels=labels[1:], ylabel='Individual exact recall (%)', title='Improved tail; stability gate failed')
    axes[1].tick_params(axis='x', labelrotation=25); axes[1].legend()
    axes[2].bar(['Exact 5', 'Exact 6', 'At least 5'], np.asarray(precision)*100, color=['#4a8a9e', '#c27b3f', '#637c69'])
    axes[2].set(ylabel='Correct among issued predictions (%)', ylim=(0, 100), title='Unadopted selective-tail precision')
    for i, v in enumerate(precision):
        axes[2].text(i, v*100+2, f'{v*100:.1f}%', ha='center')
    fig.savefig(output/'q1_before_after.png', dpi=170)
    fig.savefig(output/'q1_before_after.svg')
    plt.close(fig)
    svg = output/'q1_before_after.svg'
    svg.write_text('\n'.join(v.rstrip() for v in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--parent', type=Path, default=ROOT/'outputs/q1_target_20261026')
    parser.add_argument('--output', type=Path, required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args())
