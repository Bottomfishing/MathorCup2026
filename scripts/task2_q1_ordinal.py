"""Frozen task2 comparison: inner-selected per-target cumulative ordinal models."""
import argparse
import json
import hashlib
from pathlib import Path
import sys
import warnings
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.baseline import load_data, make_splits
from beer_models.explain import stratified_draw
from beer_models.final_model import relative_grades
from beer_models.ordinal import seven_level_probabilities, grade_decisions
from beer_models.ordered_models import make_ordered_candidate, training_choices, SelectedGradeModel


PROTOCOL = Path('configs/task2_ordinal_protocol_20261025.json')


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def performance(predicted, observed):
    error = abs(np.asarray(predicted) - np.asarray(observed))
    return dict(n=int(error.size), grade_mae=float(error.mean()),
        exact_accuracy=float((error == 0).mean()), within_one_accuracy=float((error <= 1).mean()))


def run(source, output, base_model):
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    ids, xf, yf, primary_names = load_data(source, Path('outputs/p0_audit_20261023_final'))
    final = json.loads(Path('configs/final_protocol.json').read_text())
    features = [f for f in xf if f not in final['excluded_derived_features']]
    x, y = xf[features].to_numpy(float), yf.to_numpy(float)
    styles = ids.tasting_category_fine.to_numpy()
    targets, methods = list(yf), protocol['candidates']
    primary = np.array([t in primary_names for t in targets])
    splits = make_splits(styles, protocol['outer_folds'], protocol['outer_split_seed'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', protocol)
    inner_records, split_records, choice_records, prep_records = [], [], [], []
    truth = np.empty_like(y, int)
    oof = {name: np.empty_like(y, int) for name in methods + ['inner_selected']}
    probabilities = {name: np.empty(y.shape + (7,)) for name in oof}
    outer_fold = np.empty(len(y), int)

    def fit(name, train, reference, seed, context):
        model = make_ordered_candidate(name, protocol, seed)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(x[train], relative_grades(reference, reference))
        np.testing.assert_allclose(model.named_steps['impute'].statistics_, np.nanmedian(x[train], axis=0))
        prep_records.append(dict(context=context, candidate=name, n_train=len(train),
            training_imputation_verified=True))
        return model

    def select_inside(train, context, context_seed):
        inner_pred = np.empty((len(methods), len(train), len(targets)), int)
        inner_truth = np.empty((len(train), len(targets)), int)
        for fold, (it, iv) in enumerate(make_splits(styles[train], protocol['inner_folds'], context_seed)):
            fitting, validation = train[it], train[iv]
            expected = relative_grades(y[fitting], y[validation])
            inner_truth[iv] = expected
            for role, positions in [('train', fitting), ('validation', validation)]:
                split_records.extend(dict(context=context, fold=fold, role=role, beer=ids.beer.iloc[i]) for i in positions)
            for m, name in enumerate(methods):
                model = fit(name, fitting, y[fitting], context_seed + fold, f'{context}/inner{fold}')
                pred = grade_decisions(seven_level_probabilities(model, x[validation]))['conditional_median']
                inner_pred[m, iv] = pred
                inner_records.extend(dict(context=context, fold=fold, method=name,
                    beer=ids.beer.iloc[i], target=t, observed=int(expected[k, j]), predicted=int(pred[k, j]))
                    for k, i in enumerate(validation) for j, t in enumerate(targets))
        choices, errors = training_choices(inner_pred, inner_truth)
        for j, target in enumerate(targets):
            choice_records.extend(dict(context=context, target=target, method=name,
                inner_mae=errors[m, j], selected=m == choices[j]) for m, name in enumerate(methods))
        return choices

    for fold, (train, test) in enumerate(splits):
        print(f'Task2 outer fold {fold}: inner selection', flush=True)
        choices = select_inside(train, f'outer{fold}', protocol['seed'] + 100 * fold)
        truth[test] = relative_grades(y[train], y[test])
        outer_fold[test] = fold
        for m, name in enumerate(methods):
            model = fit(name, train, y[train], protocol['current_tree_seed'] + fold, f'outer{fold}/refit')
            p = seven_level_probabilities(model, x[test])
            probabilities[name][test] = p
            oof[name][test] = grade_decisions(p)['conditional_median']
            for j in np.where(choices == m)[0]:
                probabilities['inner_selected'][test, j] = p[:, j]
                oof['inner_selected'][test, j] = oof[name][test, j]
        print(f'Task2 outer fold {fold}: evaluation complete', flush=True)

    historical = pd.read_csv('outputs/q1_task2_20261025/scores_all52_long.csv', dtype={'beer': str})
    historical = historical[historical.panel_target_available]
    expected = historical.pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[ids.beer.astype(str), targets]
    np.testing.assert_array_equal(oof[methods[0]], expected.to_numpy(int))
    expected_truth = historical.pivot(index='beer', columns='target', values='observed_relative_grade').loc[ids.beer.astype(str), targets]
    np.testing.assert_array_equal(truth, expected_truth.to_numpy(int))
    summaries, by_target, by_grade, confusion, distribution, folds, long = [], [], [], [], [], [], []
    for name, pred in oof.items():
        for scope, mask in [('primary45', primary), ('all50', np.ones(len(targets), bool))]:
            summaries.append(dict(method=name, scope=scope, **performance(pred[:, mask], truth[:, mask])))
        for j, target in enumerate(targets):
            by_target.append(dict(method=name, target=target, primary=bool(primary[j]),
                **performance(pred[:, j], truth[:, j])))
        for grade in range(7):
            mask = truth[:, primary] == grade
            by_grade.append(dict(method=name, observed_grade=grade,
                **performance(pred[:, primary][mask], truth[:, primary][mask])))
            distribution.append(dict(method=name, grade=grade,
                observed_count=int((truth[:, primary] == grade).sum()), predicted_count=int((pred[:, primary] == grade).sum())))
            for predicted in range(7):
                count = int(((truth[:, primary] == grade) & (pred[:, primary] == predicted)).sum())
                confusion.append(dict(method=name, observed=grade, predicted=predicted,
                    count=count, row_fraction=count / int(mask.sum())))
        for fold, (_, test) in enumerate(splits):
            folds.append(dict(method=name, fold=fold, **performance(pred[test][:, primary], truth[test][:, primary])))
        long.append(pd.DataFrame(dict(beer=np.repeat(ids.beer.to_numpy(), len(targets)),
            target=np.tile(targets, len(ids)), primary=np.tile(primary, len(ids)),
            fold=np.repeat(outer_fold, len(targets)), method=name, observed=truth.ravel(), predicted=pred.ravel(),
            **{f'p{k}': probabilities[name][:, :, k].ravel() for k in range(7)})))
    baseline_loss = abs(oof[methods[0]][:, primary] - truth[:, primary]).mean(axis=1)
    improvement = np.stack([baseline_loss - abs(oof[m][:, primary] - truth[:, primary]).mean(axis=1)
                           for m in list(oof)[1:]], axis=1)
    rng = np.random.default_rng(protocol['seed'])
    bootstrap = np.array([improvement[stratified_draw(styles, rng)].mean(axis=0)
                          for _ in range(protocol['uncertainty_repeats'])])
    uncertainty = [dict(method=m, mae_improvement=float(improvement[:, j].mean()),
        fixed_oof_p025=float(np.quantile(bootstrap[:, j], .025)), fixed_oof_p975=float(np.quantile(bootstrap[:, j], .975)))
        for j, m in enumerate(list(oof)[1:])]
    grade_table = pd.DataFrame(by_grade).set_index(['method', 'observed_grade'])
    selected_interval = uncertainty[-1]
    tail_checks = all(grade_table.loc[('inner_selected', g), 'grade_mae'] <= grade_table.loc[(methods[0], g), 'grade_mae']
        and grade_table.loc[('inner_selected', g), 'exact_accuracy'] >= grade_table.loc[(methods[0], g), 'exact_accuracy'] for g in [0, 6])
    qualified = bool(selected_interval['fixed_oof_p025'] > 0 and tail_checks)
    for name, rows in [('model_comparison', summaries), ('metrics_by_target', by_target),
        ('metrics_by_observed_grade', by_grade), ('confusion_primary45', confusion),
        ('grade_distribution_primary45', distribution), ('fold_metrics', folds), ('paired_uncertainty', uncertainty)]:
        save_csv(output, name + '.csv', rows)
    pd.concat(long, ignore_index=True).to_csv(output / 'oof_predictions.csv.gz', index=False,
        compression={'method': 'gzip', 'mtime': 0})
    save_csv(output, 'outer_splits.csv', ids.assign(fold=outer_fold))

    print('Task2 full-data inner selection and explicit candidate bundle', flush=True)
    choices = select_inside(np.arange(len(y)), 'full_training', protocol['seed'] + 1000)
    models = [fit(name, np.arange(len(y)), y, protocol['current_tree_seed'], 'full_training/refit') for name in methods]
    selected_model = SelectedGradeModel(models, choices)
    old_bundle = base_model
    old_summary = json.loads((base_model.parent / 'summary.json').read_text())
    if sha256(old_bundle) != old_summary['model_sha256']:
        raise ValueError('Current deployed bundle differs from recorded task2 source')
    base = joblib.load(old_bundle)
    if base['features'] != features or base['targets'] != targets or 'scorecard_protocol' not in base:
        raise ValueError('Base bundle must be a matching complete current scorecard model')
    np.testing.assert_array_equal(base['reference_y'], y)
    bundle = dict(base, selected_grade_model=selected_model,
        grade_method='task2_inner_selected_per_target_relative_grade',
        task2_protocol=protocol, task2_qualified=qualified, task2_official_default=False)
    bundle['scorecard_protocol'] = dict(bundle['scorecard_protocol'],
        main_panel_grade_method=bundle['grade_method'],
        scope='Explicit task2 experimental grade candidate; existing scorecard mapping and acetaldehyde proxy assumptions retained. Not the official default.')
    joblib.dump(bundle, output / 'model.joblib', compress=3)
    pd.DataFrame(inner_records).to_csv(output / 'inner_predictions.csv.gz', index=False,
        compression={'method': 'gzip', 'mtime': 0})
    save_csv(output, 'inner_splits.csv', split_records)
    save_csv(output, 'inner_selection.csv', choice_records)
    save_csv(output, 'preprocessing_checks.csv', prep_records)
    summary = dict(status='completed', base_commit='b207b46', protocol_content_sha256=content_hash(protocol),
        source_sha256={n: sha256(source / n) for n in ['1.csv', '4.csv', '8.xlsx']},
        model_sha256=sha256(output / 'model.joblib'), parent_bundle_sha256=sha256(old_bundle),
        feature_count=len(features), target_count=len(targets),
        primary_targets=int(primary.sum()), outer_folds=3, inner_folds=2,
        current_main_OOF_predictions_reproduced=int(y.size),
        selected_workflow_qualified_on_historical_partition=qualified,
        tail_nonworsening_checks=bool(tail_checks), main_results_replaced=False,
        raw_absolute_scale_recovered=False, task3_or_later_started=False,
        uncertainty_scope='Style-stratified paired sample bootstrap of fixed OOF losses; not refitting/selection uncertainty or independent confirmation.',
        interpretation='Frozen limited comparison on previously explored historical folds. Full-data explicit bundle has training-only per-target choices; keep default pending authorized task5 repeated validation.')
    write_json(output / 'summary.json', summary)
    plot_results(output)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    confusion = pd.read_csv(output / 'confusion_primary45.csv')
    grades = pd.read_csv(output / 'metrics_by_observed_grade.csv')
    metrics = pd.read_csv(output / 'model_comparison.csv')
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    names = ['current_multioutput_tree', 'inner_selected']
    for ax, name, title in zip(axes[0], names, ['Current model', 'Training-selected ordinal workflow']):
        matrix = confusion[confusion.method.eq(name)].pivot(index='observed', columns='predicted', values='row_fraction').to_numpy()
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
        ax.set(title=title, xlabel='Predicted relative grade', ylabel='Observed relative grade', xticks=range(7), yticks=range(7))
        for i in range(7):
            for j in range(7):
                ax.text(j, i, f'{matrix[i,j]:.0%}', ha='center', va='center', fontsize=8,
                    color='white' if matrix[i,j] > .5 else 'black')
    fig.colorbar(image, ax=axes[0].tolist(), shrink=.8, label='Row fraction')
    for name, label in zip(names, ['Current', 'Inner-selected']):
        g = grades[grades.method.eq(name)]
        axes[1, 0].plot(g.observed_grade, g.grade_mae, marker='o', label=label)
    axes[1, 0].set(xlabel='Observed relative grade', ylabel='Mean absolute grade error', xticks=range(7))
    axes[1, 0].legend()
    m = metrics[metrics.scope.eq('primary45')]
    axes[1, 1].bar(range(len(m)), m.grade_mae, color=['#708090', '#7799bb', '#557799', '#246a73'])
    axes[1, 1].set(xticks=range(len(m)), xticklabels=['Current', 'Ordinal C=.1', 'Ordinal C=1', 'Inner-selected'], ylabel='Primary45 MAE')
    axes[1, 1].tick_params(axis='x', labelrotation=18)
    fig.suptitle('Task2: historical OOF relative grades (not raw intensity)', fontsize=13)
    fig.savefig(output / 'grade_diagnostics.png', dpi=160)
    fig.savefig(output / 'grade_diagnostics.svg')
    svg = output / 'grade_diagnostics.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-model', type=Path, default=Path('outputs/q1_task2_20261025/model.joblib'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.output, args.base_model)
