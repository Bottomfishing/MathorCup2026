"""Frozen finite Q1 comparison on original panel ratings, grouped by beer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_audit.raw_sensory import check_file, load_raw, normalize_panel, link_beers
from beer_models.baseline import load_data, make_splits, mask_observed
from beer_models.explain import stratified_draw
from beer_models.raw_panel import (rating_weights, rating_distribution, median_grades,
    rating_loss, RawPanelModel, predict_raw_frame)

PROTOCOL = ROOT / 'configs/r2_raw_panel_protocol_20261026.json'
RAW_MANIFEST = ROOT / 'data/reference/zenodo23624915/download_manifest.json'


from beer_models.panel_data import content_hash


from beer_models.panel_data import load_inputs


def summary_metrics(distribution, grades):
    median = median_grades(distribution)
    distance = abs(np.arange(7) - grades[..., None])
    error = abs(grades - median)
    return dict(raw_rating_mae=float((distribution * distance).sum(axis=2).mean()),
        raw_rating_exact=float(np.take_along_axis(distribution, grades[..., None], axis=2).mean()),
        raw_rating_within_one=float((distribution * (distance <= 1)).sum(axis=2).mean()),
        panel_median_mae=float(error.mean()), panel_median_exact=float((error == 0).mean()),
        panel_median_within_one=float((error <= 1).mean()))


def metric_tables(data, predictions):
    q, primary = data['distribution'], data['primary']
    scopes = [('all50', np.ones(len(primary), bool)), ('primary45', primary)]
    overall, targets, tails, confusion, per_beer = [], [], [], [], []
    for method, result in predictions.items():
        grades = result['grade']
        for scope, mask in scopes:
            overall.append(dict(method=method, scope=scope, n_beers=len(q), n_targets=int(mask.sum()),
                **summary_metrics(q[:, mask], grades[:, mask])))
            for k in range(7):
                mass = q[:, mask, k]
                delta = abs(grades[:, mask] - k)
                tails.append(dict(method=method, scope=scope, observed_grade=k,
                    balanced_grade_mass=float(mass.sum()),
                    grade_mae=float((mass * delta).sum() / mass.sum()),
                    exact_accuracy=float((mass * (delta == 0)).sum() / mass.sum()),
                    within_one_accuracy=float((mass * (delta <= 1)).sum() / mass.sum())))
                for predicted in range(7):
                    value = float((mass * (grades[:, mask] == predicted)).sum())
                    confusion.append(dict(method=method, scope=scope, observed_grade=k,
                        predicted_grade=predicted, balanced_mass=value, row_fraction=value / float(mass.sum())))
        for j, target in enumerate(data['targets']):
            targets.append(dict(method=method, target=target, primary=bool(primary[j]),
                **summary_metrics(q[:, j:j+1], grades[:, j:j+1])))
        losses = rating_loss(q, grades).mean(axis=1)
        per_beer.extend(dict(method=method, beer=int(beer), raw_rating_mae=float(loss))
            for beer, loss in zip(data['ids'].beer, losses))
    return {'model_comparison.csv': pd.DataFrame(overall), 'metrics_by_target.csv': pd.DataFrame(targets),
        'metrics_by_observed_grade.csv': pd.DataFrame(tails), 'confusion.csv': pd.DataFrame(confusion),
        'loss_by_beer.csv': pd.DataFrame(per_beer)}


def supplement_tables(data, predictions, folds, stress):
    rows = []
    for method, result in predictions.items():
        for fold in sorted(set(folds)):
            mask = folds == fold
            rows.append(dict(method=method, outer_fold=int(fold), n_beers=int(mask.sum()),
                **summary_metrics(data['distribution'][mask], result['grade'][mask])))
    missing = []
    for (rate, repeat), frame in stress.groupby(['additional_missing_rate', 'repeat']):
        missing.append(dict(additional_missing_rate=rate, repeat=int(repeat),
            raw_rating_mae=float(np.average(frame.raw_rating_mae, weights=frame.n_beers)),
            changed_grade_fraction=float(np.average(frame.changed_grade_fraction, weights=frame.n_beers))))
    return {'fold_metrics.csv': pd.DataFrame(rows), 'missingness_summary.csv': pd.DataFrame(missing)}


def write_scorecard(output, main, ids, scope, missing_targets):
    main = main.copy()
    main['estimated_grade_0_6'] = main.predicted_grade.astype('Int64')
    main['score_status'] = 'raw_panel_balanced_median_OOF_prediction'
    extras = pd.DataFrame([dict(beer=int(beer), target=target, estimated_grade_0_6=pd.NA,
        score_status='no_raw_panel_label') for beer in ids.beer for target in missing_targets])
    complete = pd.concat([main, extras], ignore_index=True)
    save_csv(output, 'oof_scores_all52.csv', complete)
    wide = complete.pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[ids.beer, scope.target_id].reset_index()
    save_csv(output, 'estimated_grades_all52.csv', wide)
    save_csv(output, 'estimated_grades_primary47.csv', wide[['beer', *scope.loc[scope.acceptance_scope.eq('main_content'), 'target_id']]])


def run(source, raw_path, output):
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    data = load_inputs(source, raw_path, protocol)
    x, q = data['x'], data['distribution']
    ids, targets, styles = data['ids'], data['targets'], data['ids'].tasting_category_fine.to_numpy()
    methods = protocol['candidates']
    splits = make_splits(styles, protocol['outer_folds'], protocol['outer_seed'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'models').mkdir()
    write_json(output / 'protocol.json', protocol)
    fit_log, prep_log, split_log, choice_log, inner_log, stress_log = [], [], [], [], [], []
    predictions = {name: {key: np.zeros(q.shape[:2], dtype=float if key == 'raw_median_estimate' else int)
        for key in ['grade', 'unconstrained_grade', 'raw_median_estimate']} for name in methods + ['inner_selected']}
    outer_folds = np.empty(len(x), int)

    def fit(method, train, seed, context):
        print(f'Fit {context} {method}: {len(train)} beers', flush=True)
        tick = time.perf_counter()
        lookup = np.full(len(x), -1, int); lookup[train] = np.arange(len(train))
        select = lookup[data['positions']] >= 0
        model = RawPanelModel(method, protocol, seed, data['parents']).fit(x[train],
            lookup[data['positions'][select]], data['scores'][select], data['weights'][select], q[train])
        imputer_error = float(abs(model.imputer.statistics_ - np.nanmedian(x[train], axis=0)).max())
        assert imputer_error == 0
        mass = np.bincount(lookup[data['positions'][select]], weights=data['weights'][select], minlength=len(train))
        np.testing.assert_allclose(mass, 1.0, rtol=0, atol=1e-12)
        fit_log.append(dict(context=context, method=method, seed=seed, n_training_beers=len(train),
            n_rating_records=int(select.sum()), seconds=time.perf_counter()-tick))
        prep_log.append(dict(context=context, method=method, unique_beer_imputation_max_difference=imputer_error,
            maximum_per_beer_weight_error=float(abs(mass-1).max()), input_columns=x.shape[1]))
        return model

    def choose(train, context, context_number):
        inner = make_splits(styles[train], protocol['inner_folds'], protocol['inner_seed'] + context_number)
        inner_predictions = {name: np.zeros((len(train), len(targets)), int) for name in methods}
        for fold, (it, iv) in enumerate(inner):
            for role, positions in [('train', train[it]), ('validation', train[iv])]:
                split_log.extend(dict(context=context, inner_fold=fold, role=role, beer=int(ids.beer.iloc[p])) for p in positions)
            for method in methods:
                model = fit(method, train[it], protocol['model_seed'] + context_number * 100 + fold,
                            f'{context}_inner_{fold}')
                result = model.predict(x[train[iv]])
                inner_predictions[method][iv] = result['grade']
                inner_log.append(pd.DataFrame({'context': context, 'inner_fold': fold, 'method': method,
                    'beer': np.repeat(ids.beer.iloc[train[iv]].to_numpy(), len(targets)),
                    'target': np.tile(targets, len(iv)), 'predicted_grade': result['grade'].ravel()}))
        losses = [float(rating_loss(q[train], inner_predictions[name]).mean()) for name in methods]
        selected = next(name for name, loss in zip(methods, losses) if loss <= min(losses) + 1e-12)
        choice_log.extend(dict(context=context, method=name, inner_raw_rating_mae=loss, selected=name == selected)
                          for name, loss in zip(methods, losses))
        print(f'Selected {context}: {selected}', flush=True)
        return selected

    for fold, (train, test) in enumerate(splits):
        outer_folds[test] = fold
        selected = choose(train, f'outer_{fold}', fold)
        for method in methods:
            model = fit(method, train, protocol['model_seed'] + fold * 100 + 99, f'outer_{fold}_final')
            result = model.predict(x[test])
            for key in result:
                predictions[method][key][test] = result[key]
                if method == selected:
                    predictions['inner_selected'][key][test] = result[key]
            joblib.dump(model, output / 'models' / f'{method}_fold{fold}.joblib', compress=3)
            if method == selected:
                for rate in protocol['additional_missing_rates']:
                    for repeat in range(protocol['missing_repeats']):
                        masked, _ = mask_observed(x[test], rate, protocol['model_seed']+1000+fold*100+repeat)
                        after = model.predict(masked)['grade']
                        stress_log.append(dict(outer_fold=fold, additional_missing_rate=rate, repeat=repeat,
                            n_beers=len(test), selected_method=method,
                            raw_rating_mae=float(rating_loss(q[test], after).mean()),
                            changed_grade_fraction=float((after != result['grade']).mean())))

    full_selected = choose(np.arange(len(x)), 'full_training', len(splits))
    full_model = fit(full_selected, np.arange(len(x)), protocol['model_seed'] + 999, 'full_training_final')
    metrics = metric_tables(data, predictions)
    for name, frame in metrics.items():
        save_csv(output, name, frame)
    hierarchy = []
    for method, result in predictions.items():
        for stage, key in [('before_parent_cap', 'unconstrained_grade'), ('after_parent_cap', 'grade')]:
            values = result[key]
            count = sum(int((values[:, child] > values[:, parent]).sum()) for parent, child in data['parents'])
            hierarchy.append(dict(method=method, stage=stage, checked_pairs=len(x)*len(data['parents']), violations=count))
    save_csv(output, 'hierarchy.csv', hierarchy)
    all_rows = []
    for method, result in predictions.items():
        all_rows.append(pd.DataFrame({'method': method, 'beer': np.repeat(ids.beer.to_numpy(), len(targets)),
            'outer_fold': np.repeat(outer_folds, len(targets)), 'target': np.tile(targets, len(x)),
            'predicted_grade': result['grade'].ravel(), 'unconstrained_grade': result['unconstrained_grade'].ravel(),
            'raw_median_estimate': result['raw_median_estimate'].ravel(),
            'observed_panel_median': median_grades(q).ravel(),
            'weighted_raw_absolute_loss': rating_loss(q, result['grade']).ravel()}))
    pd.concat(all_rows).to_csv(output / 'oof_predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.concat(inner_log).to_csv(output / 'inner_predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    save_csv(output, 'inner_splits.csv', split_log); save_csv(output, 'inner_selection.csv', choice_log)
    save_csv(output, 'fit_log.csv', fit_log); save_csv(output, 'preprocessing_checks.csv', prep_log)
    save_csv(output, 'missingness_by_fold.csv', stress_log)
    for name, table in supplement_tables(data, predictions, outer_folds, pd.DataFrame(stress_log)).items():
        save_csv(output, name, table)
    save_csv(output, 'outer_folds.csv', pd.DataFrame({'beer': ids.beer, 'outer_fold': outer_folds}))
    weights = pd.DataFrame({'raw_record_1based': np.arange(1, len(data['raw'])+1),
        'beer': data['raw'].beer, 'taster': data['raw'].taster, 'weight_within_beer': data['weights'],
        'outer_fold': outer_folds[data['positions']]})
    save_csv(output, 'rating_weights_and_folds.csv', weights)
    truth = pd.DataFrame({'beer': np.repeat(ids.beer.to_numpy(), len(targets)), 'target': np.tile(targets, len(x)),
        'panel_weighted_median': median_grades(q).ravel(),
        'panel_weighted_mean': (q * np.arange(7)).sum(axis=2).ravel(),
        **{f'probability_grade_{k}': q[:, :, k].ravel() for k in range(7)}})
    save_csv(output, 'observed_panel_distribution.csv', truth)
    save_csv(output, 'panel_disagreement.csv', [dict(scope=scope,
        empirical_panel_median_raw_absolute_deviation=float(rating_loss(q[:, mask], median_grades(q[:, mask])).mean()),
        interpretation='Observed-panel best constant action per beer; a data diagnostic, not an achievable forecast or population noise floor')
        for scope, mask in [('all50', np.ones(len(targets), bool)), ('primary45', data['primary'])]])

    base = rating_loss(q, predictions['training_median']['grade']).mean(axis=1)
    paired = []
    for method in methods[1:] + ['inner_selected']:
        difference = base - rating_loss(q, predictions[method]['grade']).mean(axis=1)
        rng = np.random.default_rng(protocol['bootstrap_seed'])
        draws = np.array([difference[stratified_draw(styles, rng)].mean() for _ in range(protocol['bootstrap_repeats'])])
        paired.append(dict(method=method, mean_improvement=float(difference.mean()),
            lower95=float(np.quantile(draws, .025)), upper95=float(np.quantile(draws, .975)),
            scope='paired_beer_style_stratified_fixed_OOF_losses_not_refitting_uncertainty'))
    save_csv(output, 'paired_uncertainty.csv', paired)
    selected_ci = paired[-1]
    qualified = selected_ci['mean_improvement'] > 0 and selected_ci['lower95'] > 0
    bundle = dict(bundle_kind='r2_raw_panel', raw_panel_model=full_model, selected_model=full_selected,
        features=data['features'], targets=targets, primary=data['primary'],
        scorecard_scope=data['scope'].to_dict('records'), protocol=protocol,
        raw_source_sha256=data['manifest']['sha256'], qualified_on_historical_grouped_folds=qualified)
    joblib.dump(bundle, output / 'model.joblib', compress=3)
    frame = pd.read_csv(source / '1.csv', dtype=str)
    long, primary, wide, quality = predict_raw_frame(frame, bundle)
    save_csv(output, 'full_fit_scores_all52.csv', long)
    save_csv(output, 'full_fit_grades_primary47.csv', primary)
    save_csv(output, 'full_fit_grades_all52.csv', wide)
    save_csv(output, 'sample_quality.csv', quality)
    # Main delivery uses OOF estimates, with missing labels represented by explicit blank slots.
    main = pd.concat(all_rows).query("method == 'inner_selected'").copy()
    write_scorecard(output, main, ids, data['scope'], protocol['unlabelled_card_targets'])
    summary = dict(status='completed', baseline_commit=protocol['baseline_commit'],
        protocol_content_sha256=content_hash(protocol), raw_source_sha256=data['manifest']['sha256'],
        source_sha256={name: sha256(source / name) for name in ['1.csv', '4.csv', '8.xlsx']},
        n_beers=len(x), n_raw_rating_records=len(data['raw']), n_features=x.shape[1], n_targets=len(targets),
        raw_scale='observed_original_0_6_no_relative_ECDF', primary_metric=protocol['primary_metric'],
        n_model_fits=len(fit_log), full_training_selected_method=full_selected,
        qualified_raw_scale_workflow_on_historical_folds=qualified,
        mean_improvement_over_training_median=selected_ci['mean_improvement'],
        improvement_lower95=selected_ci['lower95'], improvement_upper95=selected_ci['upper95'],
        unlabelled_targets=protocol['unlabelled_card_targets'],
        model_sha256=sha256(output / 'model.joblib'),
        uncertainty_scope='Fixed OOF beer-loss resampling, conditional on fitted models and historical splits.',
        task_R3_started=False)
    write_json(output / 'summary.json', summary)
    plot_results(output)
    print(metrics['model_comparison.csv'].to_string(index=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    compare = pd.read_csv(output / 'model_comparison.csv').query("scope == 'all50'")
    tails = pd.read_csv(output / 'metrics_by_observed_grade.csv').query("scope == 'all50'")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    axes[0].bar(compare.method, compare.raw_rating_mae, color=['#8c99a5', '#55908d', '#bd9960', '#a87550', '#236b76'])
    axes[0].tick_params(axis='x', labelrotation=25)
    axes[0].set(ylabel='Beer/taster-balanced absolute error', title='Original 0-6 ratings: grouped OOF')
    for method in ['training_median', 'extra_trees', 'inner_selected']:
        frame = tails.loc[tails.method.eq(method)]
        axes[1].plot(frame.observed_grade, frame.grade_mae, marker='o', label=method)
    axes[1].set(xlabel='Observed original grade', ylabel='Conditional absolute error', xticks=range(7), title='Error by rating level')
    axes[1].legend()
    fig.savefig(output / 'raw_grade_diagnostics.png', dpi=160)
    fig.savefig(output / 'raw_grade_diagnostics.svg')
    svg = output / 'raw_grade_diagnostics.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent / 'Raw sensory data anonymized.csv')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.raw, args.output)
