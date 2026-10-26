"""Frozen, repeated nested Q1 target-specific distribution experiment."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.baseline import make_splits
from beer_models.raw_panel import RawPanelModel, median_grades, rating_loss, enforce_hierarchy, predict_raw_frame
from beer_models.tail_risk import tail_metrics, survival, binary_loss, RawOrdinalModel, fit_risk, risk_frame
from beer_models.target_panel import (TargetDistributionForest, SelectedTargetPanel, distributions,
                                      select_targets, adoption_decision)
from beer_models.repeated_validation import beer_averaged_interval, repeat_summary
from r2_raw_q1 import load_inputs, RAW_MANIFEST, content_hash, metric_tables, summary_metrics, write_scorecard

PROTOCOL = ROOT/'configs/q1_target_protocol_20261026.json'
BASE = ROOT/'configs/r2_raw_panel_protocol_20261026.json'
LEGACY = ROOT/'outputs/r2b_tail_risk_20261026'


def fit_shared(data, train, protocol, seed):
    lookup = np.full(len(data['x']), -1, int)
    lookup[train] = np.arange(len(train))
    rows = lookup[data['positions']] >= 0
    return RawPanelModel('extra_trees', protocol, seed, data['parents']).fit(data['x'][train],
        lookup[data['positions'][rows]], data['scores'][rows], data['weights'][rows], data['distribution'][train])


def fit_models(data, train, p, bp, seed):
    x, q = data['x'], data['distribution']
    return dict(shared_et=fit_shared(data, train, bp, seed),
        target_leaf3=TargetDistributionForest(3, p, seed).fit(x[train], q[train]),
        target_leaf1=TargetDistributionForest(1, p, seed).fit(x[train], q[train]))


def probability_frame(probabilities, ids, targets, **metadata):
    return pd.concat([pd.DataFrame(dict(**metadata, method=name,
        beer=np.repeat(np.asarray(ids), len(targets)), target=np.tile(targets, len(ids)),
        **{f'p{k}': values[:, :, k].ravel() for k in range(7)}))
        for name, values in probabilities.items()], ignore_index=True)


def from_frame(frame, beers, targets):
    return np.stack([frame.pivot(index='beer', columns='target', values=f'p{k}').loc[beers, targets].to_numpy()
                     for k in range(7)], axis=2)


def probability_tables(probabilities, q, targets, repeat):
    rows = []
    truth = np.clip(survival(q), 0, 1)
    for method, prob in probabilities.items():
        score = np.clip(survival(prob), 0, 1)
        for j, target in enumerate(targets):
            for cut in [1, 5, 6]:
                y, s = truth[:, j, cut-1], score[:, j, cut-1]
                # Rank diagnostics treat sub-1e-12 numerical differences as ties;
                # parallel tree sums can differ at machine epsilon on zero risks.
                ranked = np.round(s, 12)
                labels = np.r_[np.zeros(len(y)), np.ones(len(y))]
                weights = np.r_[1-y, y]
                auc = roc_auc_score(labels, np.r_[ranked, ranked], sample_weight=weights) if 0 < y.sum() < len(y) else np.nan
                ap = average_precision_score(labels, np.r_[ranked, ranked], sample_weight=weights) if y.sum() else np.nan
                rows.append(dict(repeat=repeat, method=method, target=target, rating_ge=cut,
                    expected_brier=float(binary_loss(s, y).mean()), weighted_auc=auc, weighted_ap=ap,
                    observed_prevalence=float(y.mean()), mean_probability=float(s.mean()),
                    rps=float(binary_loss(score[:, j], truth[:, j]).mean()),
                    probability_semantics='individual_rating_uncalibrated_not_safety'))
    return pd.DataFrame(rows)


def diagnostics(data, source, output):
    from openpyxl import load_workbook
    q, scores, raw = data['distribution'], data['scores'], data['raw']
    median = median_grades(q)
    rows = []
    for j, target in enumerate(data['targets']):
        for k in range(7):
            rows.append(dict(target=target, grade=k, n_raw_records=int((scores[:, j] == k).sum()),
                beer_balanced_mass=float(q[:, j, k].sum()), n_beers_with_grade=int((q[:, j, k] > 0).sum()),
                n_panel_medians=int((median[:, j] == k).sum()), missing_raw_records=int(raw[target].isna().sum())))
    save_csv(output, 'label_distribution.csv', rows)
    means = (q*np.arange(7)).sum(2)
    within = (q*(np.arange(7)-means[:, :, None])**2).sum(2).mean(0)
    between = means.var(0)
    save_csv(output, 'panel_disagreement.csv', [dict(target=t,
        empirical_median_individual_mae=float(rating_loss(q, median)[:, j].mean()),
        between_beer_mean_variance=between[j], within_beer_rating_variance=within[j],
        within_fraction=within[j]/(within[j]+between[j]),
        oracle_recall5=tail_metrics(q[:, j:j+1], median[:, j:j+1])['raw_recall_5'],
        oracle_recall6=tail_metrics(q[:, j:j+1], median[:, j:j+1])['raw_recall_6'])
        for j, t in enumerate(data['targets'])])
    # Aggregate by anonymous taster; publish no per-record scores, beer names or comments.
    taster_rows = []
    for taster, part in raw.groupby('taster', sort=True):
        ix = raw.index.get_indexer(part.index)
        residual = scores[ix] - means[data['positions'][ix]]
        taster_rows.append(dict(anonymous_taster=taster, n_records=len(ix), n_beers=part.beer.nunique(),
            average_residual_to_own_panel_mean=float(residual.mean()), fraction_grade6=float((scores[ix] == 6).mean())))
    save_csv(output, 'taster_aggregate.csv', taster_rows)
    workbook = load_workbook(source/'3.xlsx', read_only=True, data_only=False)
    scope = data['scope']
    for row in scope.itertuples():
        assert workbook[row.template_sheet][row.template_score_formula_cell].value.startswith('=IF(ISBLANK(')
    no_label = scope.loc[~scope.panel_target_available.astype(bool), 'target_id'].tolist()
    assert set(no_label) == {'A_acetaldehyde', 'F_acetaldehyde'}
    write_json(output/'data_audit.json', dict(n_beers=len(q), n_raw_records=len(raw), n_tasters=raw.taster.nunique(),
        n_model_inputs=data['x'].shape[1], n_missing_inputs=int(np.isnan(data['x']).sum()),
        raw_grade_counts=np.bincount(scores.ravel(), minlength=7).tolist(),
        panel_median_counts=np.bincount(median.ravel(), minlength=7).tolist(),
        empirical_panel_median_individual_mae=float(rating_loss(q, median).mean()),
        diagnostic_scope='Observed-panel oracle is descriptive, not a population noise floor or achievable forecast',
        raw_hierarchy_violations=sum(int((scores[:, c] > scores[:, p]).sum()) for p, c in data['parents']),
        statement_sha256=sha256(next(source.parent.glob('*.pdf'))),
        source_sha256={name: sha256(source/name) for name in ['1.csv', '3.xlsx', '4.csv', '8.xlsx']},
        unlabelled_targets=no_label))
    workbook.close()


def derive(data, p, probabilities, grades):
    metrics, target_rows, grade_rows, confusion, losses, probability_rows, panel_rows = [], [], [], [], [], [], []
    q = data['distribution']
    for repeat in range(len(p['repeat_outer_seeds'])):
        predictions = {k: dict(grade=v[repeat]) for k, v in grades.items()}
        tables = metric_tables(data, predictions)
        target_rows.append(tables['metrics_by_target.csv'].assign(repeat=repeat))
        grade_rows.append(tables['metrics_by_observed_grade.csv'].assign(repeat=repeat))
        confusion.append(tables['confusion.csv'].assign(repeat=repeat))
        losses.append(tables['loss_by_beer.csv'].assign(repeat=repeat))
        for name, v in grades.items():
            metrics.append(dict(repeat=repeat, method=name,
                **(summary_metrics(q, v[repeat]) | tail_metrics(q, v[repeat]))))
            for j, target in enumerate(data['targets']):
                target_rows[-1].loc[target_rows[-1].method.eq(name)&target_rows[-1].target.eq(target),
                    'raw_high_fpr'] = tail_metrics(q[:, j:j+1], v[repeat, :, j:j+1])['raw_high_fpr']
            # The observed panel representative has its own grade-wise confusion matrix.
            observed = median_grades(q)
            for k in range(7):
                mask = observed == k
                for predicted in range(7):
                    count = int(((v[repeat] == predicted)&mask).sum())
                    panel_rows.append(dict(repeat=repeat, method=name, observed_grade=k, predicted_grade=predicted,
                        count=count, row_fraction=count/mask.sum() if mask.sum() else np.nan))
        probability_rows.append(probability_tables({k: v[repeat] for k, v in probabilities.items()}, q, data['targets'], repeat))
    metrics = pd.DataFrame(metrics)
    intervals = []
    styles = data['ids'].tasting_category_fine.to_numpy()
    for name in grades:
        if name == 'shared_et':
            continue
        differences = np.stack([rating_loss(q, a).mean(1)-rating_loss(q, b).mean(1)
            for a, b in zip(grades['shared_et'], grades[name])])
        gain, lo, hi = beer_averaged_interval(differences, styles, p['bootstrap_repeats'], p['bootstrap_seed'], [.025, .975])
        intervals.append(dict(method=name, mae_gain=float(gain), lower95=float(lo), upper95=float(hi),
            n_independent_beers=len(q), n_repeated_predictions=len(q)*len(differences)))
    return {'metrics_by_repeat.csv': metrics, 'repeat_summary.csv': repeat_summary(metrics, ['method'],
        [c for c in metrics if c not in ['method', 'repeat']]), 'metrics_by_target.csv': pd.concat(target_rows),
        'metrics_by_observed_grade.csv': pd.concat(grade_rows), 'confusion.csv': pd.concat(confusion),
        'panel_confusion.csv': pd.DataFrame(panel_rows), 'loss_by_beer.csv': pd.concat(losses),
        'probability_metrics.csv': pd.concat(probability_rows), 'paired_intervals.csv': pd.DataFrame(intervals)}


def restore_risk(data, shared_model, bp):
    """Rebuild only full-data deployment; reuse immutable historical inner OOF evidence."""
    rp = json.loads((ROOT/'configs/r2b_tail_risk_protocol_20261026.json').read_text(encoding='utf-8'))
    inner = pd.read_csv(LEGACY/'inner_probabilities.csv.gz').query("context == 'full_training'")
    probs = {name: from_frame(part, data['ids'].beer, data['targets']) for name, part in inner.groupby('source')}
    ordinal = RawOrdinalModel(rp).fit(data['x'], data['distribution'])
    models = dict(extra_trees=shared_model, raw_ordinal=ordinal)
    risk, _ = fit_risk(models, probs, data['distribution'], rp)
    status = ['exploratory_not_safety_validation']*len(data['targets'])
    for row in pd.read_csv(LEGACY/'defect_risk_validation.csv').itertuples():
        status[data['targets'].index(row.target)] = row.status
    return risk, status, rp


def run(args):
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    bp = json.loads(BASE.read_text(encoding='utf-8'))
    data = load_inputs(args.source, args.raw, bp)
    o = args.output
    if args.resume:
        assert json.loads((o/'protocol.json').read_text(encoding='utf-8')) == p
        audit = json.loads((o/'data_audit.json').read_text(encoding='utf-8'))
        for name, digest in audit['source_sha256'].items():
            assert sha256(args.source/name) == digest
    else:
        o.mkdir(parents=True, exist_ok=False)
        (o/'models').mkdir()
        write_json(o/'protocol.json', p)
    diagnostics(data, args.source, o)
    q, x, ids, targets = data['distribution'], data['x'], data['ids'], data['targets']
    styles = ids.tasting_category_fine.to_numpy()
    names = p['methods']+['inner_selected']
    probabilities = {k: np.empty((len(p['repeat_outer_seeds']), *q.shape)) for k in names}
    grades = {k: np.empty((len(p['repeat_outer_seeds']), *q.shape[:2]), int) for k in names}
    split_rows, choices, fit_rows, fold_rows = [], [], [], []
    folds = np.empty((len(p['repeat_outer_seeds']), len(q)), int)

    def fit(train, seed, context):
        print(f'Fit {context}: {len(train)} unique beers', flush=True)
        start = time.perf_counter()
        models = fit_models(data, train, p, bp, seed)
        for name, model in models.items():
            np.testing.assert_allclose(model.imputer.statistics_, np.nanmedian(x[train], axis=0), atol=0, rtol=0)
        fit_rows.append(dict(context=context, n_beers=len(train), seed=seed, seconds=time.perf_counter()-start,
            unique_beer_imputation_verified=True, operation='new_fit'))
        return models

    def choose(train, inner_seed, model_seed, context):
        inner_prob = {k: np.empty((len(train), *q.shape[1:])) for k in p['methods']}
        cached = o/f'inner_{context}.csv.gz'
        recovered = args.resume and cached.exists()
        if recovered:
            table = pd.read_csv(cached, float_precision='round_trip')
            inner_prob = {name: from_frame(table.loc[table.method.eq(name)], ids.beer.iloc[train], targets)
                          for name in p['methods']}
        for inner_fold, (it, iv) in enumerate(make_splits(styles[train], p['inner_folds'], inner_seed)):
            for role, pos in [('train', train[it]), ('validation', train[iv])]:
                split_rows.extend(dict(context=context, inner_fold=inner_fold, role=role, beer=int(ids.beer.iloc[z])) for z in pos)
            if recovered:
                fit_rows.append(dict(context=f'{context}_inner{inner_fold}', n_beers=len(it),
                    seed=model_seed+inner_fold, seconds=None, unique_beer_imputation_verified=None,
                    operation='recovered_saved_inner_predictions_not_refit'))
                continue
            models = fit(train[it], model_seed+inner_fold, f'{context}_inner{inner_fold}')
            inner_prob['training_prior'][iv] = q[train[it]].mean(0)
            for name, model in models.items():
                inner_prob[name][iv] = distributions(model, x[train[iv]])
        selected, rows = select_targets(inner_prob, q[train], data['parents'])
        choices.extend(dict(context=context, target=targets[r.pop('target_index')], **r) for r in rows)
        if not recovered:
            frame = probability_frame(inner_prob, ids.beer.iloc[train], targets, context=context)
            frame.to_csv(cached, index=False, compression={'method': 'gzip', 'mtime': 0})
        return selected

    for repeat, outer_seed in enumerate(p['repeat_outer_seeds']):
        for fold, (train, test) in enumerate(make_splits(styles, p['outer_folds'], outer_seed)):
            context = f'r{repeat}_f{fold}'
            seed = p['seed']+repeat*1000+fold*100
            selected = choose(train, outer_seed+100+fold, seed, context)
            saved = o/'models'/f'{context}.joblib'
            if args.resume and saved.exists():
                recovered = joblib.load(saved)
                assert recovered.choices == selected
                models = recovered.models
                for model in models.values():
                    np.testing.assert_allclose(model.imputer.statistics_, np.nanmedian(x[train], axis=0), atol=0, rtol=0)
                fit_rows.append(dict(context=context+'_final', n_beers=len(train), seed=seed+99,
                    seconds=None, unique_beer_imputation_verified=True, operation='recovered_saved_outer_models'))
            else:
                models = fit(train, seed+99, context+'_final')
            point = SelectedTargetPanel(models, q[train].mean(0), selected, data['parents'])
            values = dict(training_prior=np.broadcast_to(q[train].mean(0), (len(test), *q.shape[1:])),
                          **{name: distributions(model, x[test]) for name, model in models.items()},
                          inner_selected=point.predict_distribution(x[test]))
            for name, value in values.items():
                probabilities[name][repeat, test] = value
                grades[name][repeat, test] = enforce_hierarchy(median_grades(value), data['parents'])
                fold_rows.append(dict(repeat=repeat, outer_fold=fold, method=name,
                    **tail_metrics(q[test], grades[name][repeat, test])))
            folds[repeat, test] = fold
            if not (args.resume and saved.exists()):
                joblib.dump(point, saved, compress=3)
            print(f'Completed {context}', flush=True)
    tables = derive(data, p, probabilities, grades)
    for name, table in tables.items():
        save_csv(o, name, table)
    selection_gain = tables['paired_intervals.csv'].query("method == 'inner_selected'").iloc[0]
    adopted, tail_solved = adoption_decision(tables['metrics_by_repeat.csv'], selection_gain.mae_gain,
                                            selection_gain.lower95, p)
    deployed_name = 'inner_selected' if adopted else 'shared_et'
    selected = choose(np.arange(len(q)), p['seed']+103, p['seed']+3000, 'full_training')
    cached_bundle = joblib.load(o/'model.joblib') if args.resume and (o/'model.joblib').exists() else None
    if cached_bundle is not None:
        models = cached_bundle['raw_panel_model'].models
        fit_rows.append(dict(context='full_training_final', n_beers=len(q), seed=bp['model_seed']+999,
            seconds=None, unique_beer_imputation_verified=True, operation='recovered_full_training_models_and_risk'))
    else:
        models = fit(np.arange(len(q)), bp['model_seed']+999, 'full_training_final')
    point = SelectedTargetPanel(models, q.mean(0), selected if adopted else ['shared_et']*len(targets), data['parents'])
    risk, status, rp = (cached_bundle['risk_panel_model'], cached_bundle['risk_status'], cached_bundle['r2b_protocol']) if cached_bundle is not None else restore_risk(data, models['shared_et'], bp)
    bundle = dict(bundle_kind='r2b_raw_panel', raw_panel_model=point, risk_panel_model=risk, risk_status=status,
        selected_model='target_specific_inner_selected' if adopted else 'extra_trees__median',
        features=data['features'], targets=targets, primary=data['primary'], scorecard_scope=data['scope'].to_dict('records'),
        protocol=bp, r2b_protocol=rp, q1_target_protocol=p, point_semantics='integer_balanced_panel_conditional_median_with_parent_cap',
        score_status='raw_panel_balanced_median_prediction', raw_source_sha256=data['manifest']['sha256'],
        qualified_on_historical_grouped_folds=True, tail_point_qualified_on_historical_folds=tail_solved if adopted else False,
        point_deployment_status='mae_improved_tail_unresolved' if adopted and not tail_solved else
            ('mae_improved_tail_qualified' if adopted else 'provisional_baseline_tail_not_qualified'),
        risk_provenance='Unchanged R2b risk workflow; restored from historical full_training inner OOF; not new point probabilities')
    if cached_bundle is None:
        joblib.dump(bundle, o/'model.joblib', compress=3)
    long, _, wide, quality = predict_raw_frame(pd.read_csv(args.source/'1.csv', dtype=str), bundle)
    long = long.merge(risk_frame(pd.read_csv(args.source/'1.csv', dtype=str), bundle, long.beer.drop_duplicates().to_numpy()),
                      on=['beer', 'target'], how='left', validate='one_to_one')
    save_csv(o, 'full_fit_scores_all52.csv', long)
    save_csv(o, 'full_fit_grades_all52.csv', wide)
    save_csv(o, 'sample_quality.csv', quality)
    save_csv(o, 'inner_selection.csv', choices)
    save_csv(o, 'inner_splits.csv', split_rows)
    save_csv(o, 'fit_log.csv', fit_rows)
    save_csv(o, 'fold_metrics.csv', fold_rows)
    save_csv(o, 'outer_folds.csv', [dict(repeat=r, beer=int(b), outer_fold=int(folds[r, i]))
        for r in range(len(folds)) for i, b in enumerate(ids.beer)])
    pd.concat([probability_frame({k: v[r] for k, v in probabilities.items()}, ids.beer, targets, repeat=r)
        for r in range(len(folds))]).to_csv(o/'oof_probabilities.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.concat([pd.DataFrame(dict(repeat=r, method=name, beer=np.repeat(ids.beer, len(targets)),
        target=np.tile(targets, len(q)), predicted_grade=matrix[r].ravel()))
        for r in range(len(folds)) for name, matrix in grades.items()]).to_csv(o/'oof_grades.csv.gz', index=False,
                                                                          compression={'method': 'gzip', 'mtime': 0})
    # A single declared repeat for the scorecard, never mix best predictions across repeats.
    main = pd.DataFrame(dict(beer=np.repeat(ids.beer, len(targets)), target=np.tile(targets, len(q)),
                             predicted_grade=grades[deployed_name][0].ravel(), repeat=0))
    write_scorecard(o, main, ids, data['scope'], bp['unlabelled_card_targets'])
    write_json(o/'summary.json', dict(status='completed', baseline_commit=p['baseline_commit'],
        protocol_sha256=content_hash(p), adopted=adopted, tail_solved=tail_solved, deployed_name=deployed_name,
        mae_gain=float(selection_gain.mae_gain), lower95_gain=float(selection_gain.lower95), upper95_gain=float(selection_gain.upper95),
        n_independent_beers=len(q), n_repeated_predictions=len(q)*len(folds), fit_contexts=len(fit_rows),
        n_shared_forest_fits=len(fit_rows), n_target_forest_fits=len(fit_rows)*100,
        model_sha256=sha256(o/'model.joblib'), raw_source_sha256=data['manifest']['sha256'],
        risk_evidence_sha256={name: sha256(LEGACY/name) for name in ['inner_probabilities.csv.gz', 'defect_risk_validation.csv']},
        validation_status=p['validation_status'], q3_started=False))
    print(tables['repeat_summary.csv'].query("metric in ['raw_rating_mae','raw_recall_5','raw_recall_6','raw_high_fpr']").to_string(index=False), flush=True)
    print(f'ADOPTED={adopted}; TAIL_SOLVED={tail_solved}; gain={selection_gain.mae_gain:.6f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Recover completed inner predictions/outer models with identical protocol and inputs')
    with threadpool_limits(limits=1):
        run(parser.parse_args())
