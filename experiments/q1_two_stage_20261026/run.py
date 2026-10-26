"""Independent repeated nested comparison. Never changes the official predictor."""
import argparse
import json
import sys
import time
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'src'), str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.baseline import make_splits
from beer_models.raw_panel import median_grades, enforce_hierarchy, rating_loss
from beer_models.tail_risk import tail_metrics
from beer_models.repeated_validation import beer_averaged_interval, repeat_summary
from r2_raw_q1 import load_inputs, RAW_MANIFEST, content_hash, metric_tables, summary_metrics
from q1_target_experiment import from_frame, fit_shared, BASE
from summarize_q1_target import tail_details
from two_stage import make_high_model, ExperimentalPredictor, decide

LEGACY = ROOT/'outputs/q1_target_20261026'


def policies(p):
    return [(a, b) for a in p['decision_mae_penalties'] for b in p['decision_fp_penalties']]


def scoped_metrics(q, grade, primary):
    return [dict(scope=scope, **(summary_metrics(q[:, mask], grade[:, mask]) | tail_metrics(q[:, mask], grade[:, mask])))
        for scope, mask in [('all50', np.ones(len(primary), bool)), ('primary45', primary)]]


def choose(predictions, q, base, primary, p):
    limits = p['success']
    base_metrics = {r['scope']: r for r in scoped_metrics(q, base, primary)}
    rows = []
    for i, grade in predictions.items():
        records = scoped_metrics(q, grade, primary)
        eligible = all(r['raw_rating_mae'] <= base_metrics[r['scope']]['raw_rating_mae']+limits['maximum_mae_increase']+1e-12
            and r['raw_high_fpr'] <= limits['maximum_high_fpr'] and r['panel_high_fpr'] <= limits['maximum_high_fpr'] for r in records)
        rows.extend(dict(policy=i, eligible=bool(eligible), **r) for r in records)
    frame = pd.DataFrame(rows)
    valid = frame.loc[frame.scope.eq('primary45')&frame.eligible]
    selected = (min(valid.to_dict('records'), key=lambda r: (-min(r['raw_recall_5'], r['raw_recall_6']),
        -(r['raw_recall_5']+r['raw_recall_6'])/2, r['raw_rating_mae'], r['policy']))['policy'] if len(valid) else None)
    frame['selected'] = frame.policy.eq(selected)
    return selected, frame


def probability_frame(prob, ids, targets, **meta):
    return pd.DataFrame(dict(**meta, beer=np.repeat(np.asarray(ids), len(targets)), target=np.tile(targets, len(ids)),
        **{f'p{k}': prob[:, :, k].ravel() for k in range(7)}))


def probability_metrics(probabilities, q, targets, repeat):
    # Sum actual tail masses directly; 1-cumsum can invent epsilon positives
    # for a class with no observed labels (notably coriander6).
    observed = np.stack([q[:, :, k:].sum(2) for k in range(1, 7)], axis=2).clip(0, 1)
    rows = []
    for name, p in probabilities.items():
        cumulative = np.stack([p[:, :, k:].sum(2) for k in range(1, 7)], axis=2).clip(0, 1)
        expected = observed*(1-cumulative)**2+(1-observed)*cumulative**2
        for j, target in enumerate(targets):
            for cut in [1, 5, 6]:
                y, pred = observed[:, j, cut-1], cumulative[:, j, cut-1]
                labels = np.r_[np.zeros(len(y)), np.ones(len(y))]
                weights = np.r_[1-y, y]
                rank = np.r_[pred.round(12), pred.round(12)]
                auc = roc_auc_score(labels, rank, sample_weight=weights) if 0 < y.sum() < len(y) else np.nan
                ap = average_precision_score(labels, rank, sample_weight=weights) if y.sum() > 0 else np.nan
                rows.append(dict(repeat=repeat, method=name, target=target, rating_ge=cut,
                    expected_brier=float(expected[:, j, cut-1].mean()), weighted_auc=auc, weighted_ap=ap,
                    observed_prevalence=float(y.mean()), mean_probability=float(pred.mean()), rps=float(expected[:, j].mean()),
                    probability_semantics='individual_rating_uncalibrated_not_safety'))
    return pd.DataFrame(rows)


def summarize(d, p, grades, probabilities, output):
    q, primary = d['distribution'], d['primary']
    overall, targets, levels, confusion, panels, tails, risks, losses = [], [], [], [], [], [], [], []
    for r in range(3):
        values = {k: dict(grade=v[r]) for k, v in grades.items()}
        tables = metric_tables(d, values)
        targets.append(tables['metrics_by_target.csv'].assign(repeat=r))
        levels.append(tables['metrics_by_observed_grade.csv'].assign(repeat=r))
        confusion.append(tables['confusion.csv'].assign(repeat=r))
        observed = median_grades(q)
        for name, matrix in grades.items():
            g = matrix[r]
            overall.extend(dict(repeat=r, method=name, **a) for a in scoped_metrics(q, g, primary))
            losses.extend(dict(repeat=r, method=name, beer=int(beer), raw_rating_mae=float(value))
                for beer, value in zip(d['ids'].beer, rating_loss(q, g).mean(1)))
            for scope, mask in [('all50', np.ones(len(primary), bool)), ('primary45', primary)]:
                tails.extend(dict(repeat=r, method=name, scope=scope, **a) for a in tail_details(q[:, mask], g[:, mask]))
                for k in range(7):
                    truth = observed[:, mask] == k
                    for j in range(7):
                        n = int((truth&(g[:, mask] == j)).sum())
                        panels.append(dict(repeat=r, method=name, scope=scope, observed_grade=k, predicted_grade=j,
                            count=n, row_fraction=n/truth.sum() if truth.sum() else np.nan))
        risks.append(probability_metrics({k: v[r] for k, v in probabilities.items()}, q, d['targets'], r))
    overall = pd.DataFrame(overall)
    result = {'metrics.csv': overall, 'metrics_by_target.csv': pd.concat(targets), 'metrics_by_grade.csv': pd.concat(levels),
        'confusion.csv': pd.concat(confusion), 'panel_confusion.csv': pd.DataFrame(panels), 'tail_precision.csv': pd.DataFrame(tails),
        'probability_metrics.csv': pd.concat(risks), 'loss_by_beer.csv': pd.DataFrame(losses),
        'repeat_summary.csv': repeat_summary(overall, ['method', 'scope'], [c for c in overall if c not in ['method', 'scope', 'repeat']])}
    intervals = []
    for scope, mask in [('all50', np.ones(len(primary), bool)), ('primary45', primary)]:
        for reference in ['original_median', 'old_same_selector']:
            delta = np.stack([rating_loss(q[:, mask], grades['new_selected'][r][:, mask]).mean(1)-
                rating_loss(q[:, mask], grades[reference][r][:, mask]).mean(1) for r in range(3)])
            mean, lo, hi = beer_averaged_interval(delta, d['ids'].tasting_category_fine, p['bootstrap_repeats'], p['bootstrap_seed'], [.025, .975])
            intervals.append(dict(scope=scope, reference=reference, mean_mae_increase=float(mean), lower95=float(lo), upper95=float(hi), n_independent_beers=len(q)))
    result['paired_intervals.csv'] = pd.DataFrame(intervals)
    for name, frame in result.items():
        save_csv(output, name, frame)
    return result


def success_check(tables, p):
    metrics, tails = tables['metrics.csv'], tables['tail_precision.csv']
    limits = p['success']
    rows = []
    for r in range(3):
        for scope in ['all50', 'primary45']:
            a = metrics.query("repeat == @r and scope == @scope and method == 'new_selected'").iloc[0]
            b = metrics.query("repeat == @r and scope == @scope and method == 'original_median'").iloc[0]
            row = dict(repeat=r, scope=scope, mae_ok=bool(a.raw_rating_mae <= b.raw_rating_mae+limits['maximum_mae_increase']),
                recall_ok=bool(min(a.raw_recall_5, a.raw_recall_6) >= limits['minimum_individual_recall_each5_6']
                    and min(a.panel_recall_5, a.panel_recall_6) >= limits['minimum_panel_recall_each5_6']),
                fpr_ok=bool(max(a.raw_high_fpr, a.panel_high_fpr) <= limits['maximum_high_fpr']), precision_ok=True)
            if scope == 'primary45':
                t = tails.query("repeat == @r and scope == @scope and method == 'new_selected'").set_index('event')
                row['precision_ok'] = bool(t.loc['ge5', 'raw_precision'] >= limits['minimum_primary_high_precision']
                    and t.loc['6', 'raw_precision'] >= limits['minimum_primary_exact6_precision'])
            row['pass'] = all(row[k] for k in ['mae_ok', 'recall_ok', 'fpr_ok', 'precision_ok'])
            rows.append(row)
    intervals = tables['paired_intervals.csv'].query("reference == 'original_median'")
    return bool(all(r['pass'] for r in rows) and intervals.upper95.max() <= limits['maximum_mae_increase']), pd.DataFrame(rows)


def run(args):
    p = json.loads(args.protocol.read_text(encoding='utf-8'))
    bp = json.loads(BASE.read_text(encoding='utf-8'))
    d = load_inputs(args.source, args.raw, bp)
    o = args.output
    o.mkdir(parents=True, exist_ok=False); (o/'models').mkdir()
    write_json(o/'protocol.json', p)
    q, x, ids, targets = d['distribution'], d['x'], d['ids'], d['targets']
    styles = ids.tasting_category_fine.to_numpy()
    old_all = pd.read_csv(LEGACY/'oof_probabilities.csv.gz', float_precision='round_trip').query("method == 'shared_et'")
    names = ['original_median', 'training_prior', 'old_same_selector', 'new_selected', 'old_fixed10', 'new_fixed10', 'new_median']
    grades = {k: np.empty((3, *q.shape[:2]), int) for k in names}
    high_key = p.get('family', 'two_stage')
    probs = {k: np.empty((3, *q.shape)) for k in ['original_et', high_key]}
    fixed = {k: np.empty((3, len(policies(p)), *q.shape[:2]), int) for k in ['old', 'new']}
    choice_log, selection_log, stages, fits, splits, inner_files = [], [], [], [], [], []
    outer_folds = np.empty((3, len(q)), int)

    def fit(train, seed, context):
        print(f'Fit {context}: {len(train)} unique beers', flush=True)
        tick = time.perf_counter()
        model = make_high_model(p, seed).fit(x[train], q[train])
        np.testing.assert_allclose(model.imputer.statistics_, np.nanmedian(x[train], axis=0), atol=0, rtol=0)
        stages.extend(model.stage_records(context, targets))
        fits.append(dict(context=context, seed=seed, n_beers=len(train), seconds=time.perf_counter()-tick))
        joblib.dump(model, o/'models'/f'{context}.joblib', compress=3)
        return model

    def choose_context(train, inner_seed, seed, context):
        path = LEGACY/f'inner_{context}.csv.gz'
        old = pd.read_csv(path, float_precision='round_trip').query("method == 'shared_et'")
        old = from_frame(old, ids.beer.iloc[train], targets)
        original = enforce_hierarchy(median_grades(old), d['parents'])
        pred = {name: {i: np.empty(original.shape, int) for i in range(24)} for name in ['old', 'new']}
        new = np.empty_like(old)
        for f, (it, iv) in enumerate(make_splits(styles[train], 2, inner_seed)):
            for role, pos in [('train', train[it]), ('validation', train[iv])]:
                splits.extend(dict(context=context, fold=f, role=role, beer=int(ids.beer.iloc[z])) for z in pos)
            model = fit(train[it], seed+f, f'{context}_inner{f}')
            new[iv] = model.predict_distribution(x[train[iv]], old[iv])
            prior = q[train[it]].mean((0, 1))
            for name, probability in [('old', old[iv]), ('new', new[iv])]:
                for i, policy in enumerate(policies(p)):
                    pred[name][i][iv] = decide(probability, original[iv], prior, *policy, d['parents'])
        selections = {}
        for name in ['old', 'new']:
            selected, frame = choose(pred[name], q[train], original, d['primary'], p)
            selections[name] = selected
            selection_log.append(frame.assign(context=context, source=name))
            choice_log.append(dict(context=context, source=name, policy=selected, fallback=selected is None))
        file = o/f'inner_{context}.csv.gz'
        probability_frame(new, ids.beer.iloc[train], targets).to_csv(file, index=False, compression={'method': 'gzip', 'mtime': 0})
        inner_files.append(path)
        return selections

    for r, seed in enumerate(p['repeat_outer_seeds']):
        probs['original_et'][r] = from_frame(old_all.loc[old_all.repeat.eq(r)], ids.beer, targets)
        original = enforce_hierarchy(median_grades(probs['original_et'][r]), d['parents'])
        grades['original_median'][r] = original
        for f, (train, test) in enumerate(make_splits(styles, 3, seed)):
            context = f'r{r}_f{f}'
            mseed = p['seed']+r*1000+f*100
            selected = choose_context(train, seed+100+f, mseed, context)
            model = fit(train, mseed+99, context+'_final')
            probs[high_key][r, test] = model.predict_distribution(x[test], probs['original_et'][r, test])
            prior = q[train].mean((0, 1))
            for name, probability in [('old', probs['original_et'][r, test]), ('new', probs[high_key][r, test])]:
                for i, policy in enumerate(policies(p)):
                    fixed[name][r, i, test] = decide(probability, original[test], prior, *policy, d['parents'])
                chosen = selected[name]
                grades['old_same_selector' if name == 'old' else 'new_selected'][r, test] = original[test] if chosen is None else fixed[name][r, chosen, test]
            grades['old_fixed10'][r, test] = fixed['old'][r, 10, test]
            grades['new_fixed10'][r, test] = fixed['new'][r, 10, test]
            grades['new_median'][r, test] = enforce_hierarchy(median_grades(probs[high_key][r, test]), d['parents'])
            grades['training_prior'][r, test] = enforce_hierarchy(median_grades(np.broadcast_to(q[train].mean(0), (len(test), *q.shape[1:]))), d['parents'])
            outer_folds[r, test] = f
            print(f'Completed {context}; training-only policies {selected}', flush=True)
    tables = summarize(d, p, grades, probs, o)
    success, gate_rows = success_check(tables, p)
    save_csv(o, 'success_checks.csv', gate_rows)
    save_csv(o, 'fixed_policy_metrics.csv', [dict(repeat=r, source=name, policy=i, **row)
        for r in range(3) for name in ['old', 'new'] for i in range(24)
        for row in scoped_metrics(q, fixed[name][r, i], d['primary'])])
    full_selection = choose_context(np.arange(len(q)), p['seed']+103, p['seed']+3000, 'full_training')
    high = fit(np.arange(len(q)), p['seed']+999, 'full_training_final')
    baseline = fit_shared(d, np.arange(len(q)), bp, bp['model_seed']+999)
    selected = full_selection['new']
    model = ExperimentalPredictor(baseline, high, q.mean((0, 1)), None if selected is None else policies(p)[selected], d['parents'])
    bundle = dict(kind='independent_two_stage_not_approved', model=model, features=d['features'], targets=targets,
        primary=d['primary'], scorecard_scope=d['scope'].to_dict('records'), source_sha256=d['manifest']['sha256'],
        protocol=p, base_protocol=bp, integration_approved=False, experiment_success=success)
    joblib.dump(bundle, o/'experiment.joblib', compress=3)
    pd.concat([probability_frame(v[r], ids.beer, targets, repeat=r, method=name)
        for r in range(3) for name, v in probs.items()]).to_csv(o/'oof_probabilities.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.concat([pd.DataFrame(dict(repeat=r, method=name, beer=np.repeat(ids.beer, len(targets)), target=np.tile(targets, len(q)),
        predicted_grade=g[r].ravel())) for r in range(3) for name, g in grades.items()]).to_csv(o/'oof_grades.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    save_csv(o, 'stage_training.csv', stages); save_csv(o, 'fit_log.csv', fits)
    save_csv(o, 'inner_selection.csv', pd.concat(selection_log)); save_csv(o, 'selected_policies.csv', choice_log)
    save_csv(o, 'splits.csv', splits)
    save_csv(o, 'outer_folds.csv', [dict(repeat=r, beer=int(b), fold=int(outer_folds[r, i])) for r in range(3) for i, b in enumerate(ids.beer)])
    write_json(o/'summary.json', dict(status='experiment_completed', experiment_success=success, integrated=False,
        protocol_sha256=content_hash(p), n_beers=len(q), n_repeated_predictions=750, raw_source_sha256=d['manifest']['sha256'],
        source_sha256={name: sha256(args.source/name) for name in ['1.csv', '3.xlsx', '4.csv', '8.xlsx']},
        parent_evidence_sha256={str(path.relative_to(ROOT)): sha256(path) for path in [LEGACY/'oof_probabilities.csv.gz', *inner_files]},
        model_sha256=sha256(o/'experiment.joblib'), q3_started=False))
    print(tables['repeat_summary.csv'].query("method in ['original_median','old_same_selector','new_selected'] and metric in ['raw_rating_mae','raw_recall_5','raw_recall_6','raw_high_fpr']").to_string(index=False), flush=True)
    print(f'EXPERIMENT_SUCCESS={success}; INTEGRATED=False', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--output', type=Path, default=HERE/'results')
    with threadpool_limits(limits=1):
        run(parser.parse_args())
