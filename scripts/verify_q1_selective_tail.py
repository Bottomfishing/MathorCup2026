"""Independent grouped probability refits, nested policy replay and missing-input checks."""
import argparse
import io
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
from beer_models.baseline import make_splits, mask_observed
from beer_models.raw_panel import median_grades, enforce_hierarchy, rating_loss, predict_raw_frame
from beer_models.tail_risk import base_probabilities, tail_metrics, feasible, risk_frame
from beer_models.selective_tail import selective_decision, select_policy, candidates
from beer_models.repeated_validation import beer_averaged_interval
from r2_raw_q1 import load_inputs, RAW_MANIFEST, content_hash, metric_tables, summary_metrics
from q1_target_experiment import BASE, from_frame, fit_shared
from q1_selective_tail_experiment import PROTOCOL
from summarize_q1_target import tail_details


def verify(args):
    o, parent = args.output, args.parent
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    bp = json.loads(BASE.read_text(encoding='utf-8'))
    s = json.loads((o/'summary.json').read_text(encoding='utf-8'))
    assert s['protocol_sha256'] == content_hash(p)
    assert json.loads((o/'protocol.json').read_text(encoding='utf-8')) == p
    for name, digest in s['parent_evidence_sha256'].items():
        assert sha256(parent/name) == digest
    assert sha256(o/'model.joblib') == s['model_sha256']
    d = load_inputs(args.source, args.raw, bp)
    q, ids, targets = d['distribution'], d['ids'], d['targets']
    styles = ids.tasting_category_fine.to_numpy()
    read = lambda name: pd.read_csv(o/name, float_precision='round_trip')
    table = pd.read_csv(parent/'oof_probabilities.csv.gz', float_precision='round_trip').query("method == 'shared_et'")
    grades = read('oof_grades.csv.gz')
    matrices = {(r, name): part.pivot(index='beer', columns='target', values='predicted_grade').loc[ids.beer, targets].to_numpy(int)
        for (r, name), part in grades.groupby(['repeat', 'method'])}
    selections, choices, prior_rows, missing = [], [], [], []

    def choose(context, train, seed):
        inner = pd.read_csv(parent/f'inner_{context}.csv.gz', float_precision='round_trip').query("method == 'shared_et'")
        prob = from_frame(inner, ids.beer.iloc[train], targets)
        preds = {i: np.empty((len(train), len(targets)), int) for i in range(len(candidates(p)))}
        for fold, (it, iv) in enumerate(make_splits(styles[train], 2, seed)):
            prior = q[train[it]].mean((0, 1))
            prior_rows.append(dict(context=context, inner_fold=fold, prior5=float(prior[5]), prior6=float(prior[6]), n_prior_training_beers=len(it)))
            for i, penalties in enumerate(candidates(p)):
                preds[i][iv] = selective_decision(prob[iv], prior, *penalties, d['parents'])
        base = enforce_hierarchy(median_grades(prob), d['parents'])
        selected, rows = select_policy(preds, q[train], base, p)
        choices.extend(dict(context=context, **r) for r in rows)
        selections.append(dict(context=context, selected_candidate=selected, has_eligible_candidate=selected is not None,
            prior5=float(q[train].mean((0, 1))[5]), prior6=float(q[train].mean((0, 1))[6])))
        return selected

    for r, seed in enumerate(p['repeat_outer_seeds']):
        prob = from_frame(table.loc[table.repeat.eq(r)], ids.beer, targets)
        for fold, (train, test) in enumerate(make_splits(styles, 3, seed)):
            selected = choose(f'r{r}_f{fold}', train, seed+100+fold)
            prior = q[train].mean((0, 1))
            for i, penalties in enumerate(candidates(p)):
                np.testing.assert_array_equal(selective_decision(prob[test], prior, *penalties, d['parents']), matrices[r, f'policy{i}'][test])
            baseline = enforce_hierarchy(median_grades(prob[test]), d['parents'])
            expected = baseline if selected is None else matrices[r, f'policy{selected}'][test]
            np.testing.assert_array_equal(matrices[r, 'shared_et'][test], baseline)
            np.testing.assert_array_equal(matrices[r, 'inner_selected'][test], expected)
            print(f'Independent shared-ET refit r{r}_f{fold} and held-out missingness replay', flush=True)
            model = fit_shared(d, train, bp, p['seed']+r*1000+fold*100+99)
            np.testing.assert_allclose(base_probabilities(model, d['x'][test]), prob[test], atol=1e-12, rtol=0)
            for missing_repeat in range(3):
                for rate in [.05, .1, .2]:
                    z, added = mask_observed(d['x'][test], rate, p['seed']+r*10000+fold*100+missing_repeat)
                    assert np.isnan(z)[np.isnan(d['x'][test])].all()
                    pp = base_probabilities(model, z)
                    gg = enforce_hierarchy(median_grades(pp), d['parents'])
                    after = dict(shared_et=gg, inner_selected=gg if selected is None else
                        selective_decision(pp, prior, *candidates(p)[selected], d['parents']))
                    for name, pred in after.items():
                        missing.append(dict(repeat=r, outer_fold=fold, missing_repeat=missing_repeat,
                            method=name, additional_missing_rate=rate, n_beers=len(test), added_missing_cells=int(added.sum()),
                            raw_rating_mae=float(rating_loss(q[test], pred).mean()),
                            changed_grade_fraction=float((pred != matrices[r, name][test]).mean())))
    full_choice = choose('full_training', np.arange(len(q)), p['seed']+103)
    assert full_choice == s['full_training_choice']

    def compare(name, expected):
        expected = pd.read_csv(io.StringIO(pd.DataFrame(expected).to_csv(index=False)), float_precision='round_trip')
        pd.testing.assert_frame_equal(read(name), expected, check_dtype=False, atol=1e-12, rtol=0, obj=name)
    compare('inner_selection.csv', choices)
    compare('training_priors.csv', prior_rows)
    compare('selected_policies.csv', selections)
    records, target_rows, level_rows, confusion, tails, panels, scopes = [], [], [], [], [], [], []
    methods = list(grades.method.drop_duplicates())
    observed = median_grades(q)
    for r in range(3):
        pred = {name: dict(grade=matrices[r, name]) for name in methods}
        tables = metric_tables(d, pred)
        scopes.append(tables['model_comparison.csv'].assign(repeat=r))
        target_rows.append(tables['metrics_by_target.csv'].assign(repeat=r))
        level_rows.append(tables['metrics_by_observed_grade.csv'].assign(repeat=r))
        confusion.append(tables['confusion.csv'].assign(repeat=r))
        for name in methods:
            g = matrices[r, name]
            records.append(dict(repeat=r, method=name, **(summary_metrics(q, g) | tail_metrics(q, g))))
            for scope, mask in [('all50', np.ones(len(targets), bool)), ('primary45', d['primary'])]:
                tails.extend(dict(repeat=r, method=name, scope=scope, **a) for a in tail_details(q[:, mask], g[:, mask]))
            for k in range(7):
                truth = observed == k
                for j in range(7):
                    n = int((truth&(g == j)).sum())
                    panels.append(dict(repeat=r, method=name, observed_grade=k, predicted_grade=j, count=n,
                        row_fraction=n/truth.sum() if truth.sum() else np.nan))
    compare('metrics_by_repeat.csv', records)
    compare('scope_comparison.csv', pd.concat(scopes))
    for name, expected in [('metrics_by_target.csv', pd.concat(target_rows)), ('metrics_by_observed_grade.csv', pd.concat(level_rows)),
        ('confusion.csv', pd.concat(confusion)), ('tail_precision.csv', tails), ('panel_confusion.csv', panels)]:
        compare(name, expected)
    delta = np.stack([rating_loss(q, matrices[r, 'inner_selected']).mean(1)-rating_loss(q, matrices[r, 'shared_et']).mean(1) for r in range(3)])
    mean, low, high = beer_averaged_interval(delta, styles, 2000, p['bootstrap_seed'], [.025, .975])
    compare('paired_mae_increase.csv', [dict(mean=mean, lower95=low, upper95=high, n_independent_beers=len(q))])
    selected_metrics = [tail_metrics(q, matrices[r, 'inner_selected']) for r in range(3)]
    eligible = [feasible(selected_metrics[r], tail_metrics(q, matrices[r, 'shared_et'])['raw_rating_mae'], p['point_limits']) for r in range(3)]
    assert eligible == s['eligible_repeats']
    assert bool(all(eligible) and s['pooled_feasible'] and high <= .02 and full_choice is not None) == s['adopted']
    bundle = joblib.load(o/'model.joblib')
    frame = pd.read_csv(args.source/'1.csv', dtype=str)
    official = bundle['raw_panel_model'].predict(d['x'])['grade']
    baseline = bundle['raw_panel_model'].model.predict(d['x'])['grade']
    if not s['adopted']:
        np.testing.assert_array_equal(official, baseline)
    candidate = bundle['experimental_point_model'].predict(d['x'])['grade']
    np.testing.assert_array_equal(candidate, selective_decision(base_probabilities(bundle['raw_panel_model'].model, d['x']),
        q.mean((0, 1)), *candidates(p)[full_choice], d['parents']))
    for folder, predicted, flag in [('prediction_smoke', official, False), ('experimental_smoke', candidate, True)]:
        smoke = pd.read_csv(o/folder/'scores_all52_long.csv')
        values = smoke.pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[ids.beer, targets]
        np.testing.assert_array_equal(values, predicted)
        assert smoke.estimated_grade_0_6.isna().sum() == 500
        info = json.loads((o/folder/'prediction_info.json').read_text(encoding='utf-8'))
        assert info['experimental_tail'] == flag and info['model_sha256'] == s['model_sha256']
        if flag:
            assert not info['official_point_replacement'] and not info['tail_point_qualified']
    parent_risk = pd.read_csv(parent/'full_fit_scores_all52.csv').set_index(['beer', 'target']).sort_index()
    current = pd.read_csv(o/'prediction_smoke/scores_all52_long.csv').set_index(['beer', 'target']).sort_index()
    alternative = pd.read_csv(o/'experimental_smoke/scores_all52_long.csv').set_index(['beer', 'target']).sort_index()
    for c in [k for k in current if 'probability' in k]:
        np.testing.assert_allclose(current[c], parent_risk[c], atol=1e-12, equal_nan=True)
        np.testing.assert_allclose(alternative[c], parent_risk[c], atol=1e-12, equal_nan=True)
    save_csv(o, 'missingness_by_fold.csv', missing)
    missing = pd.DataFrame(missing)
    save_csv(o, 'missingness_summary.csv', [dict(method=name, additional_missing_rate=rate,
        raw_rating_mae=float(np.average(part.raw_rating_mae, weights=part.n_beers)),
        changed_grade_fraction=float(np.average(part.changed_grade_fraction, weights=part.n_beers)))
        for (name, rate), part in missing.groupby(['method', 'additional_missing_rate'])])
    write_json(o/'verification.json', dict(status='passed', independently_refit_shared_outer_models=9,
        inner_decision_contexts_rebuilt=10, candidate_policies_per_context=len(candidates(p)),
        inner_probability_models_independently_refit=False, original_missingness_preserved=True,
        heldout_artificial_MCAR_masks=81, missingness_metric_rows=len(missing), cli_rows_checked=26000,
        unlabelled_blank_rows=1000, official_baseline_preserved=not s['adopted'],
        deployed_risk_probabilities_unchanged=True, adopted=s['adopted'],
        risk_comparison_reference='same-stage parent rebuild; original R2b numerical drift is reported by parent verification',
        scope_table_cross_checked=True,
        model_sha256=s['model_sha256'], scope='Post-development same-250-beer numerical and grouped validation, not external testing'))
    print('Selective-tail verification passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--parent', type=Path, default=ROOT/'outputs/q1_target_20261026')
    parser.add_argument('--output', type=Path, required=True)
    with threadpool_limits(limits=1):
        verify(parser.parse_args())
