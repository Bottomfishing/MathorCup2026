"""Verify the isolated experiment, saved fits, decisions, numerical tables and CLI."""
import argparse
import io
import json
import sys
import tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'src'), str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import write_json, sha256
from beer_models.baseline import make_splits
from beer_models.raw_panel import median_grades, enforce_hierarchy
from r2_raw_q1 import load_inputs, RAW_MANIFEST, content_hash
from q1_target_experiment import BASE, from_frame
from two_stage import make_high_model, decide
from run import LEGACY, policies, choose, summarize, success_check, scoped_metrics


def fixed_policy_table(d, p, probabilities):
    q, styles = d['distribution'], d['ids'].tasting_category_fine.to_numpy()
    rows = []
    high_key = p.get('family', 'two_stage')
    for r, seed in enumerate(p['repeat_outer_seeds']):
        baseline = enforce_hierarchy(median_grades(probabilities['original_et'][r]), d['parents'])
        for source, key in [('old', 'original_et'), ('new', high_key)]:
            for i, policy in enumerate(policies(p)):
                grade = np.empty(q.shape[:2], int)
                for train, test in make_splits(styles, 3, seed):
                    grade[test] = decide(probabilities[key][r, test], baseline[test], q[train].mean((0, 1)), *policy, d['parents'])
                rows.extend(dict(repeat=r, source=source, policy=i, **a) for a in scoped_metrics(q, grade, d['primary']))
    return pd.DataFrame(rows)


def verify(args):
    o = args.output
    p = json.loads(args.protocol.read_text(encoding='utf-8'))
    s = json.loads((o/'summary.json').read_text(encoding='utf-8'))
    assert s['protocol_sha256'] == content_hash(p)
    assert json.loads((o/'protocol.json').read_text(encoding='utf-8')) == p
    assert not s['integrated'] and not s['q3_started']
    for name, digest in s['source_sha256'].items():
        assert sha256(args.source/name) == digest
    for name, digest in s['parent_evidence_sha256'].items():
        assert sha256(ROOT/name) == digest
    assert sha256(args.raw) == s['raw_source_sha256']
    assert sha256(o/'experiment.joblib') == s['model_sha256']
    bp = json.loads(BASE.read_text(encoding='utf-8'))
    d = load_inputs(args.source, args.raw, bp)
    q, x, ids, targets = d['distribution'], d['x'], d['ids'], d['targets']
    styles = ids.tasting_category_fine.to_numpy()
    read = lambda name: pd.read_csv(o/name, float_precision='round_trip')
    saved = read('oof_probabilities.csv.gz'); old = saved.query("method == 'original_et'")
    high_key = p.get('family', 'two_stage')
    new = saved.loc[saved.method.eq(high_key)]
    original = {r: from_frame(f, ids.beer, targets) for r, f in old.groupby('repeat')}
    probabilities = {'original_et': np.stack([original[r] for r in range(3)]),
        high_key: np.stack([from_frame(new.loc[new.repeat.eq(r)], ids.beer, targets) for r in range(3)])}
    gt = read('oof_grades.csv.gz')
    grades = {name: np.stack([part.loc[part.repeat.eq(r)].pivot(index='beer', columns='target', values='predicted_grade').loc[ids.beer, targets].to_numpy(int)
        for r in range(3)]) for name, part in gt.groupby('method', sort=False)}
    for a in probabilities.values():
        assert np.isfinite(a).all() and (a >= 0).all()
        np.testing.assert_allclose(a.sum(3), 1, atol=1e-12, rtol=0)
    choices, logs, split_log = [], [], []
    model_replays = []

    def check_context(train, seed, model_seed, context):
        inner_old = pd.read_csv(LEGACY/f'inner_{context}.csv.gz', float_precision='round_trip').query("method == 'shared_et'")
        b = from_frame(inner_old, ids.beer.iloc[train], targets)
        a = from_frame(read(f'inner_{context}.csv.gz'), ids.beer.iloc[train], targets)
        base_grade = enforce_hierarchy(median_grades(b), d['parents'])
        preds = {name: {i: np.empty(base_grade.shape, int) for i in range(24)} for name in ['old', 'new']}
        for f, (it, iv) in enumerate(make_splits(styles[train], 2, seed)):
            for role, positions in [('train', train[it]), ('validation', train[iv])]:
                split_log.extend(dict(context=context, fold=f, role=role, beer=int(ids.beer.iloc[z])) for z in positions)
            m = joblib.load(o/'models'/f'{context}_inner{f}.joblib')
            assert m.seed == model_seed+f and m.protocol == p
            np.testing.assert_allclose(m.imputer.statistics_, np.nanmedian(x[train[it]], axis=0), atol=0, rtol=0)
            np.testing.assert_allclose(m.predict_distribution(x[train[iv]], b[iv]), a[iv], atol=1e-12, rtol=0)
            model_replays.append(f'{context}_inner{f}')
            prior = q[train[it]].mean((0, 1))
            for name, prob in [('old', b[iv]), ('new', a[iv])]:
                for i, policy in enumerate(policies(p)):
                    preds[name][i][iv] = decide(prob, base_grade[iv], prior, *policy, d['parents'])
        selected = {}
        for name in ['old', 'new']:
            selected[name], frame = choose(preds[name], q[train], base_grade, d['primary'], p)
            logs.append(frame.assign(context=context, source=name))
            choices.append(dict(context=context, source=name, policy=selected[name], fallback=selected[name] is None))
        return selected

    for r, seed in enumerate(p['repeat_outer_seeds']):
        for f, (train, test) in enumerate(make_splits(styles, 3, seed)):
            context = f'r{r}_f{f}'
            mseed = p['seed']+r*1000+f*100
            selected = check_context(train, seed+100+f, mseed, context)
            model = joblib.load(o/'models'/f'{context}_final.joblib')
            np.testing.assert_allclose(model.imputer.statistics_, np.nanmedian(x[train], axis=0), atol=0, rtol=0)
            actual = model.predict_distribution(x[test], original[r][test])
            np.testing.assert_allclose(actual, probabilities[high_key][r, test], atol=1e-12, rtol=0)
            model_replays.append(context+'_final')
            if r == 0 and f == 0:
                print('Independently refitting one complete two-stage outer model', flush=True)
                fresh = make_high_model(p, mseed+99).fit(x[train], q[train])
                np.testing.assert_allclose(fresh.predict_distribution(x[test], original[r][test]), actual, atol=1e-12, rtol=0)
            base = enforce_hierarchy(median_grades(original[r][test]), d['parents'])
            prior = q[train].mean((0, 1))
            for source, name, probability in [('old', 'old_same_selector', original[r][test]), ('new', 'new_selected', actual)]:
                policy = selected[source]
                expected = base if policy is None else decide(probability, base, prior, *policies(p)[policy], d['parents'])
                np.testing.assert_array_equal(expected, grades[name][r, test])
            for name, probability in [('old_fixed10', original[r][test]), ('new_fixed10', actual)]:
                np.testing.assert_array_equal(decide(probability, base, prior, *policies(p)[10], d['parents']), grades[name][r, test])
            np.testing.assert_array_equal(enforce_hierarchy(median_grades(actual), d['parents']), grades['new_median'][r, test])
        print(f'Completed repeat {r} saved-model and decision replay', flush=True)
    full = check_context(np.arange(len(q)), p['seed']+103, p['seed']+3000, 'full_training')

    def compare(name, expected):
        expected = pd.read_csv(io.StringIO(pd.DataFrame(expected).to_csv(index=False)), float_precision='round_trip')
        pd.testing.assert_frame_equal(read(name), expected, check_dtype=False, atol=1e-12, rtol=0, obj=name)
    compare('inner_selection.csv', pd.concat(logs)); compare('selected_policies.csv', choices); compare('splits.csv', split_log)
    compare('fixed_policy_metrics.csv', fixed_policy_table(d, p, probabilities))
    with tempfile.TemporaryDirectory() as temp:
        tables = summarize(d, p, grades, probabilities, Path(temp))
    for name, expected in tables.items():
        compare(name, expected)
    passed, checks = success_check(tables, p)
    compare('success_checks.csv', checks)
    assert passed == s['experiment_success']
    bundle = joblib.load(o/'experiment.joblib')
    assert not bundle['integration_approved'] and bundle['experiment_success'] == passed
    assert bundle['model'].policy == (None if full['new'] is None else policies(p)[full['new']])
    smoke = pd.read_csv(o/'prediction_smoke/scores52.csv')
    predicted = smoke.pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[ids.beer, targets]
    np.testing.assert_array_equal(predicted, bundle['model'].predict(x)['grade'])
    assert len(smoke) == 13000 and smoke.estimated_grade_0_6.isna().sum() == 500
    assert not smoke.safety_certified.any()
    assert smoke.query("target in ['A_acetaldehyde','F_acetaldehyde']").experimental_probability_6.isna().all()
    info = json.loads((o/'prediction_smoke/info.json').read_text(encoding='utf-8'))
    assert not info['official_integration'] and info['model_sha256'] == s['model_sha256']
    report = dict(status='passed', saved_model_replays=len(model_replays), fresh_refit_outer_contexts=1,
        independently_refit_inner_models=False, reconstructed_selection_contexts=10,
        policies_per_source=24, grouped_splits_verified=True, cli_rows=13000, unlabelled_blank_rows=500,
        experiment_success=passed, integrated=False, model_sha256=s['model_sha256'])
    write_json(o/'verification.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--output', type=Path, default=HERE/'results')
    with threadpool_limits(limits=1):
        verify(parser.parse_args())
