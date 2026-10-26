"""Rebuild nested decisions and metrics; replay nine outer models and independently refit one."""
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
from beer_audit.audit import write_json, sha256
from beer_models.baseline import make_splits
from beer_models.raw_panel import enforce_hierarchy, median_grades, predict_raw_frame
from beer_models.target_panel import select_targets, distributions, adoption_decision
from beer_models.tail_risk import risk_frame
from r2_raw_q1 import load_inputs, RAW_MANIFEST, content_hash
from q1_target_experiment import PROTOCOL, BASE, LEGACY, from_frame, derive, fit_models


def verify(args):
    o = args.output
    p = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    bp = json.loads(BASE.read_text(encoding='utf-8'))
    summary = json.loads((o/'summary.json').read_text(encoding='utf-8'))
    assert summary['protocol_sha256'] == content_hash(p)
    assert json.loads((o/'protocol.json').read_text(encoding='utf-8')) == p
    data = load_inputs(args.source, args.raw, bp)
    x, q, ids, targets = data['x'], data['distribution'], data['ids'], data['targets']
    assert sha256(args.raw) == summary['raw_source_sha256']
    audit = json.loads((o/'data_audit.json').read_text(encoding='utf-8'))
    for name, digest in audit['source_sha256'].items():
        assert sha256(args.source/name) == digest
    for name, digest in summary['risk_evidence_sha256'].items():
        assert sha256(LEGACY/name) == digest
    assert sha256(o/'model.joblib') == summary['model_sha256']
    read = lambda name: pd.read_csv(o/name, float_precision='round_trip')
    probabilities = {k: np.empty((len(p['repeat_outer_seeds']), *q.shape)) for k in p['methods']+['inner_selected']}
    grades = {k: np.empty((len(p['repeat_outer_seeds']), *q.shape[:2]), int) for k in probabilities}
    observed = read('oof_probabilities.csv.gz')
    grade_table = read('oof_grades.csv.gz')
    assert not observed.duplicated(['repeat', 'method', 'beer', 'target']).any()
    assert len(observed) == len(q)*len(targets)*len(probabilities)*len(p['repeat_outer_seeds'])
    splits, choices = read('inner_splits.csv'), read('inner_selection.csv')
    outer = read('outer_folds.csv')
    replays = 0
    styles = ids.tasting_category_fine.to_numpy()

    def verify_inner(context, train, seed):
        actual = splits.loc[splits.context.eq(context)]
        for fold, (it, iv) in enumerate(make_splits(styles[train], p['inner_folds'], seed)):
            for role, pos in [('train', train[it]), ('validation', train[iv])]:
                saved = actual.loc[actual.inner_fold.eq(fold)&actual.role.eq(role), 'beer'].to_numpy()
                np.testing.assert_array_equal(saved, ids.beer.iloc[pos])
        table = read(f'inner_{context}.csv.gz')
        assert len(table) == len(train)*len(targets)*len(p['methods'])
        assert not table.duplicated(['method', 'beer', 'target']).any()
        probs = {k: from_frame(part, ids.beer.iloc[train], targets) for k, part in table.groupby('method')}
        for value in probs.values():
            np.testing.assert_allclose(value.sum(2), 1, atol=1e-12)
            assert (value >= 0).all()
        # Verify train-only prior separately for every inner fold.
        for _, (it, iv) in enumerate(make_splits(styles[train], p['inner_folds'], seed)):
            np.testing.assert_allclose(probs['training_prior'][iv],
                np.broadcast_to(q[train[it]].mean(0), probs['training_prior'][iv].shape), atol=1e-12, rtol=0)
        selected, rows = select_targets(probs, q[train], data['parents'])
        expected = pd.DataFrame([dict(context=context, target=targets[r.pop('target_index')], **r) for r in rows])
        pd.testing.assert_frame_equal(choices.loc[choices.context.eq(context)].reset_index(drop=True),
            expected, check_dtype=False, atol=1e-12, rtol=0)
        return selected

    for repeat, seed in enumerate(p['repeat_outer_seeds']):
        for fold, (train, test) in enumerate(make_splits(styles, p['outer_folds'], seed)):
            context = f'r{repeat}_f{fold}'
            np.testing.assert_array_equal(outer.query('repeat == @repeat').set_index('beer').loc[ids.beer.iloc[test], 'outer_fold'], fold)
            selected = verify_inner(context, train, seed+100+fold)
            model = joblib.load(o/'models'/f'{context}.joblib')
            assert selected == model.choices
            for m in model.models.values():
                np.testing.assert_allclose(m.imputer.statistics_, np.nanmedian(x[train], axis=0), atol=0, rtol=0)
            values = dict(training_prior=np.broadcast_to(q[train].mean(0), (len(test), *q.shape[1:])),
                **{k: distributions(m, x[test]) for k, m in model.models.items()},
                inner_selected=model.predict_distribution(x[test]))
            for name, value in values.items():
                saved = from_frame(observed.query('repeat == @repeat and method == @name'), ids.beer, targets)
                np.testing.assert_allclose(value, saved[test], atol=1e-12, rtol=0)
                assert (saved >= 0).all()
                np.testing.assert_allclose(saved.sum(2), 1, atol=1e-12, rtol=0)
                probabilities[name][repeat, test] = value
                grades[name][repeat, test] = enforce_hierarchy(median_grades(value), data['parents'])
            replays += 1
            if repeat == 0 and fold == 0:
                print('Independently refitting r0_f0: shared forest + 100 target forests', flush=True)
                fresh = fit_models(data, train, p, bp, p['seed']+99)
                for name, m in fresh.items():
                    np.testing.assert_allclose(distributions(m, x[test]), values[name], atol=1e-12, rtol=0)
                del fresh
    for repeat in range(len(p['repeat_outer_seeds'])):
        for name in grades:
            saved = grade_table.query('repeat == @repeat and method == @name').pivot(index='beer', columns='target', values='predicted_grade').loc[ids.beer, targets]
            np.testing.assert_array_equal(saved, grades[name][repeat])
    tables = derive(data, p, probabilities, grades)
    for name, expected in tables.items():
        expected = pd.read_csv(io.StringIO(expected.to_csv(index=False)), float_precision='round_trip')
        pd.testing.assert_frame_equal(read(name), expected, check_dtype=False, atol=1e-12, rtol=0, obj=name)
    # Exact historical baseline control on the same seeds, not a new score definition.
    historical = pd.read_csv(ROOT/'outputs/task5_repeated_nested_20261026/q1_metrics_by_repeat.csv').query("method == 'extra_trees'")
    current = tables['metrics_by_repeat.csv'].query("method == 'shared_et'")
    for column in ['raw_rating_mae', 'raw_recall_5', 'raw_recall_6', 'panel_median_mae']:
        np.testing.assert_allclose(current[column], historical[column], atol=1e-12, rtol=0)
    gain = tables['paired_intervals.csv'].query("method == 'inner_selected'").iloc[0]
    adopted, tail_solved = adoption_decision(tables['metrics_by_repeat.csv'], gain.mae_gain, gain.lower95, p)
    assert adopted == summary['adopted'] and tail_solved == summary['tail_solved']
    full_choices = verify_inner('full_training', np.arange(len(q)), p['seed']+103)
    bundle = joblib.load(o/'model.joblib')
    assert bundle['raw_panel_model'].choices == (full_choices if adopted else ['shared_et']*len(targets))
    frame = pd.read_csv(args.source/'1.csv', dtype=str)
    long, _, wide, _ = predict_raw_frame(frame, bundle)
    actual_risk = risk_frame(frame, bundle, ids.beer.to_numpy())
    old = pd.read_csv(LEGACY/'full_fit_scores_all52.csv').dropna(subset=['probability_rating_ge1'])
    keys = ['beer', 'target']
    columns = [c for c in actual_risk if 'probability' in c]
    drift = []
    for column in columns:
        a = actual_risk.set_index(keys)[column].sort_index()
        b = old.set_index(keys)[column].sort_index()
        # The historical ordinal fit has an effectively-zero coriander6 event.
        # Refitting its near-single-class logit at tol=1e-6 changes ~1e-8 risks;
        # keep a narrow declared numerical exception, not a quality gate change.
        exception = (a.index.get_level_values('target') == 'coriander') if column == 'uncalibrated_probability_rating_6' else np.zeros(len(a), bool)
        np.testing.assert_allclose(a[~exception], b[~exception], atol=1e-10, rtol=0)
        if exception.any():
            assert (q[:, targets.index('coriander'), 6] == 0).all()
            assert a[exception].max() < 1e-6 and b[exception].max() < 1e-6
            np.testing.assert_allclose(a[exception], b[exception], atol=1e-7, rtol=0)
        for target in targets:
            mask = a.index.get_level_values('target') == target
            drift.append(dict(target=target, field=column, maximum_absolute_difference=float(abs(a[mask]-b[mask]).max()),
                              near_zero_optimizer_exception=bool(exception[mask].any())))
    from beer_audit.audit import save_csv
    save_csv(o, 'risk_rebuild_comparison.csv', drift)
    # Check the actual CLI export separately from direct model inference.
    smoke = pd.read_csv(o/'prediction_smoke/scores_all52_long.csv')
    long['beer'] = long.beer.astype(int)
    smoke_info = json.loads((o/'prediction_smoke/prediction_info.json').read_text(encoding='utf-8'))
    assert smoke_info['model_sha256'] == summary['model_sha256']
    assert smoke_info['input_sha256'] == sha256(args.source/'1.csv')
    assert smoke_info['grade_method'] == 'q1_target_distribution_median'
    assert smoke_info['risk_provenance'] == bundle['risk_provenance']
    pd.testing.assert_frame_equal(smoke[keys+['estimated_grade_0_6']].set_index(keys).sort_index(),
        long[keys+['estimated_grade_0_6']].set_index(keys).sort_index(), check_dtype=False)
    for column in columns:
        np.testing.assert_allclose(smoke.dropna(subset=[column]).set_index(keys)[column].sort_index(),
            actual_risk.set_index(keys)[column].sort_index(), atol=1e-12, rtol=0)
    assert len(smoke) == 13000 and smoke.estimated_grade_0_6.isna().sum() == 500
    main = read('estimated_grades_all52.csv').set_index('beer')
    deployed = 'inner_selected' if adopted else 'shared_et'
    np.testing.assert_array_equal(main.loc[ids.beer, targets], grades[deployed][0])
    assert main[bp['unlabelled_card_targets']].isna().all().all()
    report = dict(status='passed', nested_contexts_reconstructed=10, outer_saved_model_replays=replays,
        fresh_refit_outer_contexts=1, fresh_refit_shared_forests=1, fresh_refit_target_forests=100,
        independently_refit_inner_models=False, probability_metric_tables_recomputed=True,
        r2b_full_fit_risk_values_compared=len(actual_risk)*len(columns), cli_rows=13000,
        risk_workflow_preserved=True, risk_snapshot_bitwise_identical=False,
        maximum_risk_rebuild_difference=max(r['maximum_absolute_difference'] for r in drift),
        numerical_exception='uncalibrated coriander6 only, no observed6 labels, probabilities below1e-6, drift below1e-7; all other targets/events at1e-10',
        unlabelled_blank_rows=500, original_baseline_repeat_control=True,
        model_sha256=summary['model_sha256'], adopted=adopted, tail_solved=tail_solved,
        scope='Numerical, grouping, decision and inference verification; scientific gains are in repeated nested results, not test counts')
    write_json(o/'verification.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--output', type=Path, required=True)
    with threadpool_limits(limits=1):
        verify(parser.parse_args())
