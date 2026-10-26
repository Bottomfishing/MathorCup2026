"""Reconstruct every confidence gate from training-only labels; no model refits."""
import argparse
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
from beer_audit.audit import write_json, sha256
from beer_models.baseline import make_splits
from beer_models.raw_panel import median_grades, enforce_hierarchy
from r2_raw_q1 import load_inputs, RAW_MANIFEST, content_hash
from q1_target_experiment import BASE, from_frame
from run import LEGACY, policies, summarize, success_check
from two_stage import decide


def verify(args):
    parent, o = args.parent, args.output
    cp = json.loads((HERE/'protocol_confidence.json').read_text(encoding='utf-8'))
    p = json.loads((parent/'protocol.json').read_text(encoding='utf-8'))
    s = json.loads((o/'summary.json').read_text(encoding='utf-8'))
    assert s['protocol_sha256'] == content_hash(cp)
    assert sha256(parent/'experiment.joblib') == s['parent_model_sha256']
    assert sha256(o/'experiment.joblib') == s['model_sha256']
    d = load_inputs(args.source, args.raw, json.loads(BASE.read_text(encoding='utf-8')))
    q, ids, targets = d['distribution'], d['ids'], d['targets']
    styles = ids.tasting_category_fine.to_numpy()
    choices = pd.read_csv(parent/'selected_policies.csv')
    saved_gates = pd.read_csv(o/'gates.csv', float_precision='round_trip')
    outer = pd.read_csv(parent/'oof_probabilities.csv.gz', float_precision='round_trip')
    probs = {name: np.stack([from_frame(part.loc[part.repeat.eq(r)], ids.beer, targets) for r in range(3)])
        for name, part in outer.groupby('method', sort=False)}
    gt = pd.read_csv(o/'oof_grades.csv.gz')
    grades = {name: np.stack([part.loc[part.repeat.eq(r)].pivot(index='beer', columns='target', values='predicted_grade').loc[ids.beer, targets].to_numpy(int)
        for r in range(3)]) for name, part in gt.groupby('method', sort=False)}

    def check_gates(context, train, seed):
        old = from_frame(pd.read_csv(LEGACY/f'inner_{context}.csv.gz', float_precision='round_trip').query("method == 'shared_et'"), ids.beer.iloc[train], targets)
        new = from_frame(pd.read_csv(parent/f'inner_{context}.csv.gz', float_precision='round_trip'), ids.beer.iloc[train], targets)
        original = enforce_hierarchy(median_grades(old), d['parents'])
        chosen = choices.loc[choices.context.eq(context)].set_index('source').policy
        verified = {}
        for source, prob in [('old', old), ('new', new)]:
            proposed = original.copy()
            if not pd.isna(chosen[source]):
                for it, iv in make_splits(styles[train], 2, seed):
                    proposed[iv] = decide(prob[iv], original[iv], q[train[it]].mean((0, 1)), *policies(p)[int(chosen[source])], [])
            table = saved_gates.loc[saved_gates.context.eq(context)&saved_gates.source.eq(source)].set_index('target').loc[targets]
            for j, target in enumerate(targets):
                h = proposed[:, j] >= 5; six = proposed[:, j] == 6
                ph = q[train, j, 5:].sum(1)[h].mean() if h.any() else 0.
                ps = q[train, j, 6][six].mean() if six.any() else 0.
                row = table.loc[target]
                assert row.n_high == h.sum() and row.n_six == six.sum()
                np.testing.assert_allclose([row.precision_high, row.precision_six], [ph, ps], atol=1e-12)
                assert row.high == bool(h.sum() >= 3 and ph >= .5)
                assert row.six == bool(six.sum() >= 3 and ps >= .25)
            verified[source] = table
        return verified, chosen

    for r, seed in enumerate(p['repeat_outer_seeds']):
        for f, (train, test) in enumerate(make_splits(styles, 3, seed)):
            gates, chosen = check_gates(f'r{r}_f{f}', train, seed+100+f)
            original = grades['original_median'][r, test]
            for source, key, method in [('old', 'original_et', 'old_same_selector'), ('new', 'joint_tail_forest', 'new_selected')]:
                proposed = original if pd.isna(chosen[source]) else decide(probs[key][r, test], original, q[train].mean((0, 1)), *policies(p)[int(chosen[source])], [])
                expected = original.copy()
                for j in range(len(targets)):
                    high = (proposed[:, j] >= 5)&bool(gates[source].iloc[j].high)
                    expected[high, j] = 5
                    exact = high&(proposed[:, j] == 6)&bool(gates[source].iloc[j].six)
                    expected[exact, j] = 6
                for par, child in d['parents']:
                    expected[:, par] = np.maximum(expected[:, par], expected[:, child])
                np.testing.assert_array_equal(expected, grades[method][r, test])
    full, _ = check_gates('full_training', np.arange(len(q)), p['seed']+103)
    with tempfile.TemporaryDirectory() as temp:
        tables = summarize(d, p, grades, probs, Path(temp))
    for name, expected in tables.items():
        pd.testing.assert_frame_equal(pd.read_csv(o/name), expected.reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=0)
    passed, _ = success_check(tables, p)
    assert passed == s['experiment_success']
    bundle = joblib.load(o/'experiment.joblib')
    for key in ['high', 'six']:
        np.testing.assert_array_equal(bundle['model'].gates[key], full['new'][key])
    smoke = pd.read_csv(o/'prediction_smoke/scores52.csv')
    values = smoke.pivot(index='beer', columns='target', values='estimated_grade_0_6').loc[ids.beer, targets]
    np.testing.assert_array_equal(values, bundle['model'].predict(d['x'])['grade'])
    assert len(smoke) == 13000 and smoke.estimated_grade_0_6.isna().sum() == 500
    write_json(o/'verification.json', dict(status='passed', training_only_gate_rows_verified=len(saved_gates),
        outer_predictions_checked=37500, cli_rows=13000, missing_labels_blank=500,
        probability_models_refit=0, experiment_success=passed, integrated=False,
        model_sha256=s['model_sha256']))
    print('Confidence filtering verified; experiment success:', passed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--parent', type=Path, default=HERE/'forest_results')
    parser.add_argument('--output', type=Path, default=HERE/'confidence_results')
    verify(parser.parse_args())
