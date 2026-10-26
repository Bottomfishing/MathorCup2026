"""Training-only per-target confidence filter on an already trained forest."""
import argparse
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'src'), str(ROOT/'scripts')]
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.baseline import make_splits
from beer_models.raw_panel import median_grades, enforce_hierarchy
from beer_models.tail_risk import base_probabilities
from r2_raw_q1 import load_inputs, RAW_MANIFEST, content_hash
from q1_target_experiment import BASE, from_frame
from run import LEGACY, policies, summarize, success_check
from two_stage import decide, apply_gates, ConfidentPredictor


def fit_gates(q, proposed):
    high, six = proposed >= 5, proposed == 6
    nh, ns = high.sum(0), six.sum(0)
    ph = np.divide((q[:, :, 5:].sum(2)*high).sum(0), nh, out=np.zeros(len(nh)), where=nh > 0)
    ps = np.divide((q[:, :, 6]*six).sum(0), ns, out=np.zeros(len(ns)), where=ns > 0)
    return dict(high=(nh >= 3)&(ph >= .5), six=(ns >= 3)&(ps >= .25),
                n_high=nh, n_six=ns, precision_high=ph, precision_six=ps)


def run(args):
    parent = args.parent
    o = args.output
    o.mkdir(parents=True, exist_ok=False)
    cp = json.loads((HERE/'protocol_confidence.json').read_text(encoding='utf-8'))
    p = json.loads((parent/'protocol.json').read_text(encoding='utf-8'))
    write_json(o/'protocol.json', cp)
    d = load_inputs(args.source, args.raw, json.loads(BASE.read_text(encoding='utf-8')))
    q, ids, targets = d['distribution'], d['ids'], d['targets']
    styles = ids.tasting_category_fine.to_numpy()
    choices = pd.read_csv(parent/'selected_policies.csv')
    old_table = pd.read_csv(parent/'oof_probabilities.csv.gz', float_precision='round_trip')
    probs = {name: np.stack([from_frame(part.loc[part.repeat.eq(r)], ids.beer, targets) for r in range(3)])
        for name, part in old_table.groupby('method', sort=False)}
    gt = pd.read_csv(parent/'oof_grades.csv.gz')
    grades = {name: np.stack([part.loc[part.repeat.eq(r)].pivot(index='beer', columns='target', values='predicted_grade').loc[ids.beer, targets].to_numpy(int)
        for r in range(3)]) for name, part in gt.groupby('method', sort=False)}
    grades['old_ungated'], grades['new_ungated'] = grades['old_same_selector'].copy(), grades['new_selected'].copy()
    logs = []

    def context_gates(context, train, seed):
        old = from_frame(pd.read_csv(LEGACY/f'inner_{context}.csv.gz', float_precision='round_trip').query("method == 'shared_et'"), ids.beer.iloc[train], targets)
        new = from_frame(pd.read_csv(parent/f'inner_{context}.csv.gz', float_precision='round_trip'), ids.beer.iloc[train], targets)
        original = enforce_hierarchy(median_grades(old), d['parents'])
        chosen = choices.loc[choices.context.eq(context)].set_index('source').policy
        fitted = {}
        for source, values in [('old', old), ('new', new)]:
            proposal = original.copy()
            if not pd.isna(chosen[source]):
                for it, iv in make_splits(styles[train], 2, seed):
                    proposal[iv] = decide(values[iv], original[iv], q[train[it]].mean((0, 1)), *policies(p)[int(chosen[source])], [])
            fitted[source] = fit_gates(q[train], proposal)
            for j, target in enumerate(targets):
                logs.append(dict(context=context, source=source, target=target, **{k: np.asarray(v)[j].item() for k, v in fitted[source].items()}))
        return fitted, chosen

    for r, seed in enumerate(p['repeat_outer_seeds']):
        for f, (train, test) in enumerate(make_splits(styles, 3, seed)):
            context = f'r{r}_f{f}'
            gates, chosen = context_gates(context, train, seed+100+f)
            original = grades['original_median'][r, test]
            for source, key, method in [('old', 'original_et', 'old_same_selector'), ('new', 'joint_tail_forest', 'new_selected')]:
                proposed = original if pd.isna(chosen[source]) else decide(probs[key][r, test], original,
                    q[train].mean((0, 1)), *policies(p)[int(chosen[source])], [])
                grades[method][r, test] = apply_gates(original, proposed, gates[source], d['parents'])
    tables = summarize(d, p, grades, probs, o)
    success, checks = success_check(tables, p)
    save_csv(o, 'success_checks.csv', checks)
    gates, _ = context_gates('full_training', np.arange(len(q)), p['seed']+103)
    save_csv(o, 'gates.csv', logs)
    bundle = joblib.load(parent/'experiment.joblib')
    bundle['model'] = ConfidentPredictor(bundle['model'], gates['new'])
    bundle.update(confidence_protocol=cp, experiment_success=success, integration_approved=False)
    joblib.dump(bundle, o/'experiment.joblib', compress=3)
    pd.concat([pd.DataFrame(dict(repeat=r, method=name, beer=np.repeat(ids.beer, len(targets)), target=np.tile(targets, len(q)),
        predicted_grade=matrix[r].ravel())) for r in range(3) for name, matrix in grades.items()]).to_csv(o/'oof_grades.csv.gz', index=False,
            compression={'method': 'gzip', 'mtime': 0})
    write_json(o/'summary.json', dict(status='experiment_completed', experiment_success=success, integrated=False,
        protocol_sha256=content_hash(cp), parent_model_sha256=sha256(parent/'experiment.joblib'), model_sha256=sha256(o/'experiment.joblib'),
        independent_beers=250, repeated_predictions=750, probability_models_refit=0, q3_started=False))
    print(tables['repeat_summary.csv'].query("method in ['old_same_selector','new_selected'] and metric in ['raw_rating_mae','raw_recall_5','raw_recall_6','raw_high_fpr']").to_string(index=False), flush=True)
    print('SUCCESS', success, 'INTEGRATED False', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--parent', type=Path, default=HERE/'forest_results')
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--output', type=Path, default=HERE/'confidence_results')
    with threadpool_limits(limits=1):
        run(parser.parse_args())
