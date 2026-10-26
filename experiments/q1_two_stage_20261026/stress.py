"""Held-out missing-input checks for the promising joint-forest candidate only."""
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
from beer_audit.audit import save_csv, write_json
from beer_models.baseline import make_splits, mask_observed
from beer_models.raw_panel import median_grades, enforce_hierarchy, rating_loss
from beer_models.tail_risk import base_probabilities
from r2_raw_q1 import load_inputs, RAW_MANIFEST
from q1_target_experiment import BASE, fit_shared
from two_stage import decide
from run import policies


def run(args):
    o = args.output
    p = json.loads((o/'protocol.json').read_text(encoding='utf-8'))
    bp = json.loads(BASE.read_text(encoding='utf-8'))
    d = load_inputs(args.source, args.raw, bp)
    q, ids, styles = d['distribution'], d['ids'], d['ids'].tasting_category_fine.to_numpy()
    choices = pd.read_csv(o/'selected_policies.csv')
    rows = []
    for r, seed in enumerate(p['repeat_outer_seeds']):
        for f, (train, test) in enumerate(make_splits(styles, 3, seed)):
            context = f'r{r}_f{f}'
            baseline = fit_shared(d, train, bp, p['seed']+r*1000+f*100+99)
            high = joblib.load(o/'models'/f'{context}_final.joblib')
            prior = q[train].mean((0, 1))
            selected = choices.loc[choices.context.eq(context)].set_index('source').policy
            for rate in [.05, .1, .2]:
                z, added = mask_observed(d['x'][test], rate, p['seed']+r*10000+f*100)
                assert np.isnan(z)[np.isnan(d['x'][test])].all()
                old = base_probabilities(baseline, z)
                original = enforce_hierarchy(median_grades(old), d['parents'])
                new = high.predict_distribution(z, old)
                values = {'original_median': original}
                for source, name, prob in [('old', 'old_same_selector', old), ('new', 'new_selected', new)]:
                    policy = selected[source]
                    values[name] = original if pd.isna(policy) else decide(prob, original, prior, *policies(p)[int(policy)], d['parents'])
                for name, grade in values.items():
                    for scope, mask in [('all50', np.ones(len(d['primary']), bool)), ('primary45', d['primary'])]:
                        rows.append(dict(repeat=r, fold=f, method=name, scope=scope, additional_missing_rate=rate,
                            n_beers=len(test), added_missing_cells=int(added.sum()),
                            raw_rating_mae=float(rating_loss(q[test][:, mask], grade[:, mask]).mean())))
            print(f'Missingness complete {context}', flush=True)
    save_csv(o, 'missingness_by_fold.csv', rows)
    frame = pd.DataFrame(rows)
    save_csv(o, 'missingness_summary.csv', [dict(method=name, scope=scope, additional_missing_rate=rate,
        raw_rating_mae=float(np.average(part.raw_rating_mae, weights=part.n_beers)))
        for (name, scope, rate), part in frame.groupby(['method', 'scope', 'additional_missing_rate'])])
    write_json(o/'missingness_verification.json', dict(status='completed', heldout_masks=27,
        baseline_models_refit=9, original_missingness_preserved=True, labels_and_policies_not_refit=True,
        scope='Fixed MCAR additional masking; no informative-missingness or external-test claim'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--output', type=Path, default=HERE/'forest_results')
    with threadpool_limits(limits=1):
        run(parser.parse_args())
