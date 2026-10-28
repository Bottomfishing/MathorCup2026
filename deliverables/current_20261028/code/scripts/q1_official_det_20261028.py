"""Official Q1 table with the batch-independent fixed-threshold decision (T1).

The historical official table promoted cells to 5/6 with a per-(target, fold)
top-k rule whose number of promoted cells was ``round(rate * n_rows)`` -- it
depended on the request batch (whole batch vs per row differs in 217 cells and
row-by-row scoring produced zero 5/6 cells).  This script rebuilds the official
250x50 table with the per-sample rule of ``configs/q1_decision_det_20261028.json``:

* ``tau_ge5``/``tau_6`` are frozen per (target, outer fold) from the outer-training
  cells only (historical OOF probabilities + panel medians); a cell's threshold is
  never computed from its own fold, let alone from the request;
* every cell is graded from its own probabilities: ``p6 >= tau_6 -> 6``, else
  ``p_ge5 >= tau_ge5 -> 5``, else ``min(legacy_integer_median, 4)``;
* the 24 checkbox parent/child pairs are capped afterwards.

The script also proves batch invariance (whole batch / per row / chunks / uneven
chunks / shuffled / appended rows), threshold purity (perturbing outer-validation
truth changes neither thresholds nor grades), runtime/portrait consistency, and that
the historical decision directory is untouched.  Results go to
``outputs/q1_decision_det_20261028/``; the old directory is only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT/'src'))

from beer_models.q1_decision import (  # noqa: E402
    enforce_parent_child, fold_threshold, threshold_decision, training_threshold)

Q1_DIR = ROOT/'outputs/task6_final_20261026/paper/tables'
R2 = ROOT/'outputs/r2_raw_q1_20261026'
OLD = ROOT/'outputs/q1_decision_20261028'
RAW = ROOT/'data/reference/zenodo23624915/Raw sensory data anonymized.csv'
CALIBRATION = ROOT/'data/current/risk_full_training.csv.gz'
SCOPE = ROOT/'outputs/task1_acceptance_20261025/scorecard_scope.csv'

SUBMISSION_COLUMNS = ['beer', 'target', 'target_family', 'grade_0_6',
                      'probability_rating_ge5', 'probability_rating_6']
EVALUATION_COLUMNS = ['beer', 'target', 'target_family', 'grade_integer_median_old',
                      'grade_0_6', 'panel_mean_reference', 'panel_median_reference',
                      'probability_rating_ge1', 'probability_rating_ge5',
                      'probability_rating_6', 'outer_fold', 'tau_ge5', 'tau_6']
METRIC_COLUMNS = ['scope', 'method', 'panel_median_mae', 'panel_mean_mae', 'exact_match',
                  'within_one', 'macro_spearman', 'n_pred_ge5', 'precision_ge5',
                  'recall_ge5', 'n_pred_6', 'precision_6', 'recall_6']
CALIBER_COLUMNS = ['scope', 'prediction_set', 'caliber', 'n_predicted', 'true_positive',
                   'precision', 'n_observed', 'recall']
BATCH_COLUMNS = ['scenario', 'batch_size', 'n_rows', 'n_compared',
                 'max_abs_grade_diff', 'mismatch_cells', 'pass_ok']
THRESHOLD_COLUMNS = ['target', 'r_ge5', 'r_6', 'n_train_ge5', 'n_train_6', 'k_ge5',
                     'k_6', 'tau_ge5', 'tau_6', 'estimator_slice', 'fold',
                     'emits_ge5', 'emits_6']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='original attachment directory')
    parser.add_argument('--q1', type=Path, default=Q1_DIR)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--scope', type=Path, default=SCOPE)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/q1_decision_det_20261028.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q1_decision_det_20261028')
    parser.add_argument('--old-decision', type=Path, default=OLD)
    parser.add_argument('--skip-runtime', action='store_true', help='debug only: skip bundle retrain')
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else (ROOT/path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + '\n',
                    encoding='utf-8')


def target_family(target):
    if target.startswith('A_'):
        return 'Aroma'
    if target.startswith('F_'):
        return 'Flavour'
    return 'Supplementary'


def parent_child_pairs(scope, targets):
    rows = scope[scope.template_rule.eq('checkbox_zero_or_parent')]
    index = {name: position for position, name in enumerate(targets)}
    pairs = []
    for child in rows.target_id.astype(str):
        parts = child.split('_')
        parent = '_'.join(parts[:2] + ['all'])
        if child not in index or parent not in index:
            raise SystemExit('unknown parent/child pair %r -> %r' % (child, parent))
        pairs.append([index[parent], index[child]])
    pairs.sort(key=lambda pair: pair[1])
    if len(pairs) != 24:
        raise SystemExit('expected 24 checkbox pairs, got %d' % len(pairs))
    return pairs


def count_violations(matrix, pairs):
    return sum(int((matrix[:, child] > matrix[:, parent]).sum()) for parent, child in pairs)


def weighted_individual_metrics(raw_long, prediction):
    records = raw_long.merge(prediction[['beer', 'target', 'predicted_grade']],
                             on=['beer', 'target'], how='left', validate='many_to_one')
    if records['predicted_grade'].isna().any():
        raise SystemExit('individual-level merge left unmatched cells')
    weight = records['weight_within_beer'].to_numpy(float)
    error = np.abs(records['predicted_grade'].to_numpy(float) - records['rating'].to_numpy(float))
    return dict(weighted_mae=float((weight*error).sum()/weight.sum()),
                weighted_exact=float((weight*(error == 0)).sum()/weight.sum()),
                weighted_within_one=float((weight*(error <= 1)).sum()/weight.sum()),
                n_records=int(len(records)), weight_total=float(weight.sum()))


def macro_spearman(frame, column):
    import warnings
    values = []
    for _target, group in frame.groupby('target', sort=True):
        if group['panel_mean_reference'].nunique() > 2:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                values.append(spearmanr(group[column], group['panel_mean_reference']).correlation)
    return float(np.nanmean(values)) if values else float('nan')


def evaluate(frame, column):
    predicted = frame[column].to_numpy(float)
    median = frame['panel_median_reference'].to_numpy(float)
    mean = frame['panel_mean_reference'].to_numpy(float)
    observed5 = median >= 5
    observed6 = median >= 6
    def events(pred, obs):
        count = int(pred.sum())
        hits = int((pred & obs).sum())
        return count, (hits/count if count else 0.0), (hits/obs.sum() if obs.sum() else 0.0)
    n5, p5, r5 = events(predicted >= 5, observed5)
    n6, p6, r6 = events(predicted >= 6, observed6)
    return dict(panel_median_mae=float(np.abs(predicted-median).mean()),
                panel_mean_mae=float(np.abs(predicted-mean).mean()),
                exact_match=float((predicted == median).mean()),
                within_one=float((np.abs(predicted-median) <= 1).mean()),
                macro_spearman=macro_spearman(frame, column),
                n_pred_ge5=n5, precision_ge5=float(p5), recall_ge5=float(r5),
                n_pred_6=n6, precision_6=float(p6), recall_6=float(r6))


def grade_distribution(grades):
    values, counts = np.unique(np.asarray(grades, dtype=int), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def caliber_rows(frame, set_name, column='grade_0_6', scope_name='beer_level'):
    predicted = frame[column].to_numpy(int)
    median = frame['panel_median_reference'].to_numpy(int)
    rows = []
    for label, pred, obs in (('exact5', predicted == 5, median == 5),
                             ('exact6', predicted == 6, median == 6),
                             ('ge5', predicted >= 5, median >= 5)):
        hits = int((pred & obs).sum())
        rows.append(dict(scope=scope_name, prediction_set=set_name, caliber=label,
                         n_predicted=int(pred.sum()), true_positive=hits,
                         precision=float(hits/pred.sum()) if pred.sum() else 0.0,
                         n_observed=int(obs.sum()),
                         recall=float(hits/obs.sum()) if obs.sum() else 0.0))
    return rows


def individual_caliber_rows(raw_long, prediction, set_name):
    records = raw_long.merge(prediction[['beer', 'target', 'predicted_grade']],
                             on=['beer', 'target'], how='left', validate='many_to_one')
    weight = records['weight_within_beer'].to_numpy(float)
    actual = records['rating'].to_numpy(float)
    predicted = records['predicted_grade'].to_numpy(float)
    rows = []
    for label, pred, obs in (('exact5', predicted == 5, actual == 5),
                             ('exact6', predicted == 6, actual == 6),
                             ('ge5', predicted >= 5, actual >= 5)):
        hits = float((weight*pred*obs).sum())
        predicted_mass = float((weight*pred).sum())
        observed_mass = float((weight*obs).sum())
        rows.append(dict(scope='individual_taster_weighted', prediction_set=set_name,
                         caliber=label, n_predicted=predicted_mass, true_positive=hits,
                         precision=float(hits/predicted_mass) if predicted_mass else 0.0,
                         n_observed=observed_mass,
                         recall=float(hits/observed_mass) if observed_mass else 0.0))
    return rows


def load_cells(args, config):
    obs = pd.read_csv(resolve(R2/'observed_panel_distribution.csv'))
    probability = pd.read_csv(resolve(config['inputs']['oof_probabilities']))
    targets = sorted(obs.target.unique())
    probability = probability[probability.target.isin(targets)].copy()
    if len(probability) != len(obs):
        raise SystemExit('OOF probability table does not cover the 12500 cells')
    frame = obs.merge(probability[['beer', 'target', 'probability_rating_ge1',
                                   'probability_rating_ge5', 'probability_rating_6']],
                      on=['beer', 'target'], how='left', validate='one_to_one')
    if frame[['probability_rating_ge5', 'probability_rating_6']].isna().any().any():
        raise SystemExit('missing OOF probabilities')
    legacy = pd.read_csv(resolve(R2/'oof_predictions.csv.gz'))
    legacy = legacy[legacy.method.eq('extra_trees')][['beer', 'target', 'predicted_grade']]
    frame = frame.merge(legacy, on=['beer', 'target'], how='left', validate='one_to_one')
    if frame['predicted_grade'].isna().any():
        raise SystemExit('missing legacy grades')
    folds = pd.read_csv(resolve(R2/'rating_weights_and_folds.csv'))
    fold_of_beer = folds.groupby('beer').outer_fold.first().to_dict()
    frame['outer_fold'] = frame.beer.map(fold_of_beer).astype(int)
    frame['target_family'] = frame.target.map(target_family)
    frame['panel_median_reference'] = frame['panel_weighted_median'].astype(float)
    frame['panel_mean_reference'] = frame['panel_weighted_mean'].astype(float)
    frame['grade_integer_median_old'] = frame['predicted_grade'].astype(int)
    frame = frame.sort_values(['beer', 'target'], kind='stable').reset_index(drop=True)
    return frame, targets


def pair_thresholds(frame, fold):
    """Threshold parameters for one outer fold, estimated on the other folds only."""
    part = frame[frame.outer_fold != fold]
    details = {}
    for threshold, key, column in ((5, 'tau_ge5', 'probability_rating_ge5'),
                                   (6, 'tau_6', 'probability_rating_6')):
        details[key] = training_threshold(part[column].to_numpy(float),
                                          part['panel_median_reference'].to_numpy(float),
                                          part['target'].astype(str).to_numpy(), threshold)
    return details


def decision_matrix(frame, targets, pairs):
    groups = (frame['target'].astype(str) + '|' + frame['outer_fold'].astype(str)).to_numpy()
    grades = threshold_decision(frame['probability_rating_ge5'].to_numpy(float),
                                frame['probability_rating_6'].to_numpy(float),
                                frame['grade_integer_median_old'].to_numpy(float),
                                frame['tau_ge5'].to_numpy(float),
                                frame['tau_6'].to_numpy(float), groups)
    beer_order = pd.unique(frame.beer)
    target_index = {target: position for position, target in enumerate(targets)}
    beer_index = {int(beer): position for position, beer in enumerate(beer_order)}
    matrix = np.full((len(beer_order), len(targets)), -1, dtype=int)
    matrix[[beer_index[int(beer)] for beer in frame.beer],
           [target_index[target] for target in frame.target]] = grades
    if (matrix < 0).any():
        raise SystemExit('decision matrix has uncovered cells')
    capped = enforce_parent_child(matrix, [pair[0] for pair in pairs], [pair[1] for pair in pairs])
    back = capped[[beer_index[int(beer)] for beer in frame.beer],
                  [target_index[target] for target in frame.target]]
    return grades, capped, back


def batch_scenarios(cells, decide):
    base = decide(cells)
    rows = []
    def record(name, batch_size, rebuilt):
        rebuilt = np.asarray(rebuilt, dtype=int)
        rows.append(dict(scenario=name, batch_size=int(batch_size), n_rows=int(len(base)),
                         n_compared=int(base.size),
                         max_abs_grade_diff=int(np.max(np.abs(rebuilt-base))),
                         mismatch_cells=int((rebuilt != base).sum()),
                         pass_ok=bool(np.array_equal(rebuilt, base))))
    rebuilt = np.zeros_like(base)
    for start in range(len(base)):
        rebuilt[start:start+1] = decide(cells[start:start+1])
    record('per_row', 1, rebuilt)
    record('chunks_of_10', 10, np.concatenate([decide(cells[start:start+10])
                                               for start in range(0, len(base), 10)]))
    parts, position = [], 0
    for size in [1, 7, 13, 199]:
        parts.append(decide(cells[position:position+size]))
        position += size
    parts.append(decide(cells[position:]))
    record('uneven_chunks', 0, np.concatenate(parts))
    rng = np.random.default_rng(20261028)
    order = rng.permutation(len(base))
    rebuilt = np.zeros_like(base)
    rebuilt[order] = decide(cells.iloc[order].reset_index(drop=True))
    record('shuffled', 0, rebuilt)
    extra = cells.head(25).copy()
    record('appended_irrelevant_rows', 0,
           decide(pd.concat([cells, extra], ignore_index=True))[:len(base)])
    return base, rows


def runtime_checks(source, raw, frame, targets):
    from beer_models.current_runtime import train
    from beer_models.inference import _official_grade_frame, official_scores, run
    temp = Path(tempfile.mkdtemp(prefix='q1det_'))
    models = temp/'models'
    try:
        train(source, raw, models)
        import joblib
        bundle = joblib.load(models/'q1.joblib')
        decision = bundle['q1_decision']
        if decision['rule'] != 'fixed_threshold_quantile' or bundle['grade_method'] != decision['rule']:
            raise SystemExit('runtime bundle does not carry the T1 rule')
        chemistry = pd.read_csv(source/'1.csv', dtype=str)
        chemistry['beer'] = chemistry['beer'].astype(str)
        frame_beers = pd.unique(frame.beer.astype(str))
        frame_input = chemistry[chemistry.beer.isin(frame_beers)].reset_index(drop=True)
        if len(frame_input) != len(frame_beers):
            raise SystemExit('chemistry frame does not cover every beer')
        reference, _, _, _ = _official_grade_frame(frame_input, bundle)
        invariance = []
        def record(name, batch_size, rebuilt):
            rebuilt = np.asarray(rebuilt, dtype=int)
            invariance.append(dict(scenario=name, batch_size=int(batch_size),
                                   n_rows=int(reference.size), n_compared=int(reference.size),
                                   max_abs_grade_diff=int(np.max(np.abs(rebuilt-reference))),
                                   mismatch_cells=int((rebuilt != reference).sum()),
                                   pass_ok=bool(np.array_equal(rebuilt, reference))))
        per_row = np.zeros_like(reference)
        for row in range(len(frame_input)):
            values, _, _, _ = _official_grade_frame(frame_input.iloc[row:row+1], bundle)
            per_row[row] = values[0]
        record('runtime_per_row', 1, per_row)
        chunked = np.vstack([_official_grade_frame(frame_input.iloc[start:start+10], bundle)[0]
                             for start in range(0, len(frame_input), 10)])
        record('runtime_chunks_of_10', 10, chunked)
        rng = np.random.default_rng(20261029)
        order = rng.permutation(len(frame_input))
        shuffled, _, _, _ = _official_grade_frame(frame_input.iloc[order].reset_index(drop=True), bundle)
        restored = np.zeros_like(reference)
        restored[order] = shuffled
        record('runtime_shuffled', 0, restored)
        appended_input = pd.concat([frame_input,
                                    frame_input.head(5).assign(beer=lambda d: d.beer.astype(str)+'999')],
                                   ignore_index=True)
        appended, _, _, _ = _official_grade_frame(appended_input, bundle)
        record('runtime_appended_rows', 0, appended[:len(frame_input)])
        out_run = temp/'run_out'
        run(models/'q1.joblib', source/'1.csv', out_run)
        run_long = pd.read_csv(out_run/'scores_all52_long.csv')
        run_long['beer'] = run_long['beer'].astype(str)
        portrait_long, _, portrait_wide = official_scores(frame_input, bundle)
        portrait_compare = portrait_long[['beer', 'target', 'estimated_grade_0_6']].copy()
        portrait_compare['beer'] = portrait_compare['beer'].astype(str)
        merged = run_long.merge(portrait_compare, on=['beer', 'target'], how='inner',
                                validate='one_to_one', suffixes=('_run', '_portrait'))
        mismatch = int((merged.estimated_grade_0_6_run.astype('Int64')
                        != merged.estimated_grade_0_6_portrait.astype('Int64')).fillna(False).sum())
        decision_info = json.loads((out_run/'prediction_info.json').read_text(encoding='utf-8'))
        return dict(bundle=bundle, decision_info=decision_info, invariance=invariance,
                    entry_mismatch=mismatch)
    finally:
        shutil.rmtree(temp, ignore_errors=True)


def main():
    args = parse_args()
    config = json.loads(resolve(args.config).read_text(encoding='utf-8'))
    source = resolve(args.source)
    output = resolve(args.output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    scope = pd.read_csv(resolve(args.scope))
    frame, targets = load_cells(args, config)
    pairs = parent_child_pairs(scope, targets)
    if len(frame) != 12500 or len(targets) != 50:
        raise SystemExit('expected 12500 cells over 50 targets, got %d/%d' % (len(frame), len(targets)))

    protocol = dict(protocol_id=config['protocol_id'], stage='protocol_written_before_computation',
                    frozen_before_results=True, config=config,
                    inputs={name: dict(path=str(path), sha256=sha256(path)) for name, path in [
                        ('oof_probabilities', resolve(config['inputs']['oof_probabilities'])),
                        ('observed_panel_distribution', resolve(R2/'observed_panel_distribution.csv')),
                        ('oof_predictions', resolve(R2/'oof_predictions.csv.gz')),
                        ('rating_weights_and_folds', resolve(R2/'rating_weights_and_folds.csv')),
                        ('raw_sensory', resolve(args.raw)),
                        ('training_inner_oof', resolve(CALIBRATION)),
                        ('scorecard_scope', resolve(args.scope))]},
                    implementation=dict(module='src/beer_models/q1_decision.py',
                                        sha256=sha256(ROOT/'src/beer_models/q1_decision.py'),
                                        runtime_sha256=sha256(ROOT/'src/beer_models/current_runtime.py'),
                                        inference_sha256=sha256(ROOT/'src/beer_models/inference.py'),
                                        script_sha256=sha256(Path(__file__).resolve())))
    write_json(output/'protocol.json', protocol)
    old_hashes = {path.name: sha256(path) for path in sorted(resolve(args.old_decision).glob('*'))
                  if path.is_file()}

    # --- frozen per-fold thresholds from the outer-training cells only ----------------
    tau_ge5 = np.zeros(len(frame))
    tau_6 = np.zeros(len(frame))
    thresholds = {}
    fold_ge5 = fold_threshold(frame['probability_rating_ge5'].to_numpy(float),
                              frame['panel_median_reference'].to_numpy(float),
                              frame['target'].astype(str).to_numpy(),
                              frame['outer_fold'].to_numpy(int), 5)
    fold_6 = fold_threshold(frame['probability_rating_6'].to_numpy(float),
                            frame['panel_median_reference'].to_numpy(float),
                            frame['target'].astype(str).to_numpy(),
                            frame['outer_fold'].to_numpy(int), 6)
    for fold in sorted(frame.outer_fold.unique()):
        thresholds[int(fold)] = pair_thresholds(frame, fold)
        mask = (frame.outer_fold == fold).to_numpy()
        tau_ge5[mask] = fold_ge5[mask]
        tau_6[mask] = fold_6[mask]
    frame['tau_ge5'] = tau_ge5
    frame['tau_6'] = tau_6

    prepcap, capped, grades = decision_matrix(frame, targets, pairs)
    frame['grade_0_6'] = grades
    frame['grade_precap'] = prepcap

    # --- threshold purity: outer-validation truth cannot move thresholds or grades --
    perturbed = frame.copy()
    noise = np.random.default_rng(7).normal(0.5, 0.1, len(perturbed))
    perturbed['panel_median_reference'] = np.clip(
        perturbed['panel_median_reference'] + noise, 0, 6)
    rebuilt_ge5 = fold_threshold(frame['probability_rating_ge5'].to_numpy(float),
                                 frame['panel_median_reference'].to_numpy(float),
                                 frame['target'].astype(str).to_numpy(),
                                 frame['outer_fold'].to_numpy(int), 5)
    rebuilt_6 = fold_threshold(frame['probability_rating_6'].to_numpy(float),
                               frame['panel_median_reference'].to_numpy(float),
                               frame['target'].astype(str).to_numpy(),
                               frame['outer_fold'].to_numpy(int), 6)
    with np.errstate(invalid='ignore'):
        both = np.isinf(rebuilt_ge5) & np.isinf(tau_ge5)
        any_inf = np.isinf(rebuilt_ge5) | np.isinf(tau_ge5)
        tau_delta = float(np.max(np.where(any_inf, np.where(both, 0.0, np.inf),
                                          np.abs(rebuilt_ge5 - tau_ge5))))
        both = np.isinf(rebuilt_6) & np.isinf(tau_6)
        any_inf = np.isinf(rebuilt_6) | np.isinf(tau_6)
        tau_delta = max(tau_delta, float(np.max(np.where(any_inf, np.where(both, 0.0, np.inf),
                                                        np.abs(rebuilt_6 - tau_6)))))
    _, _, pert_back = decision_matrix(perturbed, targets, pairs)
    grade_delta = int(np.max(np.abs(pert_back - grades))) if grades.size else 0

    # --- batch invariance on the OOF cells -------------------------------------------
    cells = frame[['probability_rating_ge5', 'probability_rating_6',
                   'grade_integer_median_old', 'tau_ge5', 'tau_6', 'target',
                   'outer_fold']].copy()
    def decide(block):
        groups = (block['target'].astype(str) + '|' + block['outer_fold'].astype(str)).to_numpy()
        return threshold_decision(block['probability_rating_ge5'].to_numpy(float),
                                  block['probability_rating_6'].to_numpy(float),
                                  block['grade_integer_median_old'].to_numpy(float),
                                  block['tau_ge5'].to_numpy(float),
                                  block['tau_6'].to_numpy(float), groups)
    _, batch_rows = batch_scenarios(cells, decide)
    batch_frame = pd.DataFrame(batch_rows, columns=BATCH_COLUMNS)

    runtime = None
    if not args.skip_runtime:
        runtime = runtime_checks(source, resolve(args.raw), frame, targets)
        batch_frame = pd.concat([batch_frame,
                                 pd.DataFrame(runtime['invariance'], columns=BATCH_COLUMNS)],
                                ignore_index=True)
        entry_mismatch = int(runtime['entry_mismatch'])
    else:
        entry_mismatch = 0
    batch_frame['pass_ok'] = batch_frame['pass_ok'].astype(bool)

    # --- metrics ---------------------------------------------------------------------
    frame['grade_legacy'] = frame['grade_integer_median_old']
    old = pd.read_csv(resolve(args.old_decision)/'q1__evaluation_long_12500.csv')
    comparing = frame.merge(old[['beer', 'target', 'grade_0_6']].rename(
        columns={'grade_0_6': 'grade_old_topk'}), on=['beer', 'target'], how='left',
        validate='one_to_one')
    metrics = []
    metrics.append(dict(scope='beer_level', method='q1_integer_median_legacy',
                        **evaluate(frame, 'grade_legacy')))
    metrics.append(dict(scope='beer_level', method='q1_fixed_threshold_quantile',
                        **evaluate(frame, 'grade_0_6')))
    metrics.append(dict(scope='beer_level', method='q1_topk_parent_capped_old_baseline',
                        **evaluate(comparing, 'grade_old_topk')))
    raw = pd.read_csv(resolve(args.raw), encoding='cp1252')
    raw['raw_record_1based'] = np.arange(1, len(raw) + 1)
    raw = raw.rename(columns={'A_ester_fruity': 'A_esters_fruity',
                              'F_ester_fruity': 'F_esters_fruity'})
    weights = pd.read_csv(resolve(R2/'rating_weights_and_folds.csv'))
    raw = raw.merge(weights[['raw_record_1based', 'beer', 'weight_within_beer']],
                    on='raw_record_1based', how='left')
    if raw[['beer', 'weight_within_beer']].isna().any().any():
        raise SystemExit('raw sensory rows are not fully mapped to competition beers')
    raw_long = raw.melt(id_vars=['raw_record_1based', 'beer', 'weight_within_beer'],
                        value_vars=targets, var_name='target', value_name='rating')
    weight_sums = raw_long.groupby(['beer', 'target'])['weight_within_beer'].sum()
    if float((weight_sums - 1.0).abs().max()) > 1e-9:
        raise SystemExit('within-beer weights do not sum to one')
    individual_det = weighted_individual_metrics(
        raw_long, frame[['beer', 'target']].assign(predicted_grade=frame['grade_0_6']))
    individual_legacy = weighted_individual_metrics(
        raw_long, frame[['beer', 'target']].assign(predicted_grade=frame['grade_legacy']))
    metrics.append(dict(scope='individual_taster_weighted', method='q1_fixed_threshold_quantile',
                        panel_median_mae=individual_det['weighted_mae'], panel_mean_mae=np.nan,
                        exact_match=individual_det['weighted_exact'],
                        within_one=individual_det['weighted_within_one'], macro_spearman=np.nan,
                        n_pred_ge5=np.nan, precision_ge5=np.nan, recall_ge5=np.nan,
                        n_pred_6=np.nan, precision_6=np.nan, recall_6=np.nan))
    metrics.append(dict(scope='individual_taster_weighted', method='q1_integer_median_legacy',
                        panel_median_mae=individual_legacy['weighted_mae'], panel_mean_mae=np.nan,
                        exact_match=individual_legacy['weighted_exact'],
                        within_one=individual_legacy['weighted_within_one'], macro_spearman=np.nan,
                        n_pred_ge5=np.nan, precision_ge5=np.nan, recall_ge5=np.nan,
                        n_pred_6=np.nan, precision_6=np.nan, recall_6=np.nan))
    metrics_frame = pd.DataFrame(metrics, columns=METRIC_COLUMNS)

    # --- calibers --------------------------------------------------------------------
    calibers = caliber_rows(frame, 'q1_fixed_threshold_quantile')
    calibers += caliber_rows(comparing, 'q1_topk_parent_capped_old_baseline', 'grade_old_topk')
    legacy_frame = frame.copy()
    legacy_frame['grade_0_6'] = legacy_frame['grade_legacy']
    calibers += caliber_rows(legacy_frame, 'q1_integer_median_legacy')
    calibers += individual_caliber_rows(raw_long, frame[['beer', 'target']].assign(
        predicted_grade=frame['grade_0_6']), 'q1_fixed_threshold_quantile')
    calibers += individual_caliber_rows(raw_long, frame[['beer', 'target']].assign(
        predicted_grade=frame['grade_legacy']), 'q1_integer_median_legacy')
    caliber_frame = pd.DataFrame(calibers, columns=CALIBER_COLUMNS)

    # --- thresholds table ------------------------------------------------------------
    threshold_rows = []
    for fold in sorted(frame.outer_fold.unique()):
        details = thresholds[int(fold)]
        for position, target in enumerate(targets):
            d5, d6 = details['tau_ge5'], details['tau_6']
            threshold_rows.append(dict(
                target=target, fold=int(fold), estimator_slice='historical_OOF_outer_training',
                r_ge5=float(d5['rate'][position]), r_6=float(d6['rate'][position]),
                n_train_ge5=int(d5['support'][position]), n_train_6=int(d6['support'][position]),
                k_ge5=int(d5['k'][position]), k_6=int(d6['k'][position]),
                tau_ge5=float(d5['tau'][position]), tau_6=float(d6['tau'][position]),
                emits_ge5=bool(np.isfinite(d5['tau'][position])),
                emits_6=bool(np.isfinite(d6['tau'][position]))))
    calibration = pd.read_csv(CALIBRATION, float_precision='round_trip')
    part = calibration[calibration.source == 'extra_trees']
    beer_order = sorted(part.beer.unique())
    def matrix(column):
        return part.pivot(index='beer', columns='target', values=column).loc[
            beer_order, targets].to_numpy(float)
    p_ge5_train = matrix('p5') + matrix('p6')
    p6_train = matrix('p6')
    truth = frame.pivot(index='beer', columns='target', values='panel_median_reference').loc[
        beer_order, targets].to_numpy(float)
    groups = np.tile(np.asarray(targets, dtype=object), p_ge5_train.shape[0])
    run5 = training_threshold(p_ge5_train.ravel(), truth.ravel(), groups, 5)
    run6 = training_threshold(p6_train.ravel(), truth.ravel(), groups, 6)
    for position, target in enumerate(targets):
        threshold_rows.append(dict(
            target=target, fold='all', estimator_slice='historical_inner_OOF_full_training',
            r_ge5=float(run5['rate'][position]), r_6=float(run6['rate'][position]),
            n_train_ge5=int(run5['support'][position]), n_train_6=int(run6['support'][position]),
            k_ge5=int(run5['k'][position]), k_6=int(run6['k'][position]),
            tau_ge5=float(run5['tau'][position]), tau_6=float(run6['tau'][position]),
            emits_ge5=bool(np.isfinite(run5['tau'][position])),
            emits_6=bool(np.isfinite(run6['tau'][position]))))
    threshold_frame = pd.DataFrame(threshold_rows, columns=THRESHOLD_COLUMNS)

    # --- tables ----------------------------------------------------------------------
    evaluation = frame[EVALUATION_COLUMNS].copy()
    submission = evaluation[SUBMISSION_COLUMNS].copy()
    wide = submission.pivot(index='beer', columns='target', values='grade_0_6')[
        targets].reset_index()
    evaluation.to_csv(output/'q1__evaluation_long_12500.csv', index=False, encoding='utf-8-sig')
    submission.to_csv(output/'q1__submission_long_12500.csv', index=False, encoding='utf-8-sig')
    wide.to_csv(output/'q1__submission_wide_250x50.csv', index=False, encoding='utf-8-sig')
    metrics_frame.to_csv(output/'q1__metrics_summary.csv', index=False, encoding='utf-8-sig')
    caliber_frame.to_csv(output/'q1__metric_calibers.csv', index=False, encoding='utf-8-sig')
    batch_frame.to_csv(output/'q1__batch_invariance.csv', index=False, encoding='utf-8-sig')
    threshold_frame.to_csv(output/'q1__thresholds.csv', index=False, encoding='utf-8-sig')

    # --- verification ----------------------------------------------------------------
    new_dist = grade_distribution(frame['grade_0_6'])
    old_dist = grade_distribution(comparing['grade_old_topk'].dropna().astype(int))
    fallback5 = sorted(set(threshold_frame.loc[(threshold_frame.fold == 'all')
                                               & (~threshold_frame.emits_ge5), 'target']))
    fallback6 = sorted(set(threshold_frame.loc[(threshold_frame.fold == 'all')
                                               & (~threshold_frame.emits_6), 'target']))
    old_after = {path.name: sha256(path) for path in sorted(resolve(args.old_decision).glob('*'))
                 if path.is_file()}
    by_name = {row.scenario: row for row in batch_frame.itertuples(index=False)}
    purity_rows = []
    for fold in sorted(frame.outer_fold.unique()):
        train_beers = set(frame[frame.outer_fold != fold].beer)
        validation_beers = set(frame[frame.outer_fold == fold].beer)
        purity_rows.append(not (train_beers & validation_beers))
    checks = {
        'q1_rows_12500': bool(len(evaluation) == 12500 and len(submission) == 12500
                              and wide.shape == (250, 51)),
        'q1_grade_range_0_6': bool(frame['grade_0_6'].between(0, 6).all()
                                   and wide.drop(columns='beer').isin(range(0, 7)).all().all()),
        'q1_acetaldehyde_slots_na': bool('A_acetaldehyde' not in set(frame.target)
                                         and 'F_acetaldehyde' not in set(frame.target)),
        'q1_parent_child_violations_zero': bool(count_violations(capped, pairs) == 0),
        'q1_thresholds_training_only': bool(all(purity_rows)),
        'q1_threshold_purity_perturbation': bool(tau_delta == 0.0 and grade_delta == 0),
        'q1_batch_invariance_whole_vs_per_row': bool(by_name['per_row'].pass_ok),
        'q1_batch_invariance_chunks_of_10': bool(by_name['chunks_of_10'].pass_ok),
        'q1_batch_invariance_uneven_chunks': bool(by_name['uneven_chunks'].pass_ok),
        'q1_batch_invariance_shuffled': bool(by_name['shuffled'].pass_ok),
        'q1_batch_invariance_appended_irrelevant_rows': bool(by_name['appended_irrelevant_rows'].pass_ok),
        'q1_runtime_batch_invariance': bool(all(row.pass_ok for row in batch_frame.itertuples(index=False)
                                                if row.scenario.startswith('runtime'))),
        'q1_runtime_vs_official_scores_identical': bool(entry_mismatch == 0),
        'q1_old_decision_directory_unchanged': bool(old_hashes == old_after),
        'q1_metrics_present': bool(len(metrics_frame) == 5 and len(caliber_frame) == 15),
        'q1_threshold_fallback_targets_listed': bool(len(fallback5) + len(fallback6) > 0),
    }
    verification = dict(all_pass=bool(all(checks.values())), checks=checks,
                        diagnostics=dict(new_distribution=new_dist, old_distribution=old_dist,
                                         n_ge5=int((frame['grade_0_6'] >= 5).sum()),
                                         n_6=int((frame['grade_0_6'] == 6).sum()),
                                         fallback_no_ge5=fallback5, fallback_no_6=fallback6,
                                         tau_purity_max_abs_delta=tau_delta,
                                         grade_purity_max_abs_delta=grade_delta,
                                         entry_mismatch_cells=entry_mismatch,
                                         old_directory_hashes_match=bool(old_hashes == old_after)))
    write_json(output/'verification.json', verification)

    det_metrics = evaluate(frame, 'grade_0_6')
    metrics_by_key = {f"{row['scope']}|{row['method']}": {
        key: row[key] for key in METRIC_COLUMNS if key not in ('scope', 'method')}
        for row in metrics_frame.to_dict('records')}
    promoted_by_target = (frame.assign(promoted=frame['grade_0_6'] >= 5)
                                .groupby('target')['promoted'].sum().sort_values(ascending=False))
    summary = dict(protocol_id=config['protocol_id'],
                   decision_rule='fixed_threshold_quantile (per sample; frozen per-target tau)',
                   new_distribution=new_dist, old_capped_distribution=old_dist,
                   legacy_median_distribution=grade_distribution(frame['grade_legacy']),
                   metrics=metrics_by_key,
                   threshold_fallback=dict(no_ge5=fallback5, no_6=fallback6),
                   over_promotion_diagnostic=dict(
                       reference='old batch top-k table promoted 194 cells to >=5 and 14 to 6',
                       promoted_by_target_top10={str(k): int(v)
                                                 for k, v in promoted_by_target.head(10).items()},
                       note=('a fixed quantile threshold cannot bound the promoted count in a '
                             'request: when a validation fold is predicted higher than the '
                             'training folds the threshold is exceeded by many cells (e.g. '
                             'clove fold 0 where all 84 cells share p_ge5=0.006574 and the '
                             'training-fold threshold is 0.005356)')),
                   trade_off=('fixed thresholds promote %d cells to >=5 and %d to 6, versus '
                              'the old batch top-k table 194/14; the counts follow the frozen '
                              'training quantiles, not the request size'
                              % (int((frame['grade_0_6'] >= 5).sum()),
                                 int((frame['grade_0_6'] == 6).sum()))),
                   batch_invariance={row['scenario']: dict(pass_ok=bool(row['pass_ok']),
                                                           mismatch=int(row['mismatch_cells']))
                                     for row in batch_rows},
                   threshold_purity=dict(tau_max_abs_delta=tau_delta,
                                         grade_max_abs_delta=grade_delta),
                   entry_consistency=dict(mismatch_cells=entry_mismatch,
                                          run_output=(runtime['decision_info'].get('grade_method')
                                                      if runtime else 'runtime skipped')),
                   old_directory_unchanged=bool(old_hashes == old_after),
                   verification_all_pass=verification['all_pass'])
    write_json(output/'summary.json', summary)
    print(json.dumps(dict(all_pass=verification['all_pass'], checks=checks,
                          new_distribution=new_dist, old_distribution=old_dist,
                          median_mae=det_metrics['panel_median_mae'],
                          exact=det_metrics['exact_match'], within_one=det_metrics['within_one'],
                          n_ge5=summary['metrics']['beer_level|q1_fixed_threshold_quantile']['n_pred_ge5'],
                          precision_ge5=det_metrics['precision_ge5'],
                          recall_ge5=det_metrics['recall_ge5'],
                          n_6=det_metrics['n_pred_6'], precision_6=det_metrics['precision_6'],
                          recall_6=det_metrics['recall_6'],
                          fallback_no_ge5=fallback5, fallback_no_6=fallback6,
                          batch=summary['batch_invariance'], tau_purity=summary['threshold_purity'],
                          entry_mismatch=entry_mismatch,
                          files=sorted(path.name for path in output.iterdir())),
                     ensure_ascii=False, indent=1, default=str))
    if not verification['all_pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
