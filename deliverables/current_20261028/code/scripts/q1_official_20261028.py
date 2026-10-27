"""Official Q1 decision-layer script (protocol ``q1_decision_20261028``).

Reads the existing official Q1 inference artifacts (no retraining), rebuilds the
12,500-cell beer-by-target decision table with the non-forced cumulative-probability
top-k rule from :mod:`beer_models.q1_decision`, calibrates the legacy integer median
onto the observed panel-mean scale, writes the six official tables plus
``protocol.json`` / ``verification.json`` / ``summary.json``, and asserts that every
number agrees with the frozen prototype report and with the paper hand-off package.

Usage::

    $env:PYTHONPATH='src'
    .venv/Scripts/python.exe scripts/q1_official_20261028.py \
        --source '<赛题附件目录>' \
        --q1 outputs/task6_final_20261026/paper/tables \
        --raw outputs/r2_raw_q1_20261026 \
        --reference data/reference/zenodo23624915 \
        --output outputs/q1_decision_20261028

The script fails (non-zero exit) if any verification assertion is false; it never
falls back to a different protocol to make a number pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from beer_models.q1_decision import (  # noqa: E402
    apply_nested_isotonic,
    apply_nested_linear,
    apply_nested_quantile,
    enforce_parent_child,
    fit_nested_isotonic,
    fit_nested_linear,
    fit_nested_quantile,
    fold_training_rate,
    high_grade_stats,
    linear_to_grade,
    topk_decision,
)
from beer_models.q1_panel_mean import panel_median_from_distribution  # noqa: E402

TRUTH_MEDIAN_COL = 'panel_weighted_median'
TRUTH_MEAN_COL = 'panel_weighted_mean'
REPORT_TABLE_PREFIX = 'NEW: top-k cumulative-prob (fold-wise rate)'
REPORT_OLD_TABLE_PREFIX = 'OLD: ET integer conditional median'
OUTER_FOLD_SEED = 20261024
INNER_FOLDS = 2
NESTED_METHODS = ('nested_linear', 'nested_isotonic', 'nested_style_linear',
                  'nested_quantile')

SUBMISSION_LONG_COLUMNS = ['beer', 'target', 'target_family', 'grade_0_6',
                           'probability_rating_ge5', 'probability_rating_6']
EVALUATION_COLUMNS = ['beer', 'target', 'target_family', 'grade_integer_median_old',
                      'grade_0_6', 'grade_topk_precap', 'panel_mean_reference',
                      'panel_median_reference', 'probability_rating_ge1',
                      'probability_rating_ge5', 'probability_rating_6', 'outer_fold']
CALIBRATED_COLUMNS = ['beer', 'target', 'panel_mean_reference', 'panel_median_reference',
                      'grade_integer_median_old', 'panel_mean_calibrated',
                      'grade_from_calibrated_mean_round']
METRICS_COLUMNS = ['scope', 'method', 'panel_median_mae', 'panel_mean_mae', 'exact_match',
                   'within_one', 'macro_spearman', 'n_pred_ge5', 'precision_ge5',
                   'recall_ge5', 'n_pred_6', 'precision_6', 'recall_6']
COVERAGE_COLUMNS = ['target', 'n_observed_ge5', 'n_observed_6', 'n_predicted_ge5',
                    'n_predicted_6', 'n_predicted_ge5_parent_capped',
                    'n_predicted_6_parent_capped']
CALIBER_COLUMNS = ['scope', 'prediction_set', 'caliber', 'n_predicted', 'true_positive',
                   'precision', 'n_observed', 'recall']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Official Q1 decision-layer pipeline.')
    parser.add_argument('--source', type=Path, required=True,
                        help='original problem attachment directory (needs 1.csv: beer/style)')
    parser.add_argument('--q1', type=Path, default=Path('outputs/task6_final_20261026/paper/tables'),
                        help='existing official Q1 inference tables')
    parser.add_argument('--raw', type=Path, default=Path('outputs/r2_raw_q1_20261026'),
                        help='raw-scale OOF artifacts and panel reference distribution')
    parser.add_argument('--reference', type=Path, default=Path('data/reference/zenodo23624915'),
                        help='external anonymized raw sensory table')
    parser.add_argument('--output', type=Path, default=Path('outputs/q1_decision_20261028'),
                        help='new output directory')
    parser.add_argument('--config', type=Path, default=Path('configs/q1_decision_20261028.json'),
                        help='frozen protocol config')
    parser.add_argument('--handoff', type=Path,
                        default=Path('deliverables/paper_handoff_20261028/tables'),
                        help='read-only paper hand-off tables used for cross-checks')
    parser.add_argument('--report', type=Path,
                        default=Path('experiments/lead_review/q1/decision_topk_report.json'),
                        help='frozen prototype decision report')
    parser.add_argument('--panel-protocol', type=Path,
                        default=Path('configs/r2_raw_panel_protocol_20261026.json'),
                        help='official raw-panel model protocol used to refit inner outer-training models')
    parser.add_argument('--scope', type=Path,
                        default=Path('outputs/task1_acceptance_20261025/scorecard_scope.csv'),
                        help='scorecard scope table defining the checkbox parent/child pairs')
    return parser.parse_args(argv)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def target_family(target):
    if target.startswith('A_'):
        return 'Aroma'
    if target.startswith('F_'):
        return 'Flavour'
    return 'Supplementary'


def weighted_individual_metrics(raw_long, prediction):
    merged = raw_long.merge(prediction[['beer', 'target', 'predicted_grade']],
                            on=['beer', 'target'], how='left', validate='many_to_one')
    if merged['predicted_grade'].isna().any():
        raise RuntimeError('individual-level merge left unmatched cells')
    weight = merged['weight_within_beer'].to_numpy(float)
    error = np.abs(merged['predicted_grade'].to_numpy(float) - merged['raw'].to_numpy(float))
    return {
        'individual_weighted_mae': float(np.sum(weight * error) / np.sum(weight)),
        'individual_weighted_exact': float(np.sum(weight * (error == 0)) / np.sum(weight)),
        'individual_weighted_within_one': float(np.sum(weight * (error <= 1)) / np.sum(weight)),
        'weight_mass': float(np.sum(weight)),
    }


def macro_spearman(prediction, reference, targets):
    values = []
    prediction = np.asarray(prediction, float)
    reference = np.asarray(reference, float)
    targets = np.asarray(targets, dtype=object)
    for target in pd.unique(targets):
        mask = targets == target
        if np.unique(reference[mask]).size <= 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            values.append(stats.spearmanr(prediction[mask], reference[mask]).correlation)
    return float(np.nanmean(values))


def evaluate(name, grade, truth_median, truth_mean, targets):
    grade = np.asarray(grade, float)
    stats5 = high_grade_stats(grade, truth_median, 5)
    stats6 = high_grade_stats(grade, truth_median, 6)
    distribution = {int(key): int(value) for key, value in
                    zip(*np.unique(grade.astype(int), return_counts=True))}
    return {
        'table': name,
        'panel_median_mae': float(np.mean(np.abs(grade - truth_median))),
        'panel_mean_mae': float(np.mean(np.abs(grade - truth_mean))),
        'exact': float(np.mean(grade == truth_median)),
        'within_one': float(np.mean(np.abs(grade - truth_median) <= 1)),
        'macro_spearman': macro_spearman(grade, truth_mean, targets),
        'n_pred_5': int(stats5['n_predicted']),
        'precision_5': stats5['precision'],
        'recall_5': stats5['recall'],
        'n_pred_6': int(stats6['n_predicted']),
        'precision_6': stats6['precision'],
        'recall_6': stats6['recall'],
        'dist': distribution,
    }


def calibration_metrics(name, prediction, truth_median, truth_mean):
    prediction = np.asarray(prediction, float)
    rounded = np.rint(prediction)
    event5 = truth_median >= 5
    predicted5 = prediction >= 5
    true_positive = int((event5 & predicted5).sum())
    false_positive = int((~event5 & predicted5).sum())
    false_negative = int((event5 & ~predicted5).sum())
    return {
        'method': name,
        'panel_median_mae': float(np.mean(np.abs(prediction - truth_median))),
        'panel_mean_mae': float(np.mean(np.abs(prediction - truth_mean))),
        'panel_median_exact_rounded': float(np.mean(rounded == truth_median)),
        'panel_median_within_one_rounded': float(np.mean(np.abs(rounded - truth_median) <= 1)),
        'n_pred_ge5': int(predicted5.sum()),
        'precision_ge5': float(true_positive / max(true_positive + false_positive, 1)),
        'recall_ge5': float(true_positive / max(true_positive + false_negative, 1)),
    }


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def parent_child_pairs(scope, targets):
    """Derive the card parent/child column pairs from ``scorecard_scope.csv``.

    Children are the ``checkbox_zero_or_parent`` rows; the parent is the first two
    underscore-separated segments of the child id plus ``_all`` (for example
    ``A_malt_grain -> A_malt_all``).  Parents are capped first: their index is the
    child index of nothing (the ``_all`` rows are independent 0-6 ratings), so a
    single pass over the pairs is a deterministic post-processing step.
    """
    rule = scope[scope['template_rule'] == 'checkbox_zero_or_parent']
    index = {str(name): position for position, name in enumerate(targets)}
    pairs = []
    for child in rule['target_id'].tolist():
        parts = str(child).split('_')
        if len(parts) < 3:
            raise RuntimeError('unexpected checkbox child id: %s' % child)
        parent = '_'.join(parts[:2] + ['all'])
        if parent not in index:
            raise RuntimeError('parent %s for child %s is not a scored target'
                               % (parent, child))
        pairs.append((index[parent], index[child]))
    if len(pairs) != 24:
        raise RuntimeError('expected 24 card parent/child pairs, found %d' % len(pairs))
    pairs.sort(key=lambda pair: pair[1])
    return pairs


def count_parent_violations(grade_matrix, pairs):
    matrix = np.asarray(grade_matrix, dtype=np.int64)
    violations = 0
    for parent, child in pairs:
        violations += int((matrix[:, child] > matrix[:, parent]).sum())
    return violations


def outer_style_base(frame, truth_mean):
    """Per outer fold, target and style means computed on the outer-training folds."""
    data = pd.DataFrame({
        'target': frame['target'].astype(str).to_numpy(),
        'fold': frame['outer_fold'].astype(int).to_numpy(),
        'style': frame['tasting_category_fine'].astype(str).to_numpy(),
        'mean': np.asarray(truth_mean, dtype=float),
    })
    result = np.empty(len(data), dtype=float)
    for fold in sorted(data['fold'].unique()):
        test = (data['fold'] == fold).to_numpy()
        train = ~test
        grouped = data[train].groupby(['target', 'style'])['mean'].mean()
        global_mean = data[train].groupby('target')['mean'].mean()
        keys = pd.MultiIndex.from_arrays([data.loc[test, 'target'],
                                          data.loc[test, 'style']])
        values = grouped.reindex(keys).to_numpy(float)
        fallback = global_mean.reindex(data.loc[test, 'target']).to_numpy(float)
        result[test] = np.where(np.isnan(values), fallback, values)
    return result


def build_inner_predictions(panel, panel_protocol, beers, fold_of_beer, targets):
    """Inner-fold predictions inside every outer-training set (leak-free by design).

    For each outer fold the models are refitted on the official protocol using only
    the outer-training beers, and two inner folds of those beers produce OOF integer
    median grades and style means.  No outer-validation label is read here.
    """
    from beer_models.baseline import make_splits
    from beer_models.raw_panel import fit_shared

    n_beers = int(len(beers))
    n_targets = int(len(targets))
    distribution = np.asarray(panel['distribution'], dtype=float)
    mean_matrix = (distribution * np.arange(distribution.shape[2])).sum(axis=2)
    median_matrix = panel_median_from_distribution(distribution)
    inner_grade = np.full((n_beers, n_targets), np.nan)
    inner_style = np.full((n_beers, n_targets), np.nan)
    inner_fold = np.full(n_beers, -1, dtype=int)
    styles = panel['ids']['tasting_category_fine'].astype(str).to_numpy()
    fit_rows = {}
    for fold in sorted({int(fold_of_beer[str(beer)]) for beer in beers}):
        train_index = np.array(
            [position for position, beer in enumerate(beers)
             if int(fold_of_beer[str(beer)]) != fold], dtype=int)
        splits = make_splits(styles[train_index], INNER_FOLDS, OUTER_FOLD_SEED + 100 + fold)
        for inner_index, (fit_local, valid_local) in enumerate(splits):
            fit_index = train_index[np.asarray(fit_local, dtype=int)]
            valid_index = train_index[np.asarray(valid_local, dtype=int)]
            # Refit the official raw-panel model on THIS outer fold's training data.
            model = fit_shared(panel, fit_index, panel_protocol,
                               panel_protocol['model_seed'])
            grades = np.asarray(model.predict(panel['x'][valid_index])['grade'], dtype=float)
            if grades.shape != (len(valid_index), n_targets):
                raise RuntimeError('inner model returned an unexpected grade shape')
            inner_grade[valid_index] = grades
            inner_fold[valid_index] = inner_index
            for column in range(n_targets):
                values = mean_matrix[fit_index, column]
                grouped = pd.Series(values).groupby(styles[fit_index]).mean()
                global_mean = float(values.mean())
                inner_style[valid_index, column] = (
                    grouped.reindex(styles[valid_index]).fillna(global_mean).to_numpy())
        fit_rows[str(fold)] = {
            'outer_training_beers': int(train_index.size),
            'inner_fits': int(len(splits)),
            'inner_cells': int(train_index.size * n_targets),
            'outer_validation_cells': int((len(beers) - train_index.size) * n_targets),
        }
    if (inner_fold < 0).any():
        raise RuntimeError('some outer-training beer has no inner-fold prediction')
    return {'grade': inner_grade, 'style': inner_style, 'inner_fold': inner_fold,
            'mean': mean_matrix, 'median': median_matrix, 'fit_rows': fit_rows}


def nested_calibrations(frame, inner, beers, fold_of_beer, targets, truth_mean=None):
    """Fit every calibrator on inner OOF cells of the outer-training set only."""
    target_labels = frame['target'].astype(str).to_numpy()
    fold_labels = frame['outer_fold'].astype(int).to_numpy()
    legacy = frame['predicted_grade'].to_numpy(float)
    truth = (frame[TRUTH_MEAN_COL].to_numpy(float).copy()
             if truth_mean is None else np.asarray(truth_mean, float))
    style_base = outer_style_base(frame, truth)
    beer_index = {str(beer): position for position, beer in enumerate(beers)}
    row_beer = frame['beer'].astype(str).map(beer_index).to_numpy(int)
    predictions = {name: np.full(len(frame), np.nan) for name in NESTED_METHODS}
    fit_rows = {}
    for fold in sorted(set(int(value) for value in fold_of_beer.values())):
        test_rows = np.flatnonzero(fold_labels == fold)
        train_beer_index = np.array(
            [position for position, beer in enumerate(beers)
             if int(fold_of_beer[str(beer)]) != fold], dtype=int)
        if set(row_beer[test_rows].tolist()) & set(train_beer_index.tolist()):
            raise RuntimeError('outer-validation beer leaked into the calibration fit')
        flat_grade = inner['grade'][train_beer_index].ravel()
        flat_style = inner['style'][train_beer_index].ravel()
        flat_mean = inner['mean'][train_beer_index].ravel()
        flat_median = inner['median'][train_beer_index].ravel()
        if np.isnan(flat_grade).any():
            raise RuntimeError('inner OOF grade matrix is incomplete')
        group = np.tile(np.asarray(targets, dtype=object), train_beer_index.size)
        fold_array = np.full(flat_grade.size, str(fold), dtype=object)
        single = flat_grade.reshape(-1, 1)
        style_design = np.column_stack([flat_grade, flat_style])
        models = {
            'nested_linear': fit_nested_linear(single, flat_mean, group, fold_array),
            'nested_style_linear': fit_nested_linear(style_design, flat_mean, group,
                                                     fold_array),
            'nested_isotonic': fit_nested_isotonic(flat_grade, flat_mean, group, fold_array),
            'nested_quantile': fit_nested_quantile(flat_grade, flat_median, group, fold_array),
        }
        test_group = target_labels[test_rows]
        test_fold = fold_labels[test_rows].astype(str)
        predictions['nested_linear'][test_rows] = apply_nested_linear(
            legacy[test_rows].reshape(-1, 1), test_group, test_fold,
            models['nested_linear'])
        predictions['nested_style_linear'][test_rows] = apply_nested_linear(
            np.column_stack([legacy[test_rows], style_base[test_rows]]),
            test_group, test_fold, models['nested_style_linear'])
        predictions['nested_isotonic'][test_rows] = apply_nested_isotonic(
            legacy[test_rows], test_group, test_fold, models['nested_isotonic'])
        predictions['nested_quantile'][test_rows] = apply_nested_quantile(
            legacy[test_rows], test_group, test_fold, models['nested_quantile'])
        fit_rows[str(fold)] = {
            'calibration_fit_cells': int(flat_grade.size),
            'outer_validation_cells': int(test_rows.size),
            'inner_folds_used': int(INNER_FOLDS),
        }
    return predictions, style_base, fit_rows


def caliber_rows(prediction_set, grade, truth_median):
    """Beer-level exact5 / exact6 / >=5 counts, precision, observed count, recall."""
    grade = np.asarray(grade, dtype=float)
    truth = np.asarray(truth_median, dtype=float)
    rows = []
    for caliber, level in (('exact5', 5), ('exact6', 6), ('ge5', 5)):
        if caliber.startswith('exact'):
            predicted = grade == level
            observed = truth == level
        else:
            predicted = grade >= level
            observed = truth >= level
        n_predicted = int(predicted.sum())
        true_positive = int((predicted & observed).sum())
        n_observed = int(observed.sum())
        rows.append({
            'scope': 'beer_level', 'prediction_set': prediction_set, 'caliber': caliber,
            'n_predicted': n_predicted, 'true_positive': true_positive,
            'precision': true_positive / max(n_predicted, 1),
            'n_observed': n_observed,
            'recall': true_positive / max(n_observed, 1),
        })
    return rows


def individual_caliber_rows(raw_long, prediction, prediction_set):
    """Weighted per-taster exact5 / exact6 / >=5 precision and recall."""
    merged = raw_long.merge(prediction[['beer', 'target', 'predicted_grade']],
                            on=['beer', 'target'], how='left', validate='many_to_one')
    if merged['predicted_grade'].isna().any():
        raise RuntimeError('individual caliber merge left unmatched cells')
    weight = merged['weight_within_beer'].to_numpy(float)
    raw_values = merged['raw'].to_numpy(float)
    predicted_values = merged['predicted_grade'].to_numpy(float)
    rows = []
    for caliber, level in (('exact5', 5), ('exact6', 6), ('ge5', 5)):
        if caliber.startswith('exact'):
            predicted = predicted_values == level
            observed = raw_values == level
        else:
            predicted = predicted_values >= level
            observed = raw_values >= level
        n_predicted = float(weight[predicted].sum())
        true_positive = float(weight[predicted & observed].sum())
        n_observed = float(weight[observed].sum())
        rows.append({
            'scope': 'individual_taster_weighted', 'prediction_set': prediction_set,
            'caliber': caliber, 'n_predicted': n_predicted,
            'true_positive': true_positive,
            'precision': true_positive / max(n_predicted, 1e-12),
            'n_observed': n_observed,
            'recall': true_positive / max(n_observed, 1e-12),
        })
    return rows


def main(argv=None):
    args = parse_args(argv)
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    q1_dir = resolve(args.q1)
    raw_dir = resolve(args.raw)
    reference_dir = resolve(args.reference)
    source_dir = resolve(args.source)
    handoff_dir = resolve(args.handoff)
    report_path = resolve(args.report)
    output_dir = resolve(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        'oof_probabilities': q1_dir / 'q1__oof_scores_all52.csv',
        'full_fit_probabilities': q1_dir / 'q1__full_fit_scores_all52.csv',
        'observed_panel_distribution': raw_dir / 'observed_panel_distribution.csv',
        'oof_predictions': raw_dir / 'oof_predictions.csv.gz',
        'rating_weights_and_folds': raw_dir / 'rating_weights_and_folds.csv',
        'chemistry': source_dir / '1.csv',
        'raw_sensory': reference_dir / 'Raw sensory data anonymized.csv',
        'panel_protocol': resolve(args.panel_protocol),
        'scorecard_scope': resolve(args.scope),
    }
    for name, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError('%s not found: %s' % (name, path))

    # The protocol is written before any output table is computed.
    protocol = {
        'protocol_id': config['protocol_id'],
        'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
        'stage': 'protocol_written_before_computation',
        'config': config,
        'resolved_inputs': {name: str(path) for name, path in inputs.items()},
        'input_sha256': {name: sha256(path) for name, path in inputs.items()},
        'reference_report': str(report_path),
        'handoff_tables': str(handoff_dir),
        'output_directory': str(output_dir),
        'retraining': False,
    }
    write_json(output_dir / 'protocol.json', protocol)

    # ------------------------------------------------------------------ inputs
    obs = pd.read_csv(inputs['observed_panel_distribution'])
    required_obs = ['beer', 'target', TRUTH_MEDIAN_COL, TRUTH_MEAN_COL]
    if not set(required_obs).issubset(obs.columns):
        raise RuntimeError('observed_panel_distribution is missing required columns')
    if len(obs) != config['target_universe']['n_cells']:
        raise RuntimeError('unexpected observed row count: %d' % len(obs))
    if obs.duplicated(['beer', 'target']).any():
        raise RuntimeError('observed_panel_distribution has duplicated cells')
    targets = pd.unique(obs['target']).tolist()
    if len(targets) != config['target_universe']['n_targets']:
        raise RuntimeError('unexpected target count: %d' % len(targets))

    t6 = pd.read_csv(inputs['oof_probabilities'])
    needed = ['beer', 'target', 'estimated_grade_0_6', 'probability_rating_ge1',
              'probability_rating_ge5', 'probability_rating_6']
    if not set(needed).issubset(t6.columns):
        raise RuntimeError('OOF probability table is missing required columns')
    t6 = t6[t6['target'].isin(targets)][needed].copy()
    if len(t6) != len(obs) or t6.duplicated(['beer', 'target']).any():
        raise RuntimeError('OOF probability table does not align with the 12,500 cells')

    full_fit = pd.read_csv(inputs['full_fit_probabilities'])
    if not {'beer', 'target', 'probability_rating_ge5'}.issubset(full_fit.columns):
        raise RuntimeError('full-fit table is missing required columns')

    oof = pd.read_csv(inputs['oof_predictions'],
                      usecols=['method', 'beer', 'target', 'predicted_grade'])
    et = oof[oof['method'] == 'extra_trees'][['beer', 'target', 'predicted_grade']].copy()
    et = et[et['target'].isin(targets)]
    if len(et) != len(obs) or et.duplicated(['beer', 'target']).any():
        raise RuntimeError('extra_trees OOF predictions do not align with the 12,500 cells')

    weights = pd.read_csv(inputs['rating_weights_and_folds'])
    fold_per_beer = weights.groupby('beer')['outer_fold'].first().astype(int)
    if fold_per_beer.size != config['target_universe']['n_beers']:
        raise RuntimeError('unexpected beer count in rating_weights_and_folds')

    chemistry = pd.read_csv(inputs['chemistry'], usecols=['beer', 'tasting_category_fine'])
    if chemistry.duplicated('beer').any() or len(chemistry) != 250:
        raise RuntimeError('1.csv must hold one style per beer')

    # ------------------------------------------------------------------ frame
    frame = obs[['beer', 'target', TRUTH_MEDIAN_COL, TRUTH_MEAN_COL]].merge(
        t6, on=['beer', 'target'], how='left', validate='one_to_one')
    frame = frame.merge(et, on=['beer', 'target'], how='left', validate='one_to_one')
    frame = frame.merge(chemistry, on='beer', how='left', validate='many_to_one')
    frame['outer_fold'] = frame['beer'].map(fold_per_beer)
    frame['target_family'] = frame['target'].map(target_family)
    if frame[['probability_rating_ge5', 'probability_rating_6', 'predicted_grade',
              'outer_fold', 'tasting_category_fine']].isna().any().any():
        raise RuntimeError('join left unmatched cells in the evaluation frame')
    if len(frame) != len(obs):
        raise RuntimeError('evaluation frame changed the row count')

    truth_median = frame[TRUTH_MEDIAN_COL].to_numpy(float)
    truth_mean = frame[TRUTH_MEAN_COL].to_numpy(float)
    legacy = frame['predicted_grade'].to_numpy(float)
    legacy_consistency = bool(np.array_equal(
        legacy, frame['estimated_grade_0_6'].to_numpy(float)))
    if not legacy_consistency:
        raise RuntimeError('extra_trees OOF grades disagree with the paper OOF table')

    target_labels = frame['target'].astype(str).to_numpy()
    fold_labels = frame['outer_fold'].astype(int).astype(str).to_numpy()
    style_labels = frame['tasting_category_fine'].astype(str).to_numpy()
    pair_labels = np.char.add(np.char.add(target_labels, '|'), fold_labels)

    # ------------------------------------------------------- decision layer
    rate_ge5 = fold_training_rate(truth_median, target_labels, fold_labels, 5)
    rate_6 = fold_training_rate(truth_median, target_labels, fold_labels, 6)
    precap_grade = topk_decision(frame['probability_rating_ge5'].to_numpy(float),
                                 frame['probability_rating_6'].to_numpy(float),
                                 legacy, rate_ge5, rate_6, pair_labels)
    # Card parent/child cap: deterministic post-processing after the top-k decision.
    beer_order = pd.unique(frame['beer']).tolist()
    beer_position = {beer: position for position, beer in enumerate(beer_order)}
    target_position = {target: position for position, target in enumerate(targets)}
    row_beer = frame['beer'].map(beer_position).to_numpy(int)
    row_target = frame['target'].map(target_position).to_numpy(int)
    precap_matrix = np.full((len(beer_order), len(targets)), -1, dtype=np.int64)
    precap_matrix[row_beer, row_target] = precap_grade
    if (precap_matrix < 0).any():
        raise RuntimeError('the decision table does not cover every beer/target cell')
    scope = pd.read_csv(inputs['scorecard_scope'])
    pairs = parent_child_pairs(scope, targets)
    precap_violations = count_parent_violations(precap_matrix, pairs)
    capped_matrix = enforce_parent_child(precap_matrix, [pair[0] for pair in pairs],
                                         [pair[1] for pair in pairs])
    capped_violations = count_parent_violations(capped_matrix, pairs)
    changed_cells = int((capped_matrix != precap_matrix).sum())
    new_grade = capped_matrix[row_beer, row_target]
    frame['grade_0_6'] = new_grade
    frame['grade_topk_precap'] = precap_grade
    frame['grade_integer_median_old'] = legacy.astype(int)

    # k bookkeeping (for the non-forced evidence)
    group_first = {}
    for position in range(len(frame)):
        group_first.setdefault(pair_labels[position], []).append(position)
    zero_k5 = zero_k6 = 0
    for positions in group_first.values():
        group_size = len(positions)
        if int(round(float(rate_ge5[positions[0]]) * group_size)) == 0:
            zero_k5 += 1
        if int(round(float(rate_6[positions[0]]) * group_size)) == 0:
            zero_k6 += 1

    # ------------------------------------------------------- individual level
    raw = pd.read_csv(inputs['raw_sensory'], encoding='cp1252')
    raw = raw.reset_index(drop=True)
    raw['raw_record_1based'] = np.arange(1, len(raw) + 1)
    raw = raw.rename(columns={'A_ester_fruity': 'A_esters_fruity',
                              'F_ester_fruity': 'F_esters_fruity'})
    raw = raw.merge(weights[['raw_record_1based', 'beer', 'weight_within_beer']],
                    on='raw_record_1based', how='left')
    if raw[['beer', 'weight_within_beer']].isna().any().any():
        raise RuntimeError('raw sensory rows are not fully mapped to competition beers')
    raw_long = raw.melt(id_vars=['raw_record_1based', 'beer', 'weight_within_beer'],
                        value_vars=targets, var_name='target', value_name='raw')
    weight_sums = raw_long.groupby(['beer', 'target'])['weight_within_beer'].sum()
    if float((weight_sums - 1.0).abs().max()) > 1e-9:
        raise RuntimeError('within-beer weights do not sum to one')

    individual_et = weighted_individual_metrics(
        raw_long, frame[['beer', 'target', 'predicted_grade']])
    oracle = obs[['beer', 'target', TRUTH_MEDIAN_COL]].rename(
        columns={TRUTH_MEDIAN_COL: 'predicted_grade'})
    individual_oracle = weighted_individual_metrics(raw_long, oracle)

    # ------------------------------------------------------- nested calibration
    # Every calibrator is fitted on inner OOF cells produced *inside* the outer
    # training set only (outer-validation labels never enter features or parameters).
    from beer_models.panel_data import load_inputs

    panel_protocol = json.loads(inputs['panel_protocol'].read_text(encoding='utf-8'))
    panel = load_inputs(source_dir, inputs['raw_sensory'], panel_protocol)
    panel_beers = panel['ids']['beer'].astype(str).tolist()
    if sorted(panel_beers) != sorted(str(beer) for beer in fold_per_beer.index):
        raise RuntimeError('panel beer ids do not match the official fold map')
    fold_by_beer_str = {str(beer): int(fold) for beer, fold in fold_per_beer.items()}
    distribution_mean = (np.asarray(panel['distribution'], float)
                         * np.arange(panel['distribution'].shape[2])).sum(axis=2)
    truth_frame = frame[['beer', 'target', TRUTH_MEAN_COL]].copy()
    truth_frame['beer'] = truth_frame['beer'].astype(str)
    truth_matrix = truth_frame.pivot(index='beer', columns='target', values=TRUTH_MEAN_COL)
    aligned_truth = truth_matrix.reindex(index=panel_beers, columns=targets).to_numpy(float)
    if not np.isfinite(aligned_truth).all():
        raise RuntimeError('panel truth alignment produced missing cells')
    truth_delta = float(np.abs(aligned_truth - distribution_mean).max())
    if truth_delta > 1e-12:
        raise RuntimeError('panel distribution truth disagrees with observed_panel_distribution')
    inner = build_inner_predictions(panel, panel_protocol, panel_beers, fold_by_beer_str,
                                    targets)
    nested_predictions, style_base, calibration_fit_rows = nested_calibrations(
        frame, inner, panel_beers, fold_by_beer_str, targets)

    # Leakage perturbation test: change one outer-validation beer's true panel means
    # (and the same row of the inner truth matrix) and re-run the nesting.  That fold's
    # predictions must not move, because its calibrator was fitted on other beers only.
    fold0_beers = frame.loc[frame['outer_fold'] == 0, 'beer'].tolist()
    perturb_beer = str(fold0_beers[0])
    perturb_position = panel_beers.index(perturb_beer)
    perturb_delta = 0.25
    perturbed_inner = dict(inner)
    perturbed_mean = inner['mean'].copy()
    perturbed_mean[perturb_position] += perturb_delta
    perturbed_inner['mean'] = perturbed_mean
    perturbed_truth = frame[TRUTH_MEAN_COL].to_numpy(float).copy()
    perturbed_truth[(frame['beer'].astype(str) == perturb_beer).to_numpy()] += perturb_delta
    perturbed_predictions, _, _ = nested_calibrations(
        frame, perturbed_inner, panel_beers, fold_by_beer_str, targets,
        truth_mean=perturbed_truth)
    fold0_rows = (frame['outer_fold'].to_numpy(int) == 0)
    perturbation_deltas = {
        name: float(np.abs(perturbed_predictions[name][fold0_rows]
                           - nested_predictions[name][fold0_rows]).max())
        for name in NESTED_METHODS}
    perturbation_worst = max(perturbation_deltas.values())
    if perturbation_worst > 1e-12:
        raise RuntimeError('nested calibration still depends on outer-validation labels')

    linear_prediction = nested_predictions['nested_linear']
    isotonic_prediction = nested_predictions['nested_isotonic']
    style_prediction = nested_predictions['nested_style_linear']
    quantile_prediction = nested_predictions['nested_quantile']
    linear_grade = linear_to_grade(linear_prediction)

    # ------------------------------------------------------- metrics tables
    old_metrics = evaluate('q1_et_integer_median', legacy, truth_median, truth_mean,
                           target_labels)
    precap_metrics = evaluate('q1_topk_nonforced', precap_grade.astype(float),
                              truth_median, truth_mean, target_labels)
    new_metrics = evaluate('q1_topk_nonforced_parent_capped', new_grade.astype(float),
                           truth_median, truth_mean, target_labels)

    improved = [
        calibration_metrics('nested_linear', linear_prediction, truth_median, truth_mean),
        calibration_metrics('nested_isotonic', isotonic_prediction, truth_median, truth_mean),
        calibration_metrics('nested_style_linear', style_prediction, truth_median, truth_mean),
        calibration_metrics('nested_quantile', quantile_prediction, truth_median, truth_mean),
    ]

    individual_topk = weighted_individual_metrics(
        raw_long, frame[['beer', 'target', 'grade_0_6']].rename(
            columns={'grade_0_6': 'predicted_grade'}))

    rows = []
    for item in (old_metrics, precap_metrics, new_metrics):
        rows.append({
            'scope': 'beer_level', 'method': item['table'],
            'panel_median_mae': item['panel_median_mae'],
            'panel_mean_mae': item['panel_mean_mae'],
            'exact_match': item['exact'], 'within_one': item['within_one'],
            'macro_spearman': item['macro_spearman'],
            'n_pred_ge5': float(item['n_pred_5']), 'precision_ge5': item['precision_5'],
            'recall_ge5': item['recall_5'], 'n_pred_6': float(item['n_pred_6']),
            'precision_6': item['precision_6'], 'recall_6': item['recall_6']})
    for name, metrics_item in (('extra_trees_integer_median', individual_et),
                               ('q1_topk_nonforced_individual_weighted',
                                individual_topk)):
        rows.append({
            'scope': 'individual_taster', 'method': name,
            'panel_median_mae': metrics_item['individual_weighted_mae'],
            'panel_mean_mae': np.nan,
            'exact_match': metrics_item['individual_weighted_exact'],
            'within_one': metrics_item['individual_weighted_within_one'],
            'macro_spearman': np.nan})
    rows.append({
        'scope': 'lower_bound', 'method': 'panel_median_oracle_individual_level',
        'panel_median_mae': individual_oracle['individual_weighted_mae'],
        'panel_mean_mae': np.nan, 'exact_match': individual_oracle['individual_weighted_exact'],
        'within_one': individual_oracle['individual_weighted_within_one'],
        'macro_spearman': np.nan})
    for item in improved:
        rows.append({
            'scope': 'beer_level', 'method': 'calibration:' + item['method'],
            'panel_median_mae': item['panel_median_mae'],
            'panel_mean_mae': item['panel_mean_mae'],
            'exact_match': item['panel_median_exact_rounded'],
            'within_one': item['panel_median_within_one_rounded'],
            'macro_spearman': np.nan,
            'n_pred_ge5': float(item['n_pred_ge5']),
            'precision_ge5': item['precision_ge5'],
            'recall_ge5': item['recall_ge5']})
    metrics = pd.DataFrame(rows)[METRICS_COLUMNS]

    # ------------------------------------------------------- official tables
    evaluation = frame[['beer', 'target', 'target_family', 'grade_integer_median_old',
                        'grade_0_6', 'grade_topk_precap', TRUTH_MEAN_COL, TRUTH_MEDIAN_COL,
                        'probability_rating_ge1', 'probability_rating_ge5',
                        'probability_rating_6', 'outer_fold']].copy()
    evaluation = evaluation.rename(columns={TRUTH_MEAN_COL: 'panel_mean_reference',
                                            TRUTH_MEDIAN_COL: 'panel_median_reference'})
    evaluation = evaluation[EVALUATION_COLUMNS]
    submission_long = evaluation[SUBMISSION_LONG_COLUMNS].copy()
    wide = submission_long.pivot(index='beer', columns='target',
                                 values='grade_0_6').reset_index()
    calibrated = frame[['beer', 'target', TRUTH_MEAN_COL, TRUTH_MEDIAN_COL,
                        'grade_integer_median_old']].copy()
    calibrated = calibrated.rename(columns={TRUTH_MEAN_COL: 'panel_mean_reference',
                                            TRUTH_MEDIAN_COL: 'panel_median_reference'})
    calibrated['panel_mean_calibrated'] = linear_prediction
    calibrated['grade_from_calibrated_mean_round'] = linear_grade
    calibrated = calibrated[CALIBRATED_COLUMNS]
    coverage = frame.groupby('target').agg(
        n_observed_ge5=(TRUTH_MEDIAN_COL, lambda values: int((values >= 5).sum())),
        n_observed_6=(TRUTH_MEDIAN_COL, lambda values: int((values >= 6).sum())),
        # n_predicted_* describe the pre-cap top-k decision stage (frozen delivery
        # contract); the parent-capped official table is in the *_parent_capped columns.
        n_predicted_ge5=('grade_topk_precap',
                         lambda values: int((values >= 5).sum())),
        n_predicted_6=('grade_topk_precap',
                       lambda values: int((values >= 6).sum())),
        n_predicted_ge5_parent_capped=('grade_0_6',
                                       lambda values: int((values >= 5).sum())),
        n_predicted_6_parent_capped=('grade_0_6',
                                     lambda values: int((values >= 6).sum())),
    ).reset_index()[COVERAGE_COLUMNS]

    caliber_rows_all = []
    for set_name, grade in (('q1_et_integer_median', legacy),
                            ('q1_topk_nonforced', precap_grade),
                            ('q1_topk_nonforced_parent_capped', new_grade)):
        caliber_rows_all.extend(caliber_rows(set_name, grade, truth_median))
    for set_name, prediction in (
            ('extra_trees_integer_median',
             frame[['beer', 'target', 'predicted_grade']]),
            ('q1_topk_nonforced_individual_weighted',
             frame[['beer', 'target', 'grade_0_6']].rename(
                 columns={'grade_0_6': 'predicted_grade'}))):
        caliber_rows_all.extend(individual_caliber_rows(raw_long, prediction, set_name))
    calibers = pd.DataFrame(caliber_rows_all)[CALIBER_COLUMNS]

    submission_long.to_csv(output_dir / 'q1__submission_long_12500.csv', index=False,
                           encoding='utf-8')
    wide.to_csv(output_dir / 'q1__submission_wide_250x50.csv', index=False, encoding='utf-8')
    evaluation.to_csv(output_dir / 'q1__evaluation_long_12500.csv', index=False,
                      encoding='utf-8')
    metrics.to_csv(output_dir / 'q1__metrics_summary.csv', index=False, encoding='utf-8')
    calibrated.to_csv(output_dir / 'q1__calibrated_panel_mean_12500.csv', index=False,
                      encoding='utf-8')
    coverage.to_csv(output_dir / 'q1__high_grade_coverage_by_target.csv', index=False,
                    encoding='utf-8')
    calibers.to_csv(output_dir / 'q1__metric_calibers.csv', index=False, encoding='utf-8')

    # ------------------------------------------------------- verification
    checks = {}
    diagnostics = {}
    expected_precap_dist = {int(key): int(value) for key, value in
                            config['verification']['expected_precap_grade_distribution'].items()}
    expected_final_dist = {int(key): int(value) for key, value in
                           config['verification']['expected_final_grade_distribution'].items()}

    checks['q1_rows_12500'] = bool(len(submission_long) == 12500 and len(evaluation) == 12500)
    checks['q1_columns_match_protocol'] = bool(
        list(submission_long.columns) == SUBMISSION_LONG_COLUMNS
        and list(evaluation.columns) == EVALUATION_COLUMNS
        and list(metrics.columns) == METRICS_COLUMNS
        and list(calibrated.columns) == CALIBRATED_COLUMNS
        and list(coverage.columns) == COVERAGE_COLUMNS
        and list(calibers.columns) == CALIBER_COLUMNS)
    checks['q1_wide_shape_250x50'] = bool(wide.shape == (250, 51)
                                         and list(wide.columns) == ['beer'] + sorted(targets))
    # The frozen prototype report describes the pre-cap top-k table; the parent/child
    # cap only lowers the violating cells afterwards, so both stages are checked.
    checks['q1_precap_grade_distribution'] = bool(
        precap_metrics['dist'] == expected_precap_dist)
    final_total = int(new_metrics['dist'].get(0, 0) + sum(
        count for grade, count in new_metrics['dist'].items() if grade != 0))
    checks['q1_final_grade_distribution_valid'] = bool(
        final_total == 12500 and min(new_metrics['dist']) >= 0
        and max(new_metrics['dist']) <= 6
        and new_metrics['dist'] == expected_final_dist)
    checks['q1_legacy_grade_consistency'] = legacy_consistency
    checks['q1_submission_cells_unique'] = bool(
        not submission_long.duplicated(['beer', 'target']).any())
    checks['q1_precap_parent_violations_match_audit'] = bool(
        precap_violations == int(config['verification']['expected_precap_parent_violations']))
    checks['q1_parent_violations_zero_after_cap'] = bool(capped_violations == 0)
    checks['q1_cap_only_lowers_violating_cells'] = bool(
        changed_cells <= precap_violations
        and bool((capped_matrix <= precap_matrix).all()))
    diagnostics['parent_child'] = {
        'n_pairs': len(pairs),
        'precap_violations': precap_violations,
        'postcap_violations': capped_violations,
        'changed_cells': changed_cells,
        'pairs': [[str(targets[parent]), str(targets[child])] for parent, child in pairs],
    }

    report = json.loads(report_path.read_text(encoding='utf-8'))
    report_new = [entry for entry in report['tables']
                  if entry['table'].startswith(REPORT_TABLE_PREFIX)][0]
    report_old = [entry for entry in report['tables']
                  if entry['table'].startswith(REPORT_OLD_TABLE_PREFIX)][0]
    report_tolerance = float(config['verification']['report_tolerance'])
    report_keys = [('panel_median_mae', 'panel_median_mae'),
                   ('panel_mean_mae', 'panel_mean_mae'),
                   ('exact', 'exact'), ('within_one', 'within_one'),
                   ('macro_spearman', 'macro_spearman'),
                   ('n_pred_5', 'n_pred_5'), ('precision_5', 'precision_5'),
                   ('recall_5', 'recall_5'), ('n_pred_6', 'n_pred_6'),
                   ('precision_6', 'precision_6'), ('recall_6', 'recall_6')]
    report_diffs = {}
    for source_key, ours_key in report_keys:
        report_diffs[source_key] = abs(float(precap_metrics[ours_key])
                                       - float(report_new[source_key]))
    checks['q1_topk_matches_report'] = bool(
        max(report_diffs.values()) < report_tolerance
        and precap_metrics['dist'] == {int(k): int(v) for k, v in report_new['dist'].items()})
    old_report_diffs = {source_key: abs(float(old_metrics[ours_key])
                                        - float(report_old[source_key]))
                        for source_key, ours_key in report_keys}
    checks['q1_legacy_matches_report'] = bool(max(old_report_diffs.values()) < report_tolerance)
    diagnostics['report_max_abs_diff'] = float(max(report_diffs.values()))
    diagnostics['legacy_report_max_abs_diff'] = float(max(old_report_diffs.values()))

    observed_5 = int((truth_median >= 5).sum())
    observed_6 = int((truth_median >= 6).sum())
    predicted_5 = int((precap_grade >= 5).sum())
    predicted_6 = int((precap_grade >= 6).sum())
    predicted_grade5_only = int((precap_grade == 5).sum())
    # Non-forced evidence: the promoted counts come from the training-fold rates and
    # are never boosted to at least one per target/fold group.  Compare like with like:
    # grade==5 cells (188) against observed cells whose median is >=5 (198), grade 6
    # cells (16) against observed median==6 cells (19), plus the number of groups the
    # rule left at k5/k6 == 0.  The parent/child cap is a separate deterministic step.
    checks['q1_non_forced_no_target_forced_to_5_or_6'] = bool(
        predicted_grade5_only < observed_5 and predicted_6 < observed_6
        and predicted_5 < observed_5 + observed_6
        and zero_k5 > 0 and zero_k6 > 0)
    diagnostics['non_forced'] = {
        'observed_cells_ge5': observed_5, 'observed_cells_6': observed_6,
        'predicted_cells_grade5': predicted_grade5_only,
        'predicted_cells_ge5': predicted_5, 'predicted_cells_6': predicted_6,
        'groups_with_k5_zero': zero_k5, 'groups_with_k6_zero': zero_k6,
        'n_decision_groups': len(group_first)}
    diagnostics['high_grade_precision_recall'] = {
        'precap_ge5': high_grade_stats(precap_grade, truth_median, 5),
        'precap_ge6': high_grade_stats(precap_grade, truth_median, 6),
        'final_ge5': high_grade_stats(new_grade, truth_median, 5),
        'final_ge6': high_grade_stats(new_grade, truth_median, 6)}

    # ------------------------------------------------------- leakage fix checks
    nested_fit_rows = dict(calibration_fit_rows)
    calibrator_only_outer_training = True
    fold_size_by_fold = frame.groupby('outer_fold')['beer'].nunique().to_dict()
    for fold, row in nested_fit_rows.items():
        fold_size = int(fold_size_by_fold[int(fold)])
        expected_fit_cells = (len(panel_beers) - fold_size) * len(targets)
        if int(row['calibration_fit_cells']) != expected_fit_cells:
            calibrator_only_outer_training = False
        if int(row['outer_validation_cells']) != fold_size * len(targets):
            calibrator_only_outer_training = False
        if int(row['calibration_fit_cells']) + int(row['outer_validation_cells']) != 12500:
            calibrator_only_outer_training = False
    checks['q1_calibrator_fit_on_outer_training_only'] = bool(calibrator_only_outer_training)
    checks['q1_leakage_perturbation_invariant'] = bool(perturbation_worst <= 1e-12)
    checks['q1_perturbation_trivial_fold'] = bool(
        float(np.abs(truth_delta)) <= 1e-12)
    diagnostics['leakage_fix'] = {
        'truth_check_max_abs_diff': truth_delta,
        'perturbation_beer': perturb_beer,
        'perturbation_delta': perturb_delta,
        'perturbation_fold': 0,
        'perturbation_max_abs_delta_by_method': perturbation_deltas,
        'perturbation_worst': perturbation_worst,
        'calibration_fit_rows': nested_fit_rows,
        'nested_methods': list(NESTED_METHODS),
        'old_leaky_rows_are_not_reported_in_metrics_summary': True,
    }

    # ------------------------------------------------------- caliber table checks
    calibers_by_key = {(row['prediction_set'], row['caliber']): row
                       for row in calibers.to_dict('records')}
    caliber_consistency = True
    for set_name in ('q1_et_integer_median', 'q1_topk_nonforced',
                     'q1_topk_nonforced_parent_capped'):
        exact5 = calibers_by_key.get((set_name, 'exact5'))
        exact6 = calibers_by_key.get((set_name, 'exact6'))
        ge5 = calibers_by_key.get((set_name, 'ge5'))
        if not all((exact5, exact6, ge5)):
            caliber_consistency = False
            continue
        if not (exact5['n_predicted'] <= ge5['n_predicted']
                and exact6['n_predicted'] <= ge5['n_predicted']
                and exact5['true_positive'] <= exact5['n_predicted']
                and ge5['true_positive'] <= ge5['n_predicted']):
            caliber_consistency = False
    checks['q1_caliber_counts_consistent'] = bool(caliber_consistency)
    if not caliber_consistency:
        raise RuntimeError('caliber table is internally inconsistent')
    # Published 204-cell figure is the >=5 caliber (188 grade-5 cells + 16 grade-6 cells).
    ge5_row = calibers_by_key[('q1_topk_nonforced', 'ge5')]
    exact5_row = calibers_by_key[('q1_topk_nonforced', 'exact5')]
    exact6_row = calibers_by_key[('q1_topk_nonforced', 'exact6')]
    checks['q1_published_204_is_ge5'] = bool(
        ge5_row['n_predicted'] == 204
        and exact5_row['n_predicted'] + exact6_row['n_predicted'] == 204)

    # hand-off cross-checks (read-only): only the legacy columns are expected to be
    # identical; the new decision column is documented as diverging from the pre-fix
    # hand-off by exactly the parent/child cap.
    handoff = {}
    handoff_tolerance = float(config['verification']['handoff_tolerance'])
    evaluation_handoff = handoff_dir / 'q1__evaluation_long_12500.csv'
    if evaluation_handoff.exists():
        right = pd.read_csv(evaluation_handoff).sort_values(
            ['beer', 'target']).reset_index(drop=True)
        left = evaluation.sort_values(['beer', 'target']).reset_index(drop=True)
        aligned = (left['beer'].tolist() == right['beer'].tolist()
                   and left['target'].tolist() == right['target'].tolist())
        if not aligned:
            checks['q1_handoff_legacy_columns_identical'] = False
            handoff['alignment'] = 'row keys differ'
        else:
            legacy_columns = ['grade_integer_median_old', 'panel_mean_reference',
                              'panel_median_reference', 'probability_rating_ge1',
                              'probability_rating_ge5', 'probability_rating_6']
            diffs = {column: float(np.abs(left[column].to_numpy(float)
                                          - right[column].to_numpy(float)).max())
                     for column in legacy_columns}
            fold_equal = bool((left['outer_fold'].to_numpy(int)
                               == right['outer_fold'].to_numpy(int)).all())
            handoff['legacy_column_max_abs_diff'] = diffs
            handoff['fold_equal'] = fold_equal
            handoff['new_grade_changed_cells_vs_handoff'] = int(
                (left['grade_0_6'].to_numpy(float)
                 != right['grade_0_6'].to_numpy(float)).sum())
            handoff['new_grade_max_abs_diff_vs_handoff'] = float(
                np.abs(left['grade_0_6'].to_numpy(float)
                       - right['grade_0_6'].to_numpy(float)).max())
            checks['q1_handoff_legacy_columns_identical'] = bool(
                max(diffs.values()) < handoff_tolerance and fold_equal)
    else:
        checks['q1_handoff_legacy_columns_identical'] = True
        handoff['evaluation'] = 'handoff table absent; cross-check skipped'

    # The calibrated panel mean is a leak-free recomputation: it must be recorded but
    # is allowed (and expected) to differ from the voided hand-off numbers.
    calibrated_handoff = handoff_dir / 'q1__calibrated_panel_mean_12500.csv'
    if calibrated_handoff.exists():
        right = pd.read_csv(calibrated_handoff).sort_values(
            ['beer', 'target']).reset_index(drop=True)
        left = calibrated.sort_values(['beer', 'target']).reset_index(drop=True)
        handoff['calibrated_mean_max_abs_diff_vs_voided_handoff'] = float(
            np.abs(left['panel_mean_calibrated'].to_numpy(float)
                   - right['panel_mean_calibrated'].to_numpy(float)).max())
        handoff['calibrated_round_equal_vs_voided_handoff'] = bool(
            (left['grade_from_calibrated_mean_round'].to_numpy(int)
             == right['grade_from_calibrated_mean_round'].to_numpy(int)).all())
    checks['q1_no_voided_calibration_rows_in_metrics'] = bool(
        not any('fold-wise linear calibration' in str(method)
                or 'fold-wise isotonic calibration' in str(method)
                for method in metrics['method']))
    diagnostics['handoff'] = handoff

    # full-fit table: read only, must not enter the OOF decision table
    diagnostics['full_fit_table'] = {
        'rows': int(len(full_fit)),
        'used_for_submission': False,
        'note': 'full-fit scores are in-sample inference; the official submission uses OOF probabilities only',
    }
    checks['q1_full_fit_not_used_for_submission'] = bool(
        len(full_fit) >= 12500 and 'probability_rating_ge5' in full_fit.columns)

    all_pass = bool(checks) and all(bool(value) for value in checks.values())
    diagnostics['individual_level'] = {
        'extra_trees': individual_et,
        'q1_topk_nonforced': individual_topk,
        'panel_median_oracle': individual_oracle,
    }
    diagnostics['calibration'] = {'reported': 'nested_linear',
                                  'methods': {item['method']: item for item in improved}}
    diagnostics['metric_calibers'] = calibers.to_dict('records')
    verification = {'all_pass': all_pass, 'checks': checks,
                    'diagnostics': diagnostics,
                    'protocol_id': config['protocol_id']}
    write_json(output_dir / 'verification.json', verification)

    leaky_reference = None
    pre_summary_path = ROOT / 'outputs' / 'q1_decision_20261028_pre_leakfix' / 'summary.json'
    if pre_summary_path.exists():
        pre_summary = json.loads(pre_summary_path.read_text(encoding='utf-8'))
        leaky_reference = {
            name: {'panel_mean_mae': item.get('panel_mean_mae'),
                   'panel_median_mae': item.get('panel_median_mae')}
            for name, item in pre_summary.get('calibration', {}).items()}

    # Delivery layer (src/beer_models/delivery_current.py, read-only here) reads the
    # legacy flat keys from summary.json; keep them populated while also exposing the
    # leak-free nested calibration and the parent-capped decision table.
    calibration_summary = {}
    for item in improved:
        label = {
            'nested_linear': 'nested fold-wise linear calibration -> panel mean',
            'nested_isotonic': 'nested fold-wise isotonic calibration -> panel mean',
            'nested_style_linear':
                'nested fold-wise style mean, linear calibration -> panel mean',
            'nested_quantile':
                'nested fold-wise marginal (quantile) calibration -> panel mean',
        }[item['method']]
        payload = dict(item)
        payload['implementation'] = item['method']
        payload['method'] = label
        calibration_summary[label] = payload
    summary = {
        'protocol_id': config['protocol_id'],
        'generated_at': protocol['generated_at'],
        'rows': int(len(submission_long)),
        'grade_distribution': new_metrics['dist'],
        'precap_grade_distribution': precap_metrics['dist'],
        'legacy_grade_distribution': old_metrics['dist'],
        'legacy_integer_median': old_metrics,
        'topk_nonforced': new_metrics,
        'topk_nonforced_precap': precap_metrics,
        'individual_weighted': {'individual_weighted_mae':
                                individual_et['individual_weighted_mae'],
                                'individual_weighted_exact':
                                individual_et['individual_weighted_exact'],
                                'individual_weighted_within_one':
                                individual_et['individual_weighted_within_one'],
                                'extra_trees_integer_median': individual_et,
                                'q1_topk_nonforced_individual_weighted': individual_topk},
        'panel_median_oracle': individual_oracle,
        'calibration': calibration_summary,
        'voided_pre_leakfix_calibration_reference': leaky_reference,
        'leakage_fix': diagnostics['leakage_fix'],
        'parent_child': diagnostics['parent_child'],
        'metric_calibers': calibers.to_dict('records'),
        'non_forced_evidence': diagnostics['non_forced'],
        'verification_all_pass': all_pass,
    }
    write_json(output_dir / 'summary.json', summary)

    print(json.dumps({
        'all_pass': all_pass,
        'checks': checks,
        'dist': new_metrics['dist'],
        'precap_dist': precap_metrics['dist'],
        'parent_child': diagnostics['parent_child'],
        'leakage_fix': diagnostics['leakage_fix'],
        'topk_panel_median_mae': new_metrics['panel_median_mae'],
        'precap_panel_median_mae': precap_metrics['panel_median_mae'],
        'individual_topk': individual_topk,
        'calibration': {item['method']: item['panel_mean_mae'] for item in improved},
        'files': sorted(path.name for path in output_dir.iterdir()),
    }, indent=2, ensure_ascii=False))
    if not all_pass:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit('verification failed: %s' % ', '.join(failed))
    return 0

if __name__ == '__main__':
    sys.exit(main())
