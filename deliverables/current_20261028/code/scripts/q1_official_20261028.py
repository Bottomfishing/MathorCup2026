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
    calibrate_panel_mean,
    fold_style_mean,
    fold_training_rate,
    high_grade_stats,
    isotonic_calibrate,
    linear_to_grade,
    quantile_calibrate,
    topk_decision,
)

TRUTH_MEDIAN_COL = 'panel_weighted_median'
TRUTH_MEAN_COL = 'panel_weighted_mean'
REPORT_TABLE_PREFIX = 'NEW: top-k cumulative-prob (fold-wise rate)'
REPORT_OLD_TABLE_PREFIX = 'OLD: ET integer conditional median'

SUBMISSION_LONG_COLUMNS = ['beer', 'target', 'target_family', 'grade_0_6',
                           'probability_rating_ge5', 'probability_rating_6']
EVALUATION_COLUMNS = ['beer', 'target', 'target_family', 'grade_integer_median_old',
                      'grade_0_6', 'panel_mean_reference', 'panel_median_reference',
                      'probability_rating_ge1', 'probability_rating_ge5',
                      'probability_rating_6', 'outer_fold']
CALIBRATED_COLUMNS = ['beer', 'target', 'panel_mean_reference', 'panel_median_reference',
                      'grade_integer_median_old', 'panel_mean_calibrated',
                      'grade_from_calibrated_mean_round']
METRICS_COLUMNS = ['scope', 'method', 'panel_median_mae', 'panel_mean_mae', 'exact_match',
                   'within_one', 'macro_spearman', 'n_pred_ge5', 'precision_ge5',
                   'recall_ge5', 'n_pred_6', 'precision_6', 'recall_6']
COVERAGE_COLUMNS = ['target', 'n_observed_ge5', 'n_observed_6', 'n_predicted_ge5',
                    'n_predicted_6']


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
    new_grade = topk_decision(frame['probability_rating_ge5'].to_numpy(float),
                              frame['probability_rating_6'].to_numpy(float),
                              legacy, rate_ge5, rate_6, pair_labels)
    frame['grade_0_6'] = new_grade
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

    # ------------------------------------------------------- calibration
    linear_prediction, linear_coefficients = calibrate_panel_mean(
        legacy.reshape(-1, 1), truth_mean, target_labels, fold_labels)
    isotonic_prediction = isotonic_calibrate(legacy, truth_mean, target_labels, fold_labels)
    style_base = fold_style_mean(truth_mean, style_labels, target_labels, fold_labels)
    style_prediction, style_coefficients = calibrate_panel_mean(
        np.column_stack([legacy, style_base]), truth_mean, target_labels, fold_labels)
    quantile_prediction = quantile_calibrate(legacy, truth_median, target_labels, fold_labels)
    linear_grade = linear_to_grade(linear_prediction)

    # ------------------------------------------------------- metrics tables
    old_metrics = evaluate('q1_et_integer_median', legacy, truth_median, truth_mean, target_labels)
    new_metrics = evaluate('q1_topk_nonforced', new_grade.astype(float),
                           truth_median, truth_mean, target_labels)

    improved = [
        calibration_metrics('ET integer median (current)', legacy, truth_median, truth_mean),
        calibration_metrics('ET integer + fold-wise linear calibration -> panel mean',
                            linear_prediction, truth_median, truth_mean),
        calibration_metrics('ET integer + fold-wise isotonic calibration -> panel mean',
                            isotonic_prediction, truth_median, truth_mean),
        calibration_metrics('ET integer + style mean, fold-wise linear',
                            style_prediction, truth_median, truth_mean),
        calibration_metrics('ET integer + fold-wise marginal (quantile) calibration',
                            quantile_prediction, truth_median, truth_mean),
    ]

    rows = []
    for item in (old_metrics, new_metrics):
        rows.append({
            'scope': 'beer_level', 'method': item['table'],
            'panel_median_mae': item['panel_median_mae'],
            'panel_mean_mae': item['panel_mean_mae'],
            'exact_match': item['exact'], 'within_one': item['within_one'],
            'macro_spearman': item['macro_spearman'],
            'n_pred_ge5': float(item['n_pred_5']), 'precision_ge5': item['precision_5'],
            'recall_ge5': item['recall_5'], 'n_pred_6': float(item['n_pred_6']),
            'precision_6': item['precision_6'], 'recall_6': item['recall_6']})
    rows.append({
        'scope': 'individual_taster', 'method': 'extra_trees_integer_median',
        'panel_median_mae': individual_et['individual_weighted_mae'],
        'panel_mean_mae': np.nan, 'exact_match': individual_et['individual_weighted_exact'],
        'within_one': individual_et['individual_weighted_within_one'],
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
                        'grade_0_6', TRUTH_MEAN_COL, TRUTH_MEDIAN_COL,
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
        n_predicted_ge5=('grade_0_6', lambda values: int((values >= 5).sum())),
        n_predicted_6=('grade_0_6', lambda values: int((values >= 6).sum())),
    ).reset_index()[COVERAGE_COLUMNS]

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

    # ------------------------------------------------------- verification
    checks = {}
    diagnostics = {}
    expected_dist = {int(key): int(value) for key, value in
                     config['verification']['expected_grade_distribution'].items()}

    checks['q1_rows_12500'] = bool(len(submission_long) == 12500 and len(evaluation) == 12500)
    checks['q1_columns_match_protocol'] = bool(
        list(submission_long.columns) == SUBMISSION_LONG_COLUMNS
        and list(evaluation.columns) == EVALUATION_COLUMNS
        and list(metrics.columns) == METRICS_COLUMNS
        and list(calibrated.columns) == CALIBRATED_COLUMNS
        and list(coverage.columns) == COVERAGE_COLUMNS)
    checks['q1_wide_shape_250x50'] = bool(wide.shape == (250, 51)
                                         and list(wide.columns) == ['beer'] + sorted(targets))
    checks['q1_grade_distribution'] = bool(new_metrics['dist'] == expected_dist)
    checks['q1_legacy_grade_consistency'] = legacy_consistency
    checks['q1_submission_cells_unique'] = bool(
        not submission_long.duplicated(['beer', 'target']).any())

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
        report_diffs[source_key] = abs(float(new_metrics[ours_key])
                                       - float(report_new[source_key]))
    checks['q1_topk_matches_report'] = bool(
        max(report_diffs.values()) < report_tolerance
        and new_metrics['dist'] == {int(k): int(v) for k, v in report_new['dist'].items()})
    old_report_diffs = {source_key: abs(float(old_metrics[ours_key])
                                        - float(report_old[source_key]))
                        for source_key, ours_key in report_keys}
    checks['q1_legacy_matches_report'] = bool(max(old_report_diffs.values()) < report_tolerance)
    diagnostics['report_max_abs_diff'] = float(max(report_diffs.values()))
    diagnostics['legacy_report_max_abs_diff'] = float(max(old_report_diffs.values()))

    observed_5 = int((truth_median >= 5).sum())
    observed_6 = int((truth_median >= 6).sum())
    predicted_5 = int((new_grade >= 5).sum())
    predicted_6 = int((new_grade >= 6).sum())
    predicted_grade5_only = int((new_grade == 5).sum())
    # Non-forced evidence: the promoted counts come from the training-fold rates and
    # are never boosted to at least one per target/fold group.  Compare like with like:
    # grade==5 cells (188) against observed cells whose median is >=5 (198), grade 6
    # cells (16) against observed median==6 cells (19), plus the number of groups the
    # rule left at k5/k6 == 0.
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
        'ge5': high_grade_stats(new_grade, truth_median, 5),
        'ge6': high_grade_stats(new_grade, truth_median, 6)}

    # hand-off cross-checks (read-only)
    handoff = {}
    metrics_handoff = handoff_dir / 'q1__metrics_summary.csv'
    submission_handoff = handoff_dir / 'q1__submission_long_12500.csv'
    evaluation_handoff = handoff_dir / 'q1__evaluation_long_12500.csv'
    calibrated_handoff = handoff_dir / 'q1__calibrated_panel_mean_12500.csv'
    coverage_handoff = handoff_dir / 'q1__high_grade_coverage_by_target.csv'
    handoff_tolerance = float(config['verification']['handoff_tolerance'])
    rounded_tolerance = float(config['verification']['handoff_rounded_tolerance'])
    # The protocol stores rounded rows as "scope,method"; normalise both sides to the
    # same "scope||method" key used for the table comparison below.
    rounded_rows = set()
    for entry in config['verification']['handoff_rounded_rows']:
        entry_scope, entry_method = entry.split(',', 1)
        rounded_rows.add(entry_scope.strip() + '||' + entry_method.strip())

    if submission_handoff.exists():
        left = submission_long.sort_values(['beer', 'target']).reset_index(drop=True)
        right = pd.read_csv(submission_handoff).sort_values(
            ['beer', 'target']).reset_index(drop=True)
        handoff['submission_rows'] = [len(left), len(right)]
        handoff['submission_grade_max_abs_diff'] = float(
            np.abs(left['grade_0_6'].to_numpy(float) - right['grade_0_6'].to_numpy(float)).max())
        handoff['submission_ge5_max_abs_diff'] = float(
            np.abs(left['probability_rating_ge5'].to_numpy(float)
                   - right['probability_rating_ge5'].to_numpy(float)).max())
        handoff['submission_6_max_abs_diff'] = float(
            np.abs(left['probability_rating_6'].to_numpy(float)
                   - right['probability_rating_6'].to_numpy(float)).max())
        checks['q1_submission_matches_handoff'] = bool(
            left['beer'].tolist() == right['beer'].tolist()
            and left['target'].tolist() == right['target'].tolist()
            and handoff['submission_grade_max_abs_diff'] == 0.0
            and handoff['submission_ge5_max_abs_diff'] < handoff_tolerance
            and handoff['submission_6_max_abs_diff'] < handoff_tolerance)
    else:
        checks['q1_submission_matches_handoff'] = True
        handoff['submission'] = 'handoff table absent; cross-check skipped'

    if evaluation_handoff.exists():
        left = evaluation.sort_values(['beer', 'target']).reset_index(drop=True)
        right = pd.read_csv(evaluation_handoff).sort_values(
            ['beer', 'target']).reset_index(drop=True)
        handoff['evaluation_grade_max_abs_diff'] = float(
            np.abs(left['grade_integer_median_old'].to_numpy(float)
                   - right['grade_integer_median_old'].to_numpy(float)).max())
        handoff['evaluation_new_grade_max_abs_diff'] = float(
            np.abs(left['grade_0_6'].to_numpy(float) - right['grade_0_6'].to_numpy(float)).max())
        handoff['evaluation_panel_mean_max_abs_diff'] = float(
            np.abs(left['panel_mean_reference'].to_numpy(float)
                   - right['panel_mean_reference'].to_numpy(float)).max())
        handoff['evaluation_fold_equal'] = bool(
            (left['outer_fold'].to_numpy(int) == right['outer_fold'].to_numpy(int)).all())
        checks['q1_evaluation_matches_handoff'] = bool(
            handoff['evaluation_grade_max_abs_diff'] == 0.0
            and handoff['evaluation_new_grade_max_abs_diff'] == 0.0
            and handoff['evaluation_panel_mean_max_abs_diff'] < handoff_tolerance
            and handoff['evaluation_fold_equal'])
    else:
        checks['q1_evaluation_matches_handoff'] = True
        handoff['evaluation'] = 'handoff table absent; cross-check skipped'

    if calibrated_handoff.exists():
        left = calibrated.sort_values(['beer', 'target']).reset_index(drop=True)
        right = pd.read_csv(calibrated_handoff).sort_values(
            ['beer', 'target']).reset_index(drop=True)
        handoff['calibrated_mean_max_abs_diff'] = float(
            np.abs(left['panel_mean_calibrated'].to_numpy(float)
                   - right['panel_mean_calibrated'].to_numpy(float)).max())
        handoff['calibrated_round_equal'] = bool(
            (left['grade_from_calibrated_mean_round'].to_numpy(int)
             == right['grade_from_calibrated_mean_round'].to_numpy(int)).all())
        checks['q1_calibrated_matches_handoff'] = bool(
            handoff['calibrated_mean_max_abs_diff'] < handoff_tolerance
            and handoff['calibrated_round_equal'])
    else:
        checks['q1_calibrated_matches_handoff'] = True
        handoff['calibrated'] = 'handoff table absent; cross-check skipped'

    if coverage_handoff.exists():
        right = pd.read_csv(coverage_handoff).sort_values('target').reset_index(drop=True)
        left = coverage.sort_values('target').reset_index(drop=True)
        handoff['coverage_equal'] = bool(left.equals(right))
        checks['q1_coverage_matches_handoff'] = handoff['coverage_equal']
    else:
        checks['q1_coverage_matches_handoff'] = True
        handoff['coverage'] = 'handoff table absent; cross-check skipped'

    if metrics_handoff.exists():
        right = pd.read_csv(metrics_handoff)
        right_key = right['scope'].astype(str) + '||' + right['method'].astype(str)
        left_key = metrics['scope'].astype(str) + '||' + metrics['method'].astype(str)
        right = right.assign(_key=right_key).set_index('_key')
        left = metrics.assign(_key=left_key).set_index('_key')
        numeric_columns = [column for column in METRICS_COLUMNS
                           if column not in ('scope', 'method')]
        row_comparison = {}
        all_ok = bool(set(left.index) == set(right.index))
        worst = 0.0
        for key in sorted(set(left.index) & set(right.index)):
            tolerance = rounded_tolerance if key in rounded_rows else handoff_tolerance
            row_ok = True
            row_diff = 0.0
            for column in numeric_columns:
                left_value = left.loc[key, column]
                right_value = right.loc[key, column]
                if pd.isna(left_value) and pd.isna(right_value):
                    continue
                if pd.isna(left_value) != pd.isna(right_value):
                    row_ok = False
                    row_diff = float('inf')
                    continue
                diff = abs(float(left_value) - float(right_value))
                row_diff = max(row_diff, diff)
                if diff > tolerance:
                    row_ok = False
            row_comparison[key] = {'pass': row_ok, 'max_abs_diff': row_diff,
                                   'tolerance': tolerance}
            worst = max(worst, row_diff) if np.isfinite(row_diff) else float('inf')
            all_ok = all_ok and row_ok
        checks['q1_metrics_match_handoff'] = all_ok
        diagnostics['handoff_metrics_worst_abs_diff'] = worst
        diagnostics['handoff_metrics_rows'] = row_comparison
    else:
        checks['q1_metrics_match_handoff'] = True
        handoff['metrics'] = 'handoff table absent; cross-check skipped'
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
    diagnostics['individual_level'] = {'extra_trees': individual_et,
                                       'panel_median_oracle': individual_oracle}
    diagnostics['calibration_coefficients'] = {
        'linear': {'groups': [str(value) for value in linear_coefficients['groups']],
                   'folds': [str(value) for value in linear_coefficients['folds']],
                   'intercept': linear_coefficients['intercept'].tolist(),
                   'slope': linear_coefficients['slope'].tolist(),
                   'n_train': linear_coefficients['n_train'].tolist()},
        'style_linear': {'intercept': style_coefficients['intercept'].tolist(),
                         'slope': style_coefficients['slope'].tolist()},
    }
    verification = {'all_pass': all_pass, 'checks': checks,
                    'diagnostics': diagnostics,
                    'protocol_id': config['protocol_id']}
    write_json(output_dir / 'verification.json', verification)

    summary = {
        'protocol_id': config['protocol_id'],
        'generated_at': protocol['generated_at'],
        'rows': int(len(submission_long)),
        'grade_distribution': new_metrics['dist'],
        'legacy_grade_distribution': old_metrics['dist'],
        'legacy_integer_median': old_metrics,
        'topk_nonforced': new_metrics,
        'individual_weighted': individual_et,
        'panel_median_oracle': individual_oracle,
        'calibration': {item['method']: item for item in improved},
        'non_forced_evidence': diagnostics['non_forced'],
        'verification_all_pass': all_pass,
    }
    write_json(output_dir / 'summary.json', summary)

    print(json.dumps({'all_pass': all_pass, 'checks': checks,
                      'dist': new_metrics['dist'],
                      'topk_panel_median_mae': new_metrics['panel_median_mae'],
                      'topk_precision_5': new_metrics['precision_5'],
                      'topk_recall_5': new_metrics['recall_5'],
                      'topk_precision_6': new_metrics['precision_6'],
                      'topk_recall_6': new_metrics['recall_6'],
                      'linear_panel_mean_mae': improved[1]['panel_mean_mae'],
                      'files': sorted(path.name for path in output_dir.iterdir())},
                     indent=2, ensure_ascii=False))
    if not all_pass:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit('verification failed: %s' % ', '.join(failed))
    return 0


if __name__ == '__main__':
    sys.exit(main())
