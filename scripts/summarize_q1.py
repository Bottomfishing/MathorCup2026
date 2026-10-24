"""Summarize saved OOF grade accuracy without refitting or tuning models."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.final_model import relative_grades


def run(q1, output):
    manifest = json.loads((q1 / 'run_manifest.json').read_text(encoding='utf-8'))
    targets = manifest['targets']
    scores = pd.read_csv(q1 / 'q1_scores_all50_long.csv', dtype={'beer': str})
    splits = pd.read_csv(q1 / 'splits.csv', dtype={'beer': str})
    primary = pd.read_csv(q1 / 'grade_metrics.csv').set_index('target').loc[targets, 'primary'].to_numpy(bool)
    oof = pd.read_csv(q1 / 'oof_predictions.csv', dtype={'beer': str})
    calibration = {row['fold']: np.array(row['sorted_reference']) for row in json.loads((q1 / 'calibration.json').read_text(encoding='utf-8'))}
    rows = []; confusion = np.zeros((7, 7), int)
    for name, group in oof.groupby('model', sort=False):
        grades = np.empty((len(splits), len(targets)), int)
        truth = np.empty_like(grades)
        for fold, reference in calibration.items():
            idx = np.flatnonzero(splits.outer_test_fold.eq(fold))
            beers = splits.beer.iloc[idx]
            selected = group.loc[group.fold.eq(fold)]
            values = selected.pivot(index='beer', columns='target', values='prediction').loc[beers, targets].to_numpy()
            observed = selected.pivot(index='beer', columns='target', values='observed').loc[beers, targets].to_numpy()
            grades[idx] = relative_grades(reference, values)
            truth[idx] = relative_grades(reference, observed)
        error = abs(grades[:, primary] - truth[:, primary])
        rows.append({'model': name, 'n_samples': len(splits), 'n_targets': int(primary.sum()),
                     'relative_grade_mae': error.mean(), 'exact_accuracy': (error == 0).mean(),
                     'within_one_accuracy': (error <= 1).mean()})
        if name == 'nested_selected':
            actual = scores.pivot(index='beer', columns='target', values='relative_grade_0_6').loc[splits.beer, targets]
            np.testing.assert_array_equal(grades, actual)
            np.add.at(confusion, (truth[:, primary].ravel(), grades[:, primary].ravel()), 1)
    comparison = pd.DataFrame(rows)
    nested = comparison.set_index('model').loc['nested_selected']
    existing = pd.read_csv(q1 / 'grade_metrics.csv'); existing = existing.loc[existing.primary]
    np.testing.assert_allclose(nested.relative_grade_mae, existing.mae_levels.mean(), atol=1e-12)
    np.testing.assert_allclose(nested.exact_accuracy, existing.exact_accuracy.mean(), atol=1e-12)
    np.testing.assert_allclose(nested.within_one_accuracy, existing.within_one_accuracy.mean(), atol=1e-12)
    stress = pd.read_csv(q1 / 'missingness.csv')
    summary = stress.groupby('additional_missing_rate').agg(
        macro_mae_mean=('macro_mae', 'mean'), macro_mae_sd=('macro_mae', 'std'),
        relative_grade_mae_mean=('relative_grade_mae', 'mean'),
        grade_change_fraction_mean=('grade_change_fraction', 'mean'),
        within_one_accuracy_mean=('within_one_accuracy', 'mean')).reset_index()
    baseline = pd.read_csv(q1 / 'model_comparison.csv').set_index('model').loc['nested_selected', 'macro_mae']
    summary['mae_increase_fraction'] = summary.macro_mae_mean / baseline - 1
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output, 'grade_model_comparison.csv', comparison)
    save_csv(output, 'missingness_summary.csv', summary)
    save_csv(output, 'grade_confusion_primary45.csv', pd.DataFrame(confusion, columns=[f'predicted_{i}' for i in range(7)]).assign(observed_relative_grade=range(7)))
    report = {'status': 'passed', 'primary_grade_metrics': nested.to_dict(),
        'confusion_count': int(confusion.sum()), 'grade_scale': 'relative_to_outer_training_labels',
        'source_sha256': {n: sha256(q1 / n) for n in ['oof_predictions.csv', 'q1_scores_all50_long.csv', 'missingness.csv', 'calibration.json']},
        'interpretation': 'Accuracy concerns constructed relative grades, not recovered original panel ratings. Missingness SD uses 3 mask repetitions, not a confidence interval.'}
    write_json(output / 'summary.json', report)
    print(comparison.to_string(index=False)); print(summary.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--q1', type=Path, default=Path('outputs/final_20261024/q1'))
    p.add_argument('--output', type=Path, default=Path('outputs/q1_delivery_20261024'))
    args = p.parse_args(); run(args.q1, args.output)
