"""Test direct relative-grade learning against the frozen continuous-then-grade baseline."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from beer_audit.audit import sha256, save_csv, write_json
from beer_models.baseline import mask_observed
from beer_models.explain import stratified_draw
from beer_models.final_model import relative_grades
from beer_models.ordinal import grade_decisions, seven_level_probabilities
from complete_modeling import load_q1_artifacts


def metrics(prediction, truth, primary):
    error = abs(prediction[:, primary] - truth[:, primary])
    return {'grade_mae': float(error.mean()), 'exact_accuracy': float((error == 0).mean()),
            'within_one_accuracy': float((error <= 1).mean())}


def run(source, q1, output):
    config_path = Path('configs/q1_ordinal_protocol.json')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    ids, xf, yf, bundle, splits = load_q1_artifacts(source, Path('outputs/p0_audit_20261023_final'), q1)
    x, y = xf.to_numpy(float), yf.to_numpy(float); primary = bundle['primary']
    scores = pd.read_csv(q1 / 'q1_scores_all50_long.csv', dtype={'beer': str})
    baseline = scores.pivot(index='beer', columns='target', values='relative_grade_0_6').loc[ids.beer.astype(str), yf.columns].to_numpy(int)
    truth = np.empty_like(baseline); median = np.empty_like(baseline); mode = np.empty_like(baseline)
    naive = np.empty_like(baseline); probability = np.empty((len(x), len(yf.columns), 7))
    missing = {(r, k, rule): np.empty_like(baseline) for r in config['missing_rates']
               for k in range(config['missing_repeats']) for rule in ['conditional_median', 'conditional_mode']}
    for fold, (train, test) in enumerate(splits):
        train_grades = relative_grades(y[train], y[train]); truth[test] = relative_grades(y[train], y[test])
        naive[test] = np.quantile(train_grades, .5, axis=0, method='inverted_cdf')
        classifier = ExtraTreesClassifier(n_estimators=config['n_estimators'], min_samples_leaf=config['min_samples_leaf'],
            max_features=config['max_features'], random_state=config['seed'] + fold, n_jobs=1)
        pipe = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('model', classifier)]).fit(x[train], train_grades)
        clean = pipe.named_steps['impute'].transform(x[test])
        probability[test] = seven_level_probabilities(classifier, clean)
        decisions = grade_decisions(probability[test]); median[test] = decisions['conditional_median']; mode[test] = decisions['conditional_mode']
        for rate in config['missing_rates']:
            for repeat in range(config['missing_repeats']):
                masked, _ = mask_observed(x[test], rate, config['seed'] + 1000 + fold * 100 + repeat)
                decisions = grade_decisions(seven_level_probabilities(classifier, pipe.named_steps['impute'].transform(masked)))
                for rule, values in decisions.items():
                    missing[rate, repeat, rule][test] = values
        print(f'Ordinal fold {fold} complete', flush=True)
    original_truth = scores.pivot(index='beer', columns='target', values='observed_relative_grade').loc[ids.beer.astype(str), yf.columns]
    np.testing.assert_array_equal(truth, original_truth)
    predictions = {'training_median': naive, 'continuous_then_grade': baseline, 'conditional_median': median, 'conditional_mode': mode}
    comparison = [{'method': name, **metrics(pred, truth, primary)} for name, pred in predictions.items()]
    base_loss = abs(baseline[:, primary] - truth[:, primary]).mean(axis=1)
    uncertainty = []
    for name, values in [('conditional_median', median), ('conditional_mode', mode)]:
        improvement = base_loss - abs(values[:, primary] - truth[:, primary]).mean(axis=1)
        rng = np.random.default_rng(config['seed'])
        boot = [improvement[stratified_draw(ids.tasting_category_fine, rng)].mean() for _ in range(config['bootstrap_repeats'])]
        uncertainty.append({'method': name, 'grade_mae_improvement': improvement.mean(),
                            'fixed_oof_bootstrap_p025': np.quantile(boot, .025), 'fixed_oof_bootstrap_p975': np.quantile(boot, .975)})
    stress = [{'additional_missing_rate': rate, 'repeat': repeat, 'method': rule,
               **metrics(values, truth, primary)} for (rate, repeat, rule), values in missing.items()]
    rows = pd.DataFrame({'beer': np.repeat(ids.beer.to_numpy(), len(yf.columns)), 'target': np.tile(yf.columns, len(ids)),
        'observed_relative_grade': truth.ravel(), 'baseline_relative_grade': baseline.ravel(),
        'median_relative_grade': median.ravel(), 'mode_relative_grade': mode.ravel(),
        **{f'probability_grade_{k}': probability[:, :, k].ravel() for k in range(7)}})
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output, 'model_comparison.csv', comparison); save_csv(output, 'paired_uncertainty.csv', uncertainty)
    save_csv(output, 'missingness.csv', stress); save_csv(output, 'oof_predictions.csv', rows)
    per_target = []; per_observed_grade = []
    for name, values in predictions.items():
        for j, target in enumerate(yf):
            error = abs(values[:, j] - truth[:, j])
            per_target.append({'method': name, 'target': target, 'primary': bool(primary[j]),
                'grade_mae': error.mean(), 'exact_accuracy': (error == 0).mean(), 'within_one_accuracy': (error <= 1).mean()})
        for grade in range(7):
            select = truth[:, primary] == grade
            error = abs(values[:, primary] - truth[:, primary])[select]
            per_observed_grade.append({'method': name, 'observed_relative_grade': grade, 'n': len(error),
                'grade_mae': error.mean(), 'exact_accuracy': (error == 0).mean(), 'within_one_accuracy': (error <= 1).mean()})
    save_csv(output, 'metrics_by_target.csv', per_target); save_csv(output, 'metrics_by_observed_grade.csv', per_observed_grade)
    # Deploy the prespecified median rule; keep the continuous prediction model unchanged.
    classifier = ExtraTreesClassifier(n_estimators=config['n_estimators'], min_samples_leaf=config['min_samples_leaf'],
        max_features=config['max_features'], random_state=config['seed'], n_jobs=1)
    grade_model = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)), ('model', classifier)])
    grade_model.fit(x, relative_grades(y, y))
    deployed = dict(bundle, grade_model=grade_model, grade_method='direct_relative_grade_conditional_median')
    joblib.dump(deployed, output / 'model.joblib', compress=3)
    # Constructive identifiability example: integer raw scores in range with identical z scores.
    raw_a = np.array([0., 1., 2., 3.]); raw_b = raw_a + 1
    z_a = (raw_a - raw_a.mean()) / raw_a.std(); z_b = (raw_b - raw_b.mean()) / raw_b.std()
    np.testing.assert_allclose(z_a, z_b, rtol=0, atol=1e-14)
    write_json(output / 'summary.json', {'status': 'passed', 'config': config, 'config_sha256': sha256(config_path),
        'source_sha256': {n: sha256(source / n) for n in ['1.csv', '4.csv', '8.xlsx']},
        'model_sha256': sha256(output / 'model.joblib'), 'base_continuous_model_sha256': sha256(q1 / 'model.joblib'),
        'identifiability_counterexample': {'raw_scores_a': raw_a.tolist(), 'raw_scores_b': raw_b.tolist(),
                                         'identical_standardized_values': z_a.tolist()},
        'raw_scale_identified': False, 'grade_reference': 'outer_training_labels_only',
        'uncertainty_scope': 'paired_style_stratified_fixed_OOF_sample_bootstrap_not_retraining_uncertainty',
        'main_decision': 'conditional_median_prespecified_for_absolute_grade_error',
        'limitations': ['Relative-grade labels are constructed, not recovered raw ratings.',
                       'Existing dataset and folds have informed the candidate family; exploratory evidence.',
                       'Mode is a secondary decision for exact accuracy, not a separately tuned model.']})
    print(pd.DataFrame(comparison).to_string(index=False)); print(pd.DataFrame(uncertainty).to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--q1', type=Path, default=Path('outputs/final_20261024/q1'))
    p.add_argument('--output', type=Path, default=Path('outputs/q1_ordinal_20261024'))
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.q1, args.output)
