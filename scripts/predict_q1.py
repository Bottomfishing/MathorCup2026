"""Use a trusted locally trained Q1 bundle; input coordinates match attachment 1."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.predict import predict_frame
from beer_models.scorecard import predict_scorecard_frame


def run(model, input_path, output):
    bundle = joblib.load(model)
    frame = pd.read_csv(input_path, dtype=str)
    if bundle.get('bundle_kind') == 'r2_raw_panel':
        from beer_models.raw_panel import predict_raw_frame
        long, primary, wide, samples = predict_raw_frame(frame, bundle)
        output.mkdir(parents=True, exist_ok=False)
        save_csv(output, 'scores_all52_long.csv', long)
        save_csv(output, 'estimated_grades_primary47.csv', primary)
        save_csv(output, 'estimated_grades_all52.csv', wide)
        save_csv(output, 'sample_quality.csv', samples)
        write_json(output / 'prediction_info.json', {
            'samples': len(frame), 'targets': 52, 'labelled_panel_targets': len(bundle['targets']),
            'model': bundle['selected_model'], 'model_sha256': sha256(model), 'input_sha256': sha256(input_path),
            'score_scale': 'original_0_6_individual_rating_scale',
            'grade_semantics': 'integer_balanced_panel_conditional_median_with_parent_cap',
            'grade_method': 'r2_raw_panel_median', 'grade_probability_calibrated': False,
            'missing_columns_imputed': [f for f in bundle['features'] if f not in frame],
            'unlabelled_card_targets': bundle['protocol']['unlabelled_card_targets'],
            'raw_source_sha256': bundle['raw_source_sha256'],
            'qualified_on_historical_grouped_folds': bundle['qualified_on_historical_grouped_folds'],
            'input_scale': 'same_supplied_coordinates_as_attachment_1_no_automatic_log_transform',
            'evaluation_use': 'full_training_model_predictions_are_not_OOF_accuracy_estimates'})
        print(f'Predicted {len(frame)} samples on raw 0-6 scale; output: {output}')
        return
    complete_card = 'scorecard_protocol' in bundle
    if complete_card:
        long, wide, all_items, samples = predict_scorecard_frame(frame, bundle)
    else:
        long, wide, samples = predict_frame(frame, bundle)
    output.mkdir(parents=True, exist_ok=False)
    if complete_card:
        save_csv(output, 'scores_all52_long.csv', long)
        save_csv(output, 'estimated_grades_primary47.csv', wide)
        save_csv(output, 'estimated_grades_all52.csv', all_items)
    else:
        save_csv(output, 'scores_all50_long.csv', long)
        save_csv(output, 'relative_grades_primary45.csv', wide)
    save_csv(output, 'sample_quality.csv', samples)
    info = {'samples': len(frame), 'targets': long.target.nunique(),
        'labelled_panel_targets': len(bundle['targets']), 'complete_scorecard_view': complete_card,
        'model': bundle['selected_model'], 'model_sha256': sha256(model), 'input_sha256': sha256(input_path),
        'missing_columns_imputed': [f for f in bundle['features'] if f not in frame],
        'ignored_input_columns': [f for f in frame if f not in bundle['features'] and f != 'beer'],
        'input_scale': 'same_supplied_coordinates_as_attachment_1_no_automatic_log_transform',
        'score_scale': 'provided_standardized_panel_score',
        'grade_semantics': ('per_row_panel_relative_or_threshold_scenario_not_raw_card_scores' if complete_card
                            else 'relative_midrank_ECDF_of_full_training_targets_not_raw_panel_rating'),
        'grade_method': bundle.get('grade_method', 'continuous_prediction_then_training_ECDF'),
        'grade_interval_kind': ('uncalibrated_equal_tail_forecast_interval' if 'grade_model' in bundle else 'not_available_for_continuous_only_bundle'),
        'grade_interval_nominal_level': .9 if 'grade_model' in bundle else None,
        'grade_probability_calibrated': False if 'grade_model' in bundle else None,
        'grade_interval_scope': 'panel_relative_targets_only_no_interval_for_unlabelled_proxies',
        'scorecard_protocol': bundle.get('scorecard_protocol'),
        'evaluation_use': 'full_training_model_predictions_are_not_OOF_accuracy_estimates'}
    if 'selected_grade_model' in bundle:
        info.update(task2_official_default=bundle['task2_official_default'],
            task2_qualified_on_historical_partition=bundle['task2_qualified'],
            task2_protocol=bundle['task2_protocol'])
    write_json(output / 'prediction_info.json', info)
    print(f'Predicted {len(frame)} samples; output: {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('outputs/r2_raw_q1_20261026/model.joblib'))
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.model, args.input, args.output)
