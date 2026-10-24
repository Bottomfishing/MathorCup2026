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


def run(model, input_path, output):
    bundle = joblib.load(model)
    frame = pd.read_csv(input_path, dtype=str)
    long, wide, samples = predict_frame(frame, bundle)
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output, 'scores_all50_long.csv', long)
    save_csv(output, 'relative_grades_primary45.csv', wide)
    save_csv(output, 'sample_quality.csv', samples)
    write_json(output / 'prediction_info.json', {'samples': len(frame), 'targets': len(bundle['targets']),
        'model': bundle['selected_model'], 'model_sha256': sha256(model), 'input_sha256': sha256(input_path),
        'missing_columns_imputed': [f for f in bundle['features'] if f not in frame],
        'ignored_input_columns': [f for f in frame if f not in bundle['features'] and f != 'beer'],
        'input_scale': 'same_supplied_coordinates_as_attachment_1_no_automatic_log_transform',
        'score_scale': 'provided_standardized_panel_score',
        'grade_semantics': 'relative_midrank_ECDF_of_full_training_targets_not_raw_panel_rating',
        'evaluation_use': 'full_training_model_predictions_are_not_OOF_accuracy_estimates'})
    print(f'Predicted {len(frame)} samples; output: {output}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('outputs/final_20261024/q1/model.joblib'))
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        run(args.model, args.input, args.output)
