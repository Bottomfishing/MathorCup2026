"""Standalone experimental inference; never modifies or calls the official default."""
import argparse
import json
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT/'src')]
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_audit.checks import numeric_frame
from beer_models.raw_panel import predict_raw_frame


def run(model_path, input_path, output):
    bundle = joblib.load(model_path)
    assert bundle['kind'] == 'independent_two_stage_not_approved' and not bundle['integration_approved']
    frame = pd.read_csv(input_path, dtype=str)
    adapter = dict(features=bundle['features'], targets=bundle['targets'], raw_panel_model=bundle['model'],
        scorecard_scope=bundle['scorecard_scope'], protocol=bundle['base_protocol'],
        score_status='independent_experiment_not_approved')
    long, main, wide, quality = predict_raw_frame(frame, adapter)
    x = numeric_frame(frame.reindex(columns=bundle['features'])).to_numpy(float)
    p = bundle['model'].predict_distribution(x)
    risks = pd.DataFrame([dict(beer=str(beer), target=target,
        experimental_probability_ge5=float(p[i, j, 5:].sum()), experimental_probability_6=float(p[i, j, 6]))
        for i, beer in enumerate(long.beer.drop_duplicates()) for j, target in enumerate(bundle['targets'])])
    long = long.merge(risks, on=['beer', 'target'], how='left', validate='one_to_one')
    long['safety_certified'] = False
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output, 'scores52.csv', long); save_csv(output, 'grades52.csv', wide)
    save_csv(output, 'main47.csv', main); save_csv(output, 'quality.csv', quality)
    write_json(output/'info.json', dict(model_sha256=sha256(model_path), input_sha256=sha256(input_path),
        samples=len(frame), independent_experiment=True, official_integration=False,
        experiment_success=bundle['experiment_success'], policy=bundle['model'].policy,
        score_semantics='selective_tail_decision_on_original_0_6_scale_not_conditional_median',
        probability_status='class_weight_odds_restored_but_uncalibrated_not_defect_safety',
        training_source_sha256=bundle['source_sha256'],
        original_official_predictor_changed=False, evaluation_use='full_fit_inference_not_OOF_accuracy'))
    print(f'Experimental inference only: {len(frame)} beers, {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=HERE/'results/experiment.joblib')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE/'results/prediction_smoke')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.model, args.input, args.output)
