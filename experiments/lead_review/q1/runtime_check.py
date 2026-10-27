"""End-to-end proof: current_runtime.train -> bundle -> inference.run official rule."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

from beer_models import current_runtime, inference  # noqa: E402
from beer_models.q1_decision import (enforce_parent_child, observed_high_grade_rates,
                                     runtime_decision, topk_decision)  # noqa: E402
from beer_models.raw_panel import median_grades  # noqa: E402

SOURCE = Path(r'C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题'
              r'\2026年MathorCup大数据挑战赛-赛道A初赛\附件')
RAW = ROOT / 'data/reference/zenodo23624915/Raw sensory data anonymized.csv'
OUT = ROOT / 'experiments/lead_review/q1/runtime_check'


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    models = OUT / 'models'
    current_runtime.train(SOURCE, RAW, models)
    bundle = json.loads((models / 'training.json').read_text(encoding='utf-8'))
    model_path = models / 'q1.joblib'

    predictions = OUT / 'predict'
    inference.run(model_path, SOURCE / '1.csv', predictions)
    info = json.loads((predictions / 'prediction_info.json').read_text(encoding='utf-8'))
    long = pd.read_csv(predictions / 'scores_all52_long.csv')
    primary = pd.read_csv(predictions / 'estimated_grades_primary47.csv')
    wide = pd.read_csv(predictions / 'estimated_grades_all52.csv')

    frame = pd.read_csv(SOURCE / '1.csv', dtype=str)
    import joblib
    loaded = joblib.load(model_path)
    decision = loaded['q1_decision']
    grades, p_ge5, p6, legacy = inference._official_grade_frame(frame, loaded)
    targets = decision['targets']
    pairs = decision['parent_child_pairs']
    manual = runtime_decision(p_ge5, p6, legacy,
                              np.asarray(decision['rate_ge5']), np.asarray(decision['rate_6']),
                              targets, [p[0] for p in pairs], [p[1] for p in pairs])
    pieces = topk_decision(p_ge5.ravel(), p6.ravel(), np.minimum(legacy, 4).ravel(),
                           np.tile(np.asarray(decision['rate_ge5']), len(frame)),
                           np.tile(np.asarray(decision['rate_6']), len(frame)),
                           np.tile(np.asarray(targets, object), len(frame))).reshape(len(frame), -1)
    manual_pieces = enforce_parent_child(pieces, [p[0] for p in pairs], [p[1] for p in pairs])

    beer_order = pd.DataFrame({'beer': frame.beer.astype(str)})

    def _grid(value_column):
        part = long[long.target.isin(targets)].pivot(
            index='beer', columns='target', values=value_column).reset_index()
        part['beer'] = part['beer'].astype(str)
        merged = beer_order.merge(part, on='beer', how='left', validate='one_to_one')
        return merged[targets].to_numpy(float)

    grid = _grid('estimated_grade_0_6')
    legacy_grid = _grid('grade_legacy_integer_median')

    panel_median = _panel_median()
    rates_ge5, rates_6 = observed_high_grade_rates(panel_median)
    payload = {
        'bundle_decision_rule': decision['rule'],
        'bundle_implementation': decision['implementation'],
        'bundle_implementation_sha256': decision['implementation_sha256'],
        'current_implementation_sha256': file_sha256(ROOT / 'src/beer_models/q1_decision.py'),
        'training_json_rule': bundle.get('q1_decision_rule'),
        'prediction_grade_method': info.get('grade_method'),
        'prediction_rule_sha256': info.get('decision_rule', {}).get('implementation_sha256'),
        'official_vs_runtime_decision_max_abs_diff': float(np.abs(grid - manual).max()),
        'official_vs_topk_pieces_max_abs_diff': float(np.abs(grid - manual_pieces).max()),
        'legacy_column_equals_model_median': float(np.abs(
            np.nan_to_num(legacy_grid, nan=-1) - np.nan_to_num(legacy, nan=-1)).max()),
        'row_counts': {'long': len(long), 'primary': len(primary), 'wide': len(wide)},
        'columns_present': ['grade_legacy_integer_median', 'probability_rating_ge5',
                            'probability_rating_6'],
        'parent_violations_official': int(sum(
            (grid[:, child] > grid[:, parent]).sum() for parent, child in pairs)),
        'official_distribution': {str(int(k)): int(v) for k, v in
                                  zip(*np.unique(grid, return_counts=True))},
        'decision_rates_match_observed': float(max(
            np.abs(np.asarray(decision['rate_ge5']) - rates_ge5).max(),
            np.abs(np.asarray(decision['rate_6']) - rates_6).max())),
    }
    (OUT / 'runtime_check.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload, indent=2))


def _panel_median():
    from beer_models.panel_data import load_inputs
    protocol = json.loads((ROOT / 'configs/r2_raw_panel_protocol_20261026.json')
                          .read_text(encoding='utf-8'))
    data = load_inputs(SOURCE, RAW, protocol)
    return median_grades(data['distribution'])


if __name__ == '__main__':
    main()
