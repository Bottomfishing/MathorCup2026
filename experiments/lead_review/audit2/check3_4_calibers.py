"""Independent audit items 3+4: parent/child cap and high-grade calibers.

Recomputes, from the delivered Q1 long table and the raw sensory records with
within-beer taster weights, every number quoted in the review item:
  * pre-cap vs capped parent/child violations (expected 12 -> 0, 12 changed)
  * beer-level exact5 / exact6 / ge5 counts, precision and recall
  * individual-taster weighted exact5 / exact6 / ge5

Run from repo root:
    .venv\\Scripts\\python.exe experiments/lead_review/audit2/check3_4_calibers.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / 'experiments' / 'lead_review' / 'audit2'
Q1D = ROOT / 'outputs' / 'q1_decision_20261028'
RAW = ROOT / 'data' / 'reference' / 'zenodo23624915' / 'Raw sensory data anonymized.csv'
WEIGHTS = ROOT / 'outputs' / 'r2_raw_q1_20261026' / 'rating_weights_and_folds.csv'
PAIRS = json.loads((ROOT / 'configs' / 'q1_decision_20261028.json').read_text(encoding='utf-8'))


def pairs_from_summary(summary):
    parent_child = summary['parent_child']
    pairs = []
    for pair in parent_child['pairs']:
        if isinstance(pair, dict):
            pairs.append((int(pair['parent']), int(pair['child'])))
        elif isinstance(pair, (list, tuple)):
            pairs.append((int(pair[0]), int(pair[1])))
        else:
            text = str(pair)
            for separator in ('→', '->'):
                if separator in text:
                    left, right = [part.strip() for part in text.split(separator, 1)]
                    pairs.append((left, right))
                    break
    return pairs


def violations(frame, pairs):
    count = 0
    for parent, child in pairs:
        count += int((frame[:, child] > frame[:, parent]).sum())
    return count


def calibers(pred, truth):
    out = {}
    for caliber in (5, 6):
        pred_mask = pred == caliber
        true_mask = truth == caliber
        tp = int((pred_mask & true_mask).sum())
        n_pred = int(pred_mask.sum())
        n_obs = int(true_mask.sum())
        out[f'exact{caliber}'] = {'n_predicted': n_pred, 'true_positive': tp,
                                  'precision': round(tp / n_pred, 6) if n_pred else None,
                                  'n_observed': n_obs,
                                  'recall': round(tp / n_obs, 6) if n_obs else None}
    pred_ge5 = pred >= 5
    true_ge5 = truth >= 5
    tp = int((pred_ge5 & true_ge5).sum())
    n_pred = int(pred_ge5.sum())
    n_obs = int(true_ge5.sum())
    out['ge5'] = {'n_predicted': n_pred, 'true_positive': tp,
                  'precision': round(tp / n_pred, 6) if n_pred else None,
                  'n_observed': n_obs,
                  'recall': round(tp / n_obs, 6) if n_obs else None}
    return out


def weighted_calibers(pred_long, raw_long, weights=None):
    """pred_long/raw_long: beer,target,value; weights optional record->weight table."""
    merged = raw_long.merge(pred_long, on=['beer', 'target'], how='inner', validate='many_to_one')
    if weights is not None:
        merged = merged.merge(weights, on='raw_record_1based', how='left', validate='many_to_one')
    missing_weight = int(merged['weight_within_beer'].isna().sum())
    merged = merged.dropna(subset=['weight_within_beer'])
    out = {'raw_records_with_prediction': int(len(merged)),
           'raw_records_missing_weight': missing_weight,
           'beers': int(merged['beer'].nunique()),
           'targets': int(merged['target'].nunique()),
           'weight_sum': round(float(merged['weight_within_beer'].sum()), 6)}
    w = merged['weight_within_beer'].to_numpy(float)
    for caliber in (5, 6):
        pred_mask = merged['prediction_grade'].to_numpy(float) == caliber
        raw_mask = merged['rating'].to_numpy(float) == caliber
        out[f'exact{caliber}'] = {
            'n_predicted': round(float(w[pred_mask].sum()), 6),
            'true_positive': round(float(w[pred_mask & raw_mask].sum()), 6),
            'precision': round(float(w[pred_mask & raw_mask].sum() / w[pred_mask].sum()), 6),
            'n_observed': round(float(w[raw_mask].sum()), 6),
            'recall': round(float(w[pred_mask & raw_mask].sum() / w[raw_mask].sum()), 6)}
    pred_ge5 = merged['prediction_grade'].to_numpy(float) >= 5
    raw_ge5 = merged['rating'].to_numpy(float) >= 5
    out['ge5'] = {
        'n_predicted': round(float(w[pred_ge5].sum()), 6),
        'true_positive': round(float(w[pred_ge5 & raw_ge5].sum()), 6),
        'precision': round(float(w[pred_ge5 & raw_ge5].sum() / w[pred_ge5].sum()), 6),
        'n_observed': round(float(w[raw_ge5].sum()), 6),
        'recall': round(float(w[pred_ge5 & raw_ge5].sum() / w[raw_ge5].sum()), 6)}
    return out


def main():
    summary = json.loads((Q1D / 'summary.json').read_text(encoding='utf-8'))
    long = pd.read_csv(Q1D / 'q1__evaluation_long_12500.csv', dtype={'beer': str})
    targets = sorted(long['target'].unique())
    wide = long.pivot(index='beer', columns='target', values='grade_0_6')[targets].to_numpy(int)
    precap = long.pivot(index='beer', columns='target', values='grade_topk_precap')[targets].to_numpy(int)
    legacy = long.pivot(index='beer', columns='target', values='grade_integer_median_old')[targets].to_numpy(int)
    pairs = []
    bundle_path = ROOT / 'outputs' / 'current_models' / 'q1.joblib'
    if bundle_path.exists():
        import joblib
        bundle = joblib.load(bundle_path)
        decision = bundle['q1_decision']
        bundle_targets = [str(name) for name in decision['targets']]
        position = {name: index for index, name in enumerate(targets)}
        for parent, child in decision['parent_child_pairs']:
            pairs.append((position[bundle_targets[int(parent)]], position[bundle_targets[int(child)]]))
    else:
        pairs = pairs_from_summary(summary)

    report = {'item': 'parent_child_cap_and_high_grade_calibers'}
    report['parent_child'] = {
        'n_pairs': len(pairs),
        'violations_precap': violations(precap, pairs),
        'violations_capped': violations(wide, pairs),
        'violations_legacy_integer_median': violations(legacy, pairs),
        'changed_cells': int((precap != wide).sum()),
        'changed_cells_only_down': bool(((precap > wide) | (precap == wide)).all()),
        'summary_claims': summary['parent_child']}
    report['grade_distribution'] = {
        'long': {int(k): int(v) for k, v in long['grade_0_6'].value_counts().sort_index().items()},
        'summary': summary['grade_distribution'],
        'precap_long': {int(k): int(v) for k, v in long['grade_topk_precap'].value_counts().sort_index().items()},
        'summary_precap': summary['precap_grade_distribution']}

    truth = long.pivot(index='beer', columns='target', values='panel_median_reference')[targets].to_numpy(float)
    truth_round = np.rint(truth).astype(int)
    report['beer_level'] = calibers(wide, truth_round)

    # individual-taster weighted calibers: raw sensory x within-beer weights.
    # Competition beer ids come from the weight table (the anonymised raw file
    # only carries session/taster codes), exactly as the official script does.
    raw = pd.read_csv(RAW, encoding='cp1252')
    raw = raw.reset_index(drop=True)
    raw['raw_record_1based'] = np.arange(1, len(raw) + 1)
    raw = raw.rename(columns={'A_ester_fruity': 'A_esters_fruity',
                              'F_ester_fruity': 'F_esters_fruity'})
    weights = pd.read_csv(WEIGHTS)
    weights['raw_record_1based'] = weights['raw_record_1based'].astype(int)
    weights = weights[['raw_record_1based', 'beer', 'weight_within_beer']]
    raw = raw.merge(weights, on='raw_record_1based', how='left', validate='one_to_one')
    missing = int(raw[['beer', 'weight_within_beer']].isna().any(axis=1).sum())
    raw_long = raw.melt(id_vars=['raw_record_1based', 'beer', 'weight_within_beer'],
                        value_vars=targets, var_name='target', value_name='rating')
    raw_long['rating'] = pd.to_numeric(raw_long['rating'], errors='coerce')
    raw_long = raw_long.dropna(subset=['rating'])
    raw_long['beer'] = raw_long['beer'].astype(str)
    predictions = long[['beer', 'target', 'grade_0_6']].rename(columns={'grade_0_6': 'prediction_grade'})
    predictions['beer'] = predictions['beer'].astype(str)
    report['individual_weighted'] = weighted_calibers(predictions, raw_long)
    report['individual_weighted']['unmapped_raw_records'] = missing

    # delivered caliber table for comparison.
    delivered = pd.read_csv(Q1D / 'q1__metric_calibers.csv')
    report['delivered_table'] = delivered.to_dict(orient='records')

    # scorecard: does the committed README / docs still quote the old mixed caliber numbers?
    readme = (ROOT / 'README.md').read_text(encoding='utf-8', errors='replace')
    docs = (ROOT / 'docs' / '待优化问题.md')
    doc_text = docs.read_text(encoding='utf-8', errors='replace') if docs.exists() else ''
    needles_old = ['204', '25.49', '26.26', '188', '20.74', '21.79', '16/1', '6.25']
    needles_new = ['180', '21.11', '21.23', '194', '26.29', '25.76', '14/1', '7.14', '5.26']
    report['docs_scan'] = {
        'readme_old_numbers': {n: readme.count(n) for n in needles_old},
        'readme_new_numbers': {n: readme.count(n) for n in needles_new},
        'doc_old_numbers': {n: doc_text.count(n) for n in needles_old},
        'doc_new_numbers': {n: doc_text.count(n) for n in needles_new}}

    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / 'check3_4_calibers.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
