"""Current Q3 conditional case, same-refit controls, chemical/sensory evidence."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json, sha256
from beer_models.local_transfer import defect_predictions
from beer_models.tail_risk import survival
from q3_stage2 import setup, source_records
from beer_models.raw_panel import portrait
from r2_raw_q1 import RAW_MANIFEST


def run(args):
    _, d, b, g, classifiers, z, radius, center, _, protected, dictionary, _ = setup(args.source, args.raw, args.q1)
    out = args.output
    read = lambda name: pd.read_csv(out/name, float_precision='round_trip')
    p = json.loads((ROOT/'configs/q3_stage2_protocol_20261026.json').read_text(encoding='utf-8'))
    sources, metadata, _, evidence = source_records(d, b, p)
    profiles = read('selected_profiles.csv')
    assessment = read('plan_assessment.csv').loc[lambda t: t.method.eq('risk_aware_pruned')].copy()
    assessment = assessment.merge(read('design_assessment.csv'), on='beer', suffixes=('', '_design'))
    assessment['status'] = np.where(assessment.stable_four_proxy_conditional,
        'four_proxy_stability_qualified_CONDITIONAL_Stale_unknown', 'restricted_proposal_NOT_passing_transfer')
    assessment['source_zero_diacetyl_assumption'] = assessment.beer.isin([119, 28])
    assessment['physical_feasibility_certified'] = False
    save_csv(out, 'current_plan_status.csv', assessment)
    before_x = np.stack([sources[int(beer)] for beer in profiles.beer])
    after_x = profiles[d['features']].to_numpy(float)
    a = defect_predictions(b, before_x, p['defects']); c = defect_predictions(b, after_x, p['defects'])
    chemical, defects, changes, styles = [], [], [], []
    for i, row in enumerate(profiles.itertuples()):
        source = sources[row.beer]; values = after_x[i]
        for j, field in enumerate(d['features']):
            changes.append(dict(beer=row.beer, field=field, chemical_name=dictionary.loc[field, 'chemical_name'],
                origin=dictionary.loc[field, 'origin_paper'], dictionary_unit=dictionary.loc[field, 'unit'],
                transform=dictionary.loc[field, 'candidate_transform'], before=source[j], after=values[j],
                coordinate_delta=values[j]-source[j], physical_control_validated=False))
        for j, target in enumerate(p['defects']):
            item = dict(beer=row.beer, target=target, grade_before=int(a['grade'][i, j]), grade_after=int(c['grade'][i, j]),
                true_nonworsening_certified=False, historical_probability_support=j < 4)
            for version in ['calibrated', 'uncalibrated']:
                for cut, label in [(0, 'ge1'), (4, 'ge5'), (5, 'six')]:
                    u, v = survival(a[version])[i, j, cut], survival(c[version])[i, j, cut]
                    item[f'{version}_{label}_before'], item[f'{version}_{label}_after'], item[f'{version}_{label}_delta'] = u, v, v-u
            defects.append(item)
            if j < 4:
                field = p['protected_coordinates'][j]; k = d['features'].index(field)
                e = evidence.loc[evidence.beer.eq(row.beer)&evidence.field.eq(field)].iloc[0]
                positive_log = j in [0, 2] and e.attachment8_calibrated_value > 0
                identity = j in [1, 3]
                known = positive_log or identity
                chemical.append(dict(beer=row.beer, field=field, coordinate_before=source[k], coordinate_after=values[k],
                    source_reference_concentration=e.attachment8_calibrated_value,
                    concentration_before=10**source[k] if positive_log else source[k] if identity else np.nan,
                    concentration_after=10**values[k] if positive_log else values[k] if identity else np.nan,
                    mapping_status='positive_log10_pair' if positive_log else 'identity_conditional_zero_bound' if identity and not np.isfinite(e.csv_coordinate) else 'identity_observed_pair' if identity else 'zero_censored_reference_no_inverse',
                    unit=dictionary.loc[field, 'unit'], threshold_mean=dictionary.loc[field, 'threshold_mean'], threshold_median=dictionary.loc[field, 'threshold_median'],
                    matrix_threshold_applicability_established=False, raw_missing_source=not np.isfinite(e.csv_coordinate)))
        for name, model in classifiers.items():
            probabilities = model.predict_proba(np.vstack([source, values]))
            ti = list(model.classes_).index('Tripel')
            styles.append(dict(beer=row.beer, model=name, source_prediction=model.predict(source[None])[0], after_prediction=model.predict(values[None])[0],
                source_Tripel_probability=probabilities[0, ti], after_Tripel_probability=probabilities[1, ti], calibration_established=False))
    save_csv(out, 'chemical_changes.csv', changes); save_csv(out, 'defect_before_after.csv', defects)
    save_csv(out, 'protected_chemical_checks.csv', chemical); save_csv(out, 'style_diagnostics.csv', styles)
    sensitivity = []
    for beer, t in pd.DataFrame(defects).groupby('beer', sort=False):
        probability_fields = [col for col in t if (col.startswith('calibrated_') or col.startswith('uncalibrated_')) and col.endswith('_delta')]
        sensitivity.append(dict(beer=int(beer), five_integer_nonincrease=bool((t.grade_after <= t.grade_before).all()),
            five_uncalibrated_nonzero_nonincrease=bool((t.uncalibrated_ge1_delta <= 1e-10).all()),
            five_calibrated_nonzero_nonincrease=bool((t.calibrated_ge1_delta <= 1e-10).all()),
            all_thirty_probability_nonincrease=bool((t[probability_fields] <= 1e-10).all().all()),
            stale_safety_certified=False))
    save_csv(out, 'probability_sensitivity.csv', sensitivity)
    before_frame = pd.DataFrame([dict(beer=f'source_{beer}', **dict(zip(d['features'], x))) for beer, x in sources.items()])
    after_frame = profiles.copy(); after_frame['beer'] = 'pruned_'+after_frame.beer.astype(str)
    scores, _, _, _ = portrait(pd.concat([before_frame, after_frame], ignore_index=True), b)
    scores.to_csv(out/'sensory52.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    comparisons = read('plan_assessment.csv').pivot(index='beer', columns='method', values='validation_joint_fraction').reset_index()
    comparisons = comparisons.merge(assessment[['beer', 'stable_four_proxy_conditional', 'changed_fields', 'distance_improvement']], on='beer')
    save_csv(out, 'matched_comparison.csv', comparisons)
    checks = read('validation_checks.csv')
    oob = []
    for (beer, method), t in checks.groupby(['beer', 'method'], sort=False):
        subset = t.loc[t.source_oob]
        oob.append(dict(beer=int(beer), method=method, n_source_oob_refits=len(subset), source_oob_joint_fraction=float(subset.joint_pass.mean()) if len(subset) else np.nan))
    save_csv(out, 'source_oob_summary.csv', oob)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    case = read('plan_assessment.csv').loc[lambda t: t.beer.eq(119)].set_index('method').loc[['previous_broad', 'previous_sparse', 'risk_aware_pruned']]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    labels = ['Previous broad', 'First sparse', 'Risk-aware pruned']
    axes[0].bar(labels, case.changed_fields, color=['#8c9aa6', '#d2a66e', '#438c83'])
    axes[0].set(ylabel='Changed supplied coordinates', title='Beer119: fewer changes than broad plan')
    axes[1].bar(labels, case.validation_joint_fraction*100, color=['#8c9aa6', '#d2a66e', '#438c83'])
    axes[1].axhline(75, color='#b34d42', ls='--', lw=1, label='Frozen75% gate')
    axes[1].set(ylabel='Joint pass in SAME12 new refits (%)', ylim=(0, 100), title='Four supported risk proxies + grades + style')
    axes[1].legend(fontsize=8)
    for ax in axes: ax.tick_params(axis='x', rotation=12)
    fig.savefig(out/'case119_comparison.png', dpi=160); fig.savefig(out/'case119_comparison.svg'); plt.close(fig)
    svg = out/'case119_comparison.svg'; svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    summary.update(current_stable_four_proxy_conditional_beers=assessment.loc[assessment.stable_four_proxy_conditional, 'beer'].astype(int).tolist(),
        q1_model_sha256=sha256(args.q1/'model.joblib'), source_sha256={name: sha256(args.source/name) for name in ['1.csv', '3.xlsx', '4.csv', '8.xlsx']},
        raw_sha256=sha256(args.raw), narrative_4vg_zero_count_correction=dict(censored_sources=9, positive_sources=[230, 119]),
        generation_reused_without_scientific_change=300, experimental_Q1_adopted=False)
    write_json(out/'summary.json', summary)
    print(assessment[['beer', 'changed_fields', 'distance_improvement', 'validation_joint_fraction', 'parameter_qualified_fraction', 'status']].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_pruning_20261027')
    with threadpool_limits(limits=1): run(parser.parse_args())
