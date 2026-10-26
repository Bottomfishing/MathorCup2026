"""Produce interpretable before/after changes and per-source Q3 bottlenecks."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from beer_audit.audit import save_csv, write_json
from beer_models.local_transfer import defect_predictions
from beer_models.tail_risk import survival
from q3_stage2 import setup, source_records
from r2_raw_q1 import RAW_MANIFEST


def run(args):
    p, d, b, g, classifiers, z, radius, center, target_indices, protected, dictionary, domains = setup(args.source, args.raw, args.q1)
    out = args.output
    read = lambda name: pd.read_csv(out/name, float_precision='round_trip')
    checks, results = read('candidate_checks.csv.gz'), read('transfer_results.csv')
    profiles, sources_table = read('selected_profiles.csv'), read('source_profiles.csv')
    sources, metadata, _, evidence = source_records(d, b, p)
    before = defect_predictions(b, np.stack([sources[int(beer)] for beer in profiles.beer]), p['defects'])
    after = defect_predictions(b, profiles[d['features']].to_numpy(float), p['defects'])
    defect_rows, changes, style_rows, blockers = [], [], [], []
    for i, row in enumerate(profiles.itertuples()):
        source = sources[row.beer]
        values = profiles.iloc[i][d['features']].to_numpy(float)
        for j, target in enumerate(p['defects']):
            field = p['protected_coordinates'][j] if j < 4 else None
            coordinate_before = source[d['features'].index(field)] if field else np.nan
            coordinate_after = values[d['features'].index(field)] if field else np.nan
            reference, compatible, threshold_mean, threshold_median, status = np.nan, False, np.nan, np.nan, 'no_measured_chemical_coordinate_for_Stale'
            if field:
                e = evidence.loc[evidence.beer.eq(row.beer)&evidence.field.eq(field)].iloc[0]
                reference = e.attachment8_calibrated_value
                threshold_mean, threshold_median = dictionary.loc[field, ['threshold_mean', 'threshold_median']]
                if not np.isfinite(e.csv_coordinate):
                    status = 'conditional_zero_bound_not_raw_measurement'
                elif reference <= 0:
                    status = 'zero_or_censored_reference_no_inverse_conversion'
                else:
                    expected = np.log10(reference) if field in ['DMS', 'X4VG'] else reference
                    compatible = bool(np.isclose(coordinate_before, expected, atol=1e-8, rtol=0))
                    status = 'positive_reference_pair_compatible' if compatible else 'reference_mapping_conflict'
            item = dict(beer=row.beer, domain=row.domain, target=target, field=field,
                coordinate_before=coordinate_before, coordinate_after=coordinate_after,
                coordinate_delta=coordinate_after-coordinate_before, reference_source_concentration=reference,
                unit=dictionary.loc[field, 'unit'] if field else None, mapping_status=status,
                threshold_mean=threshold_mean, threshold_median=threshold_median,
                reference_exceedance_mean=max(reference/threshold_mean-1, 0) if compatible and np.isfinite(threshold_mean) and threshold_mean > 0 else np.nan,
                reference_exceedance_median=max(reference/threshold_median-1, 0) if compatible and np.isfinite(threshold_median) and threshold_median > 0 else np.nan,
                threshold_applicability='reference_matrix_threshold_not_established_for_new_recipe',
                grade_before=int(before['grade'][i, j]), grade_after=int(after['grade'][i, j]),
                true_nonworsening_established=False, stale_evidence_sufficient=False if j == 4 else None)
            for version in ['calibrated', 'uncalibrated']:
                a, c = survival(before[version])[i, j], survival(after[version])[i, j]
                for cut, suffix in [(0, 'ge1'), (4, 'ge5'), (5, '6')]:
                    item[f'{version}_{suffix}_before'], item[f'{version}_{suffix}_after'] = a[cut], c[cut]
                    item[f'{version}_{suffix}_delta'] = c[cut]-a[cut]
            defect_rows.append(item)
        zs, za = g.robust_.transform(source[None])[0], g.robust_.transform(values[None])[0]
        for j, field in enumerate(d['features']):
            changes.append(dict(beer=row.beer, domain=row.domain, field=field,
                chemical_name=dictionary.loc[field, 'chemical_name'], origin=dictionary.loc[field, 'origin_paper'],
                dictionary_unit=dictionary.loc[field, 'unit'], supplied_transform=dictionary.loc[field, 'candidate_transform'],
                before=source[j], after=values[j], coordinate_delta=values[j]-source[j], robust_asinh_delta=za[j]-zs[j],
                physical_control_validated=False))
        for name, model in classifiers.items():
            labels = model.classes_
            probs = model.predict_proba(np.vstack([source, values]))
            ti = list(labels).index(p['target_style'])
            style_rows.append(dict(beer=row.beer, domain=row.domain, model=name,
                predicted_before=model.predict(source[None])[0], predicted_after=model.predict(values[None])[0],
                Tripel_probability_before=probs[0, ti], Tripel_probability_after=probs[1, ti],
                probability_calibrated=False, model_inference_not_OOF=True))
    for (beer, domain), table in checks.groupby(['beer', 'domain'], sort=False):
        structural = table.loc[table.structural_pass]
        reason = 'conditional_proxy_plan_Stale_and_physical_effect_unknown' if table.main_computational_pass.any() else 'no_joint_proxy_pass_in_frozen_search'
        item = dict(beer=int(beer), domain=domain, searched=len(table), structural_passes=len(structural),
            both_style_passes=int((table.logistic_style_pass&table.extra_trees_style_pass).sum()),
            main_passes=int(table.main_computational_pass.sum()), explanation=reason)
        for target in p['defects']:
            c = target+'_uncalibrated_delta_ge1'
            item[target+'_positive_nonzero_count_in_structural'] = int((structural[c] > p['numeric_tolerance']).sum())
            item[target+'_minimum_nonzero_delta_in_structural'] = float(structural[c].min()) if len(structural) else np.nan
        blockers.append(item)
    save_csv(out, 'defect_before_after.csv', defect_rows)
    save_csv(out, 'chemical_changes.csv', changes)
    change_frame = pd.DataFrame(changes)
    top = (change_frame.assign(magnitude=change_frame.robust_asinh_delta.abs()).sort_values('magnitude', ascending=False)
        .groupby(['beer', 'domain'], sort=False).head(12))
    save_csv(out, 'largest_coordinate_changes.csv', top)
    save_csv(out, 'style_diagnostics.csv', style_rows)
    save_csv(out, 'bottlenecks.csv', blockers)
    baseline_rows = []
    for row in metadata.loc[metadata.conditional_zero_bound].itertuples():
        beer = row.beer
        raw = d['x'][row.source_index]
        a = defect_predictions(b, raw[None], p['defects'])
        c = defect_predictions(b, sources[beer][None], p['defects'])
        for j, target in enumerate(p['defects']):
            for version in ['uncalibrated', 'calibrated']:
                baseline_rows.append(dict(beer=beer, target=target, risk=version,
                    raw_missing_imputed_grade=int(a['grade'][0, j]), conditional_zero_grade=int(c['grade'][0, j]),
                    raw_missing_imputed_nonzero=float(survival(a[version])[0, j, 0]),
                    conditional_zero_nonzero=float(survival(c[version])[0, j, 0]),
                    neither_is_recovered_measurement=True))
        for i in profiles.index[profiles.beer.eq(beer)]:
            for j, target in enumerate(p['defects']):
                baseline_rows.append(dict(beer=beer, target=target, risk='selected_uncalibrated_vs_both_baselines', domain=profiles.loc[i, 'domain'],
                    selected_nonzero=float(survival(after['uncalibrated'])[i, j, 0]),
                    delta_vs_raw_missing=float(survival(after['uncalibrated'])[i, j, 0]-survival(a['uncalibrated'])[0, j, 0]),
                    delta_vs_conditional_zero=float(survival(after['uncalibrated'])[i, j, 0]-survival(c['uncalibrated'])[0, j, 0]),
                    neither_is_recovered_measurement=True))
    save_csv(out, 'source_zero_baseline_sensitivity.csv', baseline_rows)
    ledger = [
        ('question', 'Pils/Lager_to_Tripel', 'Style transfer and style-conditioned generated spectra with sensory profiles'),
        ('question', 'specified_above_threshold_defects_nonworsening', 'Chemical threshold exceedance and predicted sensory proxies must be distinguished; Stale remains unestablished'),
        ('data_implementation', 'four_locked_chemical_coordinates', 'No inferred concentration increase under a fixed compatible scale; zero references and missing source values remain conditional'),
        ('data_implementation', 'four_uncalibrated_nonzero_guards', 'Historically supported predictive association, not causal threshold severity'),
        ('data_implementation', 'all_five_integer_nonincrease', 'Official Q1 interface consistency; near-zero grades alone are weak evidence'),
        ('team_extra', 'yeast_only_main_domain', 'Preserve Hops/Malt/other coordinates; yeast-origin annotation is not independent controllability'),
        ('team_extra', 'local_simplex_and_support', 'Five nearest measured Tripels, global empirical support radius and ten-percent distance improvement'),
        ('team_extra', 'both_style_classifiers', 'Two fixed classifiers, with limited historical OOF accuracy'),
        ('team_extra', 'thirty_probability_guards', 'Five targets times three cuts times calibrated/uncalibrated, retained sensitivity only'),
        ('team_extra', 'expanded_nondefect_domain', 'Predeclared broader recipe-coordinate alternative; not promoted after outcomes'),
        ('team_extra', 'old_full_target_box', 'Historical range comparison only; a defect below target minimum need not be increased')]
    save_csv(out, 'constraint_ledger.csv', [dict(kind=k, condition=c, meaning=m) for k, c, m in ledger])
    assessment = results.merge(read('bootstrap_summary.csv'), on=['beer', 'domain']).merge(read('parameter_sensitivity_summary.csv'), on=['beer', 'domain'])
    assessment['delivery_status'] = np.where(assessment.selected_main_pass,
        'nominal_conditional_spectrum_only_weak_refit_stability', 'restricted_proposal_not_a_passing_transfer')
    assessment['adopted_as_reliable_transfer'] = False
    save_csv(out, 'plan_assessment.csv', assessment)
    generated = pd.read_csv(ROOT/'outputs/task4_raw_q3_20261026/generated_profiles.csv', dtype={'generated_id': str})
    scores = read('generated_sensory52.csv.gz')
    scores['beer'] = scores.beer.astype(str)
    scores = scores.merge(generated[['generated_id', 'style']].rename(columns={'generated_id': 'beer'}), on='beer', validate='many_to_one')
    fingerprints = scores.groupby(['style', 'target'], sort=False, dropna=False).agg(
        generated_spectra=('beer', 'size'), mean_model_grade=('estimated_grade_0_6', 'mean'),
        median_model_grade=('estimated_grade_0_6', 'median'),
        mean_uncalibrated_nonzero=('uncalibrated_probability_rating_ge1', 'mean')).reset_index()
    fingerprints['measured_sensory_outcome'] = False
    save_csv(out, 'generated_fingerprints52.csv', fingerprints)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plans = assessment.loc[assessment.selected_main_pass]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    pivot = results.pivot(index='beer', columns='domain', values='distance_improvement')*100
    pivot.plot.bar(ax=axes[0], color=['#788b9b', '#c8d3db'], rot=0)
    axes[0].set(xlabel='Source beer', ylabel='Tripel center distance reduction (%)', title='Selected spectra (including failed proposals)')
    axes[0].legend(['Expanded sensitivity', 'Yeast block'], fontsize=8)
    x = np.arange(len(plans))
    axes[1].bar(x-.18, plans.joint_fraction*100, width=.36, label='Joint pass in 10 paired refits', color='#bc6045')
    axes[1].bar(x+.18, plans.main_computational_pass_fraction*100, width=.36, label='Pass in 32 parameter perturbations', color='#528a86')
    axes[1].set(xticks=x, xticklabels=plans.beer.astype(str), xlabel='Nominal expanded candidate', ylabel='Pass fraction (%)', ylim=(0, 100), title='Nominal passes have weak refit stability')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'transfer_evidence.png', dpi=160)
    fig.savefig(out/'transfer_evidence.svg')
    plt.close(fig)
    svg = out/'transfer_evidence.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    summary = json.loads((out/'summary.json').read_text(encoding='utf-8'))
    summary['evidence_sha256'] = {key.replace('\\', '/'): value for key, value in summary['evidence_sha256'].items()}
    summary['adopted_reliable_transfers'] = 0
    summary['expanded_plan_joint_refit_fraction_range'] = [float(plans.joint_fraction.min()), float(plans.joint_fraction.max())]
    write_json(out/'summary.json', summary)
    print('Analysis tables written; all actual-safety and reliable-transfer adoption flags remain false.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--raw', type=Path, default=RAW_MANIFEST.parent/'Raw sensory data anonymized.csv')
    parser.add_argument('--q1', type=Path, default=ROOT/'outputs/q1_selective_tail_20261026')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/q3_stage2_20261026')
    with threadpool_limits(limits=1):
        run(parser.parse_args())
