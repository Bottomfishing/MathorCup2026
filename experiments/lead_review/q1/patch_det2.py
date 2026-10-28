from pathlib import Path

path = Path('scripts/q1_official_det_20261028.py')
text = path.read_text(encoding='utf-8')

old_caliber = """def caliber_rows(frame, set_name, scope_name='beer_level'):
    rows = []
    predicted = frame['grade_0_6'].to_numpy(int) if 'grade_0_6' in frame.columns else None
"""
new_caliber = """def caliber_rows(frame, set_name, column='grade_0_6', scope_name='beer_level'):
    rows = []
    predicted = frame[column].to_numpy(int)
"""
assert old_caliber in text
text = text.replace(old_caliber, new_caliber)
text = text.replace("calibers = caliber_rows(old, 'q1_topk_parent_capped_old_baseline')",
                    "calibers = caliber_rows(old, 'q1_topk_parent_capped_old_baseline', 'grade_old_topk')")

old_raw = """    raw_long = pd.read_csv(resolve(args.raw), encoding='cp1252')
    raw_long = raw_long.rename(columns={'A_ester_fruity': 'A_esters_fruity',
                                        'F_ester_fruity': 'F_esters_fruity'})
    weights = pd.read_csv(resolve(R2/'rating_weights_and_folds.csv'))
    values = raw_long.melt(id_vars=['Beer'], value_vars=targets, var_name='target', value_name='rating')
    values['raw_record_1based'] = values.groupby('Beer').cumcount() + 1
    long = values.merge(weights[['raw_record_1based', 'beer', 'target', 'weight_within_beer']],
                        on=['raw_record_1based', 'target'], how='inner', validate='many_to_one')
"""
new_raw = """    raw = pd.read_csv(resolve(args.raw), encoding='cp1252')
    raw['raw_record_1based'] = np.arange(1, len(raw) + 1)
    raw = raw.rename(columns={'A_ester_fruity': 'A_esters_fruity',
                              'F_ester_fruity': 'F_esters_fruity'})
    weights = pd.read_csv(resolve(R2/'rating_weights_and_folds.csv'))
    raw = raw.merge(weights[['raw_record_1based', 'beer', 'weight_within_beer']],
                    on='raw_record_1based', how='left')
    if raw[['beer', 'weight_within_beer']].isna().any().any():
        raise SystemExit('raw sensory rows are not fully mapped to competition beers')
    long = raw.melt(id_vars=['raw_record_1based', 'beer', 'weight_within_beer'],
                    value_vars=targets, var_name='target', value_name='rating')
    weight_sums = long.groupby(['beer', 'target'])['weight_within_beer'].sum()
    if float((weight_sums - 1.0).abs().max()) > 1e-9:
        raise SystemExit('within-beer weights do not sum to one')
"""
assert old_raw in text
text = text.replace(old_raw, new_raw)

old_purity = """    tau_perturbed = perturbed_frame[['tau_ge5', 'tau_6']].to_numpy(float).copy()
    for fold in sorted(frame.outer_fold.unique()):
        part = inner[inner.outer_fold == fold]
        labels = np.clip(part.panel_median.to_numpy(float) + noise[:len(part)], 0, 6)
        for threshold, key in ((5, 'tau_ge5'), (6, 'tau_6')):
            details_fold = training_threshold(
                part['p_ge5' if threshold == 5 else 'p6'].to_numpy(), labels,
                part['target'].astype(str).to_numpy(), threshold)
            for target in targets:
                position = details_fold['keys'].index(target)
                mask = (perturbed_frame.outer_fold == fold) & (perturbed_frame.target == target)
                tau_perturbed[np.flatnonzero(mask), 0 if key == 'tau_ge5' else 1] = float(
                    details_fold['tau'][position])
    tau_original = frame[['tau_ge5', 'tau_6']].to_numpy(float)
    tau_delta = float(np.nanmax(np.abs(tau_perturbed - tau_original)))
    pert_grade, _, pert_back = decision_matrix(perturbed_frame, targets, pairs, None)
    grade_delta = int(np.max(np.abs(pert_back - grades)))
    perturbed_frame['grade_0_6'] = pert_back
"""
new_purity = """    rebuilt_tau = frame[['tau_ge5', 'tau_6']].to_numpy(float).copy()
    for fold in sorted(frame.outer_fold.unique()):
        part = inner[inner.outer_fold == fold]
        for threshold, column in ((5, 0), (6, 1)):
            details_fold = training_threshold(
                part['p_ge5' if threshold == 5 else 'p6'].to_numpy(),
                part['panel_median'].to_numpy(),
                part['target'].astype(str).to_numpy(), threshold)
            for target in targets:
                position = details_fold['keys'].index(target)
                mask = (perturbed_frame.outer_fold == fold) & (perturbed_frame.target == target)
                rebuilt_tau[np.flatnonzero(mask), column] = float(details_fold['tau'][position])
    tau_original = frame[['tau_ge5', 'tau_6']].to_numpy(float)
    tau_delta = float(np.nanmax(np.abs(rebuilt_tau - tau_original)))
    pert_grade, _, pert_back = decision_matrix(perturbed_frame, targets, pairs, None)
    grade_delta = int(np.max(np.abs(pert_back - grades)))
"""
assert old_purity in text
text = text.replace(old_purity, new_purity)

text = text.replace("    feature_frame = frame[['beer', 'target']].drop_duplicates('beer')[['beer']].copy()\n", '')
path.write_text(text, encoding='utf-8')
print('patched det script')
