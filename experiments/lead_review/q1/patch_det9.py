from pathlib import Path

path = Path('scripts/q1_official_det_20261028.py')
text = path.read_text(encoding='utf-8')

old = """    run_long = pd.read_csv(out_run/'scores_all52_long.csv')
    merged = run_long.merge(portrait_long[['beer', 'target', 'estimated_grade_0_6']],
                            on=['beer', 'target'], how='inner', validate='one_to_one',
                            suffixes=('_run', '_portrait'))
"""
new = """    run_long = pd.read_csv(out_run/'scores_all52_long.csv')
    run_long['beer'] = run_long['beer'].astype(str)
    portrait_compare = portrait_long[['beer', 'target', 'estimated_grade_0_6']].copy()
    portrait_compare['beer'] = portrait_compare['beer'].astype(str)
    merged = run_long.merge(portrait_compare, on=['beer', 'target'], how='inner',
                            validate='one_to_one', suffixes=('_run', '_portrait'))
"""
assert old in text
text = text.replace(old, new)

old = """    both_inf = np.isinf(rebuilt_tau) & np.isinf(tau_original)
    any_inf = np.isinf(rebuilt_tau) | np.isinf(tau_original)
    tau_delta = float(np.max(np.where(any_inf,
                                      np.where(both_inf, 0.0, np.inf),
                                      np.abs(rebuilt_tau - tau_original))))
"""
new = """    with np.errstate(invalid='ignore'):
        both_inf = np.isinf(rebuilt_tau) & np.isinf(tau_original)
        any_inf = np.isinf(rebuilt_tau) | np.isinf(tau_original)
        tau_delta = float(np.max(np.where(any_inf,
                                          np.where(both_inf, 0.0, np.inf),
                                          np.abs(rebuilt_tau - tau_original))))
"""
assert old in text
text = text.replace(old, new)
path.write_text(text, encoding='utf-8')
print('patched merge and errstate')
