from pathlib import Path

path = Path('scripts/q1_official_det_20261028.py')
text = path.read_text(encoding='utf-8')

old = """    extra = cells.head(25).copy()
    extra['beer'] = extra['beer'].astype(int) + 10000
    appended = decide(pd.concat([cells, extra], ignore_index=True))
"""
new = """    extra = cells.head(25).copy()
    appended = decide(pd.concat([cells, extra], ignore_index=True))
"""
assert old in text
text = text.replace(old, new)

old = """    tau_original = frame[['tau_ge5', 'tau_6']].to_numpy(float)
    tau_delta = float(np.nanmax(np.abs(rebuilt_tau - tau_original)))
"""
new = """    tau_original = frame[['tau_ge5', 'tau_6']].to_numpy(float)
    both_inf = np.isinf(rebuilt_tau) & np.isinf(tau_original)
    any_inf = np.isinf(rebuilt_tau) | np.isinf(tau_original)
    tau_delta = float(np.max(np.where(any_inf,
                                      np.where(both_inf, 0.0, np.inf),
                                      np.abs(rebuilt_tau - tau_original))))
"""
assert old in text
text = text.replace(old, new)
path.write_text(text, encoding='utf-8')
print('patched batch append and inf-safe tau delta')
