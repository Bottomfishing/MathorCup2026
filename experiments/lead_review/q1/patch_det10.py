from pathlib import Path

path = Path('scripts/q1_official_det_20261028.py')
text = path.read_text(encoding='utf-8')

old = """    records = raw_long.merge(prediction, on=['beer', 'target'], how='inner',
                             validate='one_to_one')
"""
new = """    records = raw_long.merge(prediction, on=['beer', 'target'], how='inner',
                             validate='many_to_one')
"""
assert old in text
text = text.replace(old, new)

old = """def macro_spearman(frame, column):
    values = []
    for _target, group in frame.groupby('target', sort=True):
        if group['panel_mean_reference'].nunique() > 2:
            values.append(spearmanr(group[column], group['panel_mean_reference']).correlation)
    return float(np.nanmean(values)) if values else float('nan')
"""
new = """def macro_spearman(frame, column):
    import warnings
    values = []
    for _target, group in frame.groupby('target', sort=True):
        if group['panel_mean_reference'].nunique() > 2:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                values.append(spearmanr(group[column], group['panel_mean_reference']).correlation)
    return float(np.nanmean(values)) if values else float('nan')
"""
assert old in text
text = text.replace(old, new)
path.write_text(text, encoding='utf-8')
print('patched merge validate and spearman warnings')
