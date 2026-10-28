from pathlib import Path

path = Path('scripts/q1_official_det_20261028.py')
text = path.read_text(encoding='utf-8')
old = """    merge['outer_fold'] = merge.beer.map(fold_of_beer).astype(int)
    merge['target_family'] = merge.target.map(target_family)
"""
new = """    merge['outer_fold'] = merge.beer.map(fold_of_beer).astype(int)
    merge['target_family'] = merge.target.map(target_family)
    merge['panel_median_reference'] = merge['panel_weighted_median'].astype(float)
    merge['panel_mean_reference'] = merge['panel_weighted_mean'].astype(float)
"""
assert old in text
text = text.replace(old, new)
path.write_text(text, encoding='utf-8')
print('patched reference columns')
