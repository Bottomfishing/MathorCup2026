from pathlib import Path

path = Path('scripts/q1_official_det_20261028.py')
text = path.read_text(encoding='utf-8')

old = """    targets = [str(target) for target in panel['targets']]
    pairs = parent_child_pairs(scope, targets)
    truth_median = median_grades(panel['distribution'])
"""
new = """    panel_targets = [str(target) for target in panel['targets']]
    truth_raw = median_grades(panel['distribution'])
"""
assert old in text
text = text.replace(old, new)

old = """    frame, file_targets = load_cells(args, config)
    if file_targets != targets:
        raise SystemExit('target list mismatch between OOF tables and the panel protocol')
"""
new = """    frame, file_targets = load_cells(args, config)
    if set(file_targets) != set(panel_targets):
        raise SystemExit('target list mismatch between OOF tables and the panel protocol')
    targets = list(file_targets)
    truth_median = truth_raw[:, [panel_targets.index(target) for target in targets]]
    pairs = parent_child_pairs(scope, targets)
"""
assert old in text
text = text.replace(old, new)
path.write_text(text, encoding='utf-8')
print('patched targets order')
