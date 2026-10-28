from pathlib import Path

path = Path('scripts/q1_official_det_20261028.py')
text = path.read_text(encoding='utf-8')
old = "            probabilities = base_probabilities(model.model, panel['x'][valid_index])"
new = "            probabilities = base_probabilities(model, panel['x'][valid_index])"
assert old in text
path.write_text(text.replace(old, new), encoding='utf-8')
print('patched base_probabilities call')
