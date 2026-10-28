from pathlib import Path

path = Path('scripts/q1_official_det_20261028.py')
text = path.read_text(encoding='utf-8')
text = text.replace("'max_abs_grade_diff', 'mismatch_cells', 'pass']",
                    "'max_abs_grade_diff', 'mismatch_cells', 'pass_ok']")
text = text.replace('pass=bool(', 'pass_ok=bool(')
text = text.replace("batch_frame['pass'] = batch_frame['pass'].astype(bool)",
                    "batch_frame['pass_ok'] = batch_frame['pass_ok'].astype(bool)")
text = text.replace('.pass_)', '.pass_ok)')
text = text.replace("dict(pass_=bool(row['pass']),", "dict(pass_ok=bool(row['pass_ok']),")
path.write_text(text, encoding='utf-8')
print('patched')
