from pathlib import Path

path = Path('scripts/q1_official_det_20261028.py')
text = path.read_text(encoding='utf-8')
text = text.replace("bool(all(row.pass_ for row in batch_frame.itertuples(index=False)",
                    "bool(all(row.pass_ok for row in batch_frame.itertuples(index=False)")
text = text.replace("verification = dict(all_pass_ok=bool(all(checks.values())), checks=checks,",
                    "verification = dict(all_pass=bool(all(checks.values())), checks=checks,")
path.write_text(text, encoding='utf-8')
print('patched all_pass key')
