from pathlib import Path

path = Path('scripts/q1_official_det_20261028.py')
text = path.read_text(encoding='utf-8')
old = """    for fold in sorted(frame.outer_fold.unique()):
        part = frame[frame.outer_fold == fold]
        for target in targets:
            mask = part.target == target
            for threshold, key in ((5, 'tau_ge5'), (6, 'tau_6')):
                details_fold = details[(int(fold), key)]
                position = details_fold['keys'].index(target)
                frame.loc[mask, key] = float(details_fold['tau'][position])
"""
new = """    for fold in sorted(frame.outer_fold.unique()):
        for target in targets:
            for threshold, key in ((5, 'tau_ge5'), (6, 'tau_6')):
                details_fold = details[(int(fold), key)]
                position = details_fold['keys'].index(target)
                mask = ((frame.outer_fold == fold) & (frame.target == target)).to_numpy()
                frame.iloc[np.flatnonzero(mask), frame.columns.get_loc(key)] = float(
                    details_fold['tau'][position])
"""
assert old in text
path.write_text(text.replace(old, new), encoding='utf-8')
print('patched tau assignment')
