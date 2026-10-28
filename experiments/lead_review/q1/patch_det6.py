from pathlib import Path

path = Path('scripts/q1_official_det_20261028.py')
text = path.read_text(encoding='utf-8')

old = """    grades = threshold_decision(frame['probability_rating_ge5'].to_numpy(float),
                                frame['probability_rating_6'].to_numpy(float),
                                frame['grade_integer_median_old'].to_numpy(float),
                                frame['tau_ge5'].to_numpy(float),
                                frame['tau_6'].to_numpy(float),
                                frame['target'].astype(str).to_numpy())
"""
new = """    groups = (frame['target'].astype(str) + '|' + frame['outer_fold'].astype(str)).to_numpy()
    grades = threshold_decision(frame['probability_rating_ge5'].to_numpy(float),
                                frame['probability_rating_6'].to_numpy(float),
                                frame['grade_integer_median_old'].to_numpy(float),
                                frame['tau_ge5'].to_numpy(float),
                                frame['tau_6'].to_numpy(float), groups)
"""
assert old in text
text = text.replace(old, new)

old = """    cells = frame[['probability_rating_ge5', 'probability_rating_6',
                   'grade_integer_median_old', 'tau_ge5', 'tau_6', 'target']].copy()
    def decide(block):
        return threshold_decision(block['probability_rating_ge5'].to_numpy(float),
                                  block['probability_rating_6'].to_numpy(float),
                                  block['grade_integer_median_old'].to_numpy(float),
                                  block['tau_ge5'].to_numpy(float),
                                  block['tau_6'].to_numpy(float),
                                  block['target'].astype(str).to_numpy())
"""
new = """    cells = frame[['probability_rating_ge5', 'probability_rating_6',
                   'grade_integer_median_old', 'tau_ge5', 'tau_6', 'target',
                   'outer_fold']].copy()
    def decide(block):
        groups = (block['target'].astype(str) + '|' + block['outer_fold'].astype(str)).to_numpy()
        return threshold_decision(block['probability_rating_ge5'].to_numpy(float),
                                  block['probability_rating_6'].to_numpy(float),
                                  block['grade_integer_median_old'].to_numpy(float),
                                  block['tau_ge5'].to_numpy(float),
                                  block['tau_6'].to_numpy(float), groups)
"""
assert old in text
text = text.replace(old, new)
path.write_text(text, encoding='utf-8')
print('patched groups')
