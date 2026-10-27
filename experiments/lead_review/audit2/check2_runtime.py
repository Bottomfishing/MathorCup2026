"""Independent audit item 2: runtime Q1 decision rule vs official table.

Recomputes the frozen top-k + parent/child cap from the runtime predict output's
own probability/legacy columns and the shipped bundle parameters, then compares
against (a) the runtime's own estimated grades and (b) the delivered full-fit
portrait table.  Also hashes the implementation file named in the bundle.

Run from repo root:
    .venv\\Scripts\\python.exe experiments/lead_review/audit2/check2_runtime.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / 'experiments' / 'lead_review' / 'audit2'
OUTPUT = AUDIT / 'runtime_predict' / 'scores_all52_long.csv'
PACKAGE_TABLE = ROOT / 'deliverables' / 'current_20261028' / 'tables' / 'current__official_Q1_fullfit_scores52.csv'
MODEL = ROOT / 'outputs' / 'current_models' / 'q1.joblib'

sys.path.insert(0, str(ROOT / 'src'))
from beer_models.q1_decision import runtime_decision  # noqa: E402


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    bundle = joblib.load(MODEL)
    decision = bundle['q1_decision']
    targets = [str(name) for name in decision['targets']]
    pairs = [(int(pair[0]), int(pair[1])) for pair in decision['parent_child_pairs']]
    rates5 = np.asarray(decision['rate_ge5'], dtype=float)
    rates6 = np.asarray(decision['rate_6'], dtype=float)

    long = pd.read_csv(OUTPUT)
    beer_order = list(dict.fromkeys(long['beer'].astype(str)))

    def pivot(column):
        wide = long.pivot(index='beer', columns='target', values=column)
        wide.index = wide.index.astype(str)
        return wide.loc[beer_order, targets]

    p5 = pivot('probability_rating_ge5_y').to_numpy(float)
    p6 = pivot('probability_rating_6_y').to_numpy(float)
    legacy = pivot('grade_legacy_integer_median').to_numpy(float)
    runtime_grade = pivot('estimated_grade_0_6').to_numpy(float)

    recomputed = runtime_decision(p5, p6, legacy, rates5, rates6, targets,
                                  [a for a, _ in pairs], [b for _, b in pairs])

    mismatch = int((recomputed != runtime_grade).sum())
    max_abs = float(np.abs(recomputed - runtime_grade).max()) if mismatch else 0.0
    violations = int(sum(int((recomputed[:, b] > recomputed[:, a]).sum()) for a, b in pairs))

    # (b) delivered full-fit portrait table (long form: beer, target, estimated grade).
    package = pd.read_csv(PACKAGE_TABLE)
    package['beer'] = package['beer'].astype(str)
    package = package[package['target'].isin(targets)][['beer', 'target', 'estimated_grade_0_6']]
    package = package.rename(columns={'estimated_grade_0_6': 'package_grade'})
    package = package[package['beer'].isin(beer_order)]
    runtime_long = pd.DataFrame(runtime_grade, index=beer_order, columns=targets)
    runtime_long = runtime_long.stack().rename('runtime_grade').reset_index()
    runtime_long.columns = ['beer', 'target', 'runtime_grade']
    merged = runtime_long.merge(package, on=['beer', 'target'], how='outer', validate='one_to_one')
    package_mismatch = int((merged['package_grade'] != merged['runtime_grade']).sum())
    package_rows = int(merged['runtime_grade'].notna().sum())

    # (c) implementation hash recorded in the bundle.
    implementation = ROOT / 'src' / 'beer_models' / 'q1_decision.py'
    impl_hash = file_sha256(implementation)
    declared = str(decision.get('implementation_sha256', ''))

    report = {
        'item': 'runtime_decision_rule_consistency',
        'runtime_predict_output': str(OUTPUT.relative_to(ROOT)),
        'package_table': str(PACKAGE_TABLE.relative_to(ROOT)),
        'model': str(MODEL.relative_to(ROOT)),
        'rows': int(len(beer_order)),
        'targets': len(targets),
        'n_parent_child_pairs': len(pairs),
        'recomputed_vs_runtime_mismatch_cells': mismatch,
        'recomputed_vs_runtime_max_abs_diff': max_abs,
        'runtime_parent_child_violations': violations,
        'package_vs_runtime_comparison_rows': package_rows,
        'package_vs_runtime_mismatch_cells': package_mismatch,
        'bundle_declares_implementation': decision.get('implementation'),
        'bundle_declared_implementation_sha256': declared,
        'actual_q1_decision_sha256': impl_hash,
        'implementation_hash_matches': declared == impl_hash,
        'rule': decision.get('rule'),
        'middle_cap': decision.get('middle_cap'),
        'forced': decision.get('forced'),
        'rate_source': decision.get('rate_source'),
        'legacy_grade_column': decision.get('legacy_grade_column'),
        'official_grade_column': decision.get('official_grade_column'),
        'verdict': 'fixed' if (mismatch == 0 and violations == 0 and package_mismatch == 0 and declared == impl_hash) else 'broken',
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / 'check2_runtime.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
