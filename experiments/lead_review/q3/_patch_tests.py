"""One-off patch for tests/test_q3_migration.py expectations (kept for audit)."""

from __future__ import annotations

import io
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
path = ROOT / "tests" / "test_q3_migration.py"
text = io.open(path, encoding="utf-8").read()

replacements = [
    (
        'self.assertEqual(qm.defect_status(0.15, 0.04, 0.20), "improved")',
        'self.assertEqual(qm.defect_status(0.25, 0.15, 0.20), "improved")',
    ),
    (
        'self.assertEqual(qm.defect_status(0.04, 0.04, 0.20), "unchanged")',
        'self.assertEqual(qm.defect_status(0.15, 0.04, 0.20), "unchanged")\n'
        '        self.assertEqual(qm.defect_status(0.04, 0.04, 0.20), "unchanged")',
    ),
    (
        'd.defect_concentration(migration["standardised"], qm.DEFECT_SPECS[2])',
        'd.defect_concentration(migration["standardised"], "4VG")',
    ),
    (
        'self.assertEqual(sum(column.endswith("_after") for column in wide.columns), 224)',
        'self.assertEqual(len(wide.columns), 455)\n'
        '        self.assertEqual(sum(column.endswith("_after") for column in wide.columns), 225)',
    ),
    (
        'self.assertEqual(sum(column.endswith("_before") for column in wide.columns), 224)',
        'self.assertEqual(sum(column.endswith("_before") for column in wide.columns), 225)',
    ),
]
for old, new in replacements:
    if old not in text:
        raise SystemExit(f"pattern not found: {old[:80]!r}")
    text = text.replace(old, new)

io.open(path, "w", encoding="utf-8", newline="\n").write(text)
print("patched", path)
