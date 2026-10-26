import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location('final_reproduction',
    Path(__file__).resolve().parents[1]/'scripts/reproduce_final.py')
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


class FinalComparisonTests(unittest.TestCase):
    def test_csv_only_normalizes_line_endings_and_optional_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'a.csv', Path(folder)/'b.csv'
            a.write_bytes(b'id,text,value,seconds\n1,"line1\r\nline2",0.125,3\n')
            b.write_bytes(b'id,text,value,seconds\n1,"line1\nline2",0.125,9\n')
            repro.compare_csv(a, b, ignore_timing=True)
            with self.assertRaises(AssertionError):
                repro.compare_csv(a, b)
            b.write_bytes(b'id,text,value,seconds\n1,"line1\nline2",0.126,9\n')
            with self.assertRaises(AssertionError):
                repro.compare_csv(a, b, ignore_timing=True)

    def test_npz_compares_values_and_identifiers(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder)/'a.npz', Path(folder)/'b.npz'
            np.savez(a, values=[0.2, np.nan], beer=['001','002'])
            np.savez_compressed(b, values=[0.2, np.nan], beer=['001','002'])
            repro.compare_npz(a, b)
            np.savez(b, values=[0.2, np.nan], beer=['001','003'])
            with self.assertRaises(AssertionError):
                repro.compare_npz(a, b)


if __name__ == '__main__':
    unittest.main()
