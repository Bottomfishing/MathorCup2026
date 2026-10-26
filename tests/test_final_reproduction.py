import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


spec = importlib.util.spec_from_file_location('final_reproduction',
    Path(__file__).resolve().parents[1]/'scripts/reproduce_final.py')
repro = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repro)


class FinalComparisonTests(unittest.TestCase):
    def test_unlabelled_oof_rows_do_not_claim_fitted_calibration(self):
        frame = pd.read_csv(Path(__file__).resolve().parents[1]/
            'outputs/r2b_tail_risk_20261026/oof_scores_all52.csv')
        missing = frame.target.isin(['A_acetaldehyde','F_acetaldehyde'])
        self.assertEqual(int(missing.sum()), 500)
        columns = ['estimated_grade_0_6','probability_rating_ge1','probability_rating_ge5',
                   'probability_rating_6','calibration_quality_status']
        self.assertTrue(frame.loc[missing,columns].isna().all().all())

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
