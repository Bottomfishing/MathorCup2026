from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reproduce_current import compare_csv


class ReproductionComparisonTests(unittest.TestCase):
    def test_embedded_excel_line_endings_do_not_change_table_meaning(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a.csv',Path(folder)/'b.csv'
            pd.DataFrame({'label':['Panel\r\nscore'],'value':[1.]}).to_csv(a,index=False,lineterminator='\n')
            pd.DataFrame({'label':['Panel\nscore'],'value':[1.+1e-12]}).to_csv(b,index=False,lineterminator='\n')
            compare_csv(a,b)

    def test_real_label_and_numeric_changes_still_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a.csv',Path(folder)/'b.csv'
            pd.DataFrame({'label':['Panel\nscore'],'value':[1.]}).to_csv(a,index=False)
            for label,value in [('Changed score',1.),('Panel\nscore',1.01)]:
                pd.DataFrame({'label':[label],'value':[value]}).to_csv(b,index=False)
                with self.assertRaises(AssertionError):compare_csv(a,b)

    def test_nullable_boolean_objects_are_not_treated_as_strings(self):
        with tempfile.TemporaryDirectory() as folder:
            a,b=Path(folder)/'a.csv',Path(folder)/'b.csv'
            pd.DataFrame({'label':['Panel\r\nscore','Other'],'flag':[True,None]}).to_csv(a,index=False,lineterminator='\n')
            pd.DataFrame({'label':['Panel\nscore','Other'],'flag':[True,None]}).to_csv(b,index=False,lineterminator='\n')
            compare_csv(a,b)


if __name__=='__main__':unittest.main()
