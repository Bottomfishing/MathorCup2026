# -*- coding: utf-8 -*-
"""Inspect raw attachments for Q1: shapes, missingness, columns."""
import sys, os, json
import numpy as np
import pandas as pd

SRC = r'C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件'
OUT = os.path.dirname(os.path.abspath(__file__))
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 80)

def csv_report(name, max_cols=8):
    p = os.path.join(SRC, name)
    df = pd.read_csv(p, encoding='utf-8-sig')
    n, m = df.shape
    num = df.select_dtypes(include=[np.number])
    miss = df.isna().sum().sum()
    print(f'== {name}: shape={n}x{m}, numeric_cols={num.shape[1]}, missing_cells={miss} ({miss/(n*m)*100:.4f}%)')
    print('  head cols:', list(df.columns[:max_cols]))
    # per-column missing
    cm = df.isna().sum()
    cm = cm[cm > 0].sort_values(ascending=False)
    if len(cm):
        print('  cols with NA (top10):')
        for k, v in cm.head(10).items():
            print(f'    {k}: {v} ({v/n*100:.2f}%)')
    return df

def xlsx_report(name):
    p = os.path.join(SRC, name)
    xl = pd.ExcelFile(p)
    print(f'== {name}: sheets={xl.sheet_names}')
    for sh in xl.sheet_names:
        df = pd.read_excel(p, sheet_name=sh, header=None)
        nonnull = df.notna().sum().sum()
        print(f'  [{sh}] shape={df.shape} nonnull_cells={nonnull}')
    return xl

if __name__ == '__main__':
    print('numpy', np.__version__, 'pandas', pd.__version__)
    df1 = csv_report('1.csv')
    df2 = csv_report('2.csv')
    df4 = csv_report('4.csv')
    df5 = csv_report('5.csv')
    df6 = csv_report('6.csv')
    df7 = csv_report('7.csv')
    x3 = xlsx_report('3.xlsx')
    x8 = xlsx_report('8.xlsx')
