# -*- coding: utf-8 -*-
"""Check inner_predictions columns, final protocol feature list, 4.csv columns vs 3.xlsx summary."""
import os, json
import pandas as pd, numpy as np
os.chdir(r'C:\Users\wb198\Desktop\MathorCup数学应用挑战赛文件')

ip = pd.read_csv(r'outputs\r2_raw_q1_20261026\inner_predictions.csv.gz', nrows=2000)
print('== inner_predictions sample', ip.shape)
print(ip.columns.tolist())
print(ip.head(3).to_string())
print('methods', ip.method.unique()[:5] if 'method' in ip.columns else 'no method')
if 'method' in ip.columns:
    print(ip.groupby('method').size().head(20))

p = json.load(open(r'configs\final_protocol.json', encoding='utf-8'))
print('\n== final_protocol keys:', list(p.keys()))
print(json.dumps(p, ensure_ascii=False)[:2500])

# 4.csv columns vs 3.xlsx summary names
src = r'C:\Users\wb198\Desktop\2026年第七届MathorCup数学应用挑战赛—大数据竞赛初赛赛题\2026年MathorCup大数据挑战赛-赛道A初赛\附件'
s = pd.read_excel(os.path.join(src, '3.xlsx'), sheet_name='summary', header=None)
names = [str(x) for x in s.iloc[0].tolist()]
print('\n== 3.xlsx summary names (%d):' % len(names), names)
d4 = pd.read_csv(os.path.join(src, '4.csv'), encoding='utf-8-sig')
print('\n== 4.csv cols (%d):' % d4.shape[1], d4.columns.tolist())
print('4.csv desc:'); print(d4.describe().T.head(12).to_string())

# summary nonzero cells
vals = s.values
nz = [(i + 1, j + 1, vals[i, j]) for i in range(vals.shape[0]) for j in range(vals.shape[1]) if pd.notna(vals[i, j]) and vals[i, j] != 0]
print('\n== summary nonzero cells:', nz[:30], 'count', len(nz))
