# Q3 跨风格迁移（正式跑，2026-10-28）

- 口径：题面“缺陷以所列超阈值项为准”；三口径 A（超阈集合不扩张+严重度不增）、B（逐项 e_after≤e_before）、C（严重度加总不增），median 与 mean 两种阈值情景均评估。
- 结果：Pils/Lager 11 款 × Tripel/Kriek/Lambic 3 目标 = 33 条；A/B/C 各 33/33 条通过；候选级 660 个 λ 候选全通过。
- λ*：Tripel 0.5/0.55，Kriek/Lambic 均为 1.0；距目标质心平均缩短 12.50% / 99.35% / 81.17%。
- 119 号 Tripel：4VG 0.1438→0.0405 mg/L，DMS 保持 16.661 µg/L，双乙酰源缺失按 min(目标中位,0.9T)=0.081 mg/L（条件性）。
- Lambic 仅 2/11 条迁移后最近质心为 Lambic，如实记录；风格可辨识度最近质心 LOO=0.132（多数类基线 0.124，22 类）。
- 证据分级（s2c_evidence_by_case.csv）：claim_level {'conditional': 24, 'model_supported': 9}；风格诊断支持数 Tripel 0/11、Kriek 11/11、Lambic 2/11，Tripel 与 Lambic 标“风格诊断未支持”；缺陷状态 {'DMS': {'unchanged': 24, 'improved': 9}, 'Diacetyl': {'unchanged': 18, 'improved': 9, 'unknown': 6}, '4VG': {'unchanged': 33}, 't2n': {'unknown': 33}, 'Stale_hops': {'unknown': 33}}。
- 完整谱与感官画像：s2c_migration_spectra_224_long.csv（66 行）、s2c_migration_spectra_224_wide.csv（33 行）、s2c_migration_after_input_33.csv（33×227）、s2c_sensory_before_after_52.csv（3432 行 = 66 谱 × 52 目标，官方 Q1 入口 beer_models.inference.run）。
- 画像可复现性（N2 修复）：用存储谱 + 当前 q1.joblib（sha256 3312c314bf26ca8f…）重放，3432 行逐格比对 mismatch=0、max|Δ|=1.1e-15（容差 1e-12）；表 sha256 a36621b787e18f72…。
- 正式口径：estimated_grade_0_6 为决策层 top-k + 24 对父子 cap；prob_ge5/prob_6 为决策层概率，risk-head 概率另存 prob_ge5_risk_head/prob_6_risk_head。
- t2n 无阈值、Stale hops 无化学坐标：状态记 unknown，不属预注册 A/B/C 范围。

诚实说明：迁移谱是附件 1 坐标体系内的统计目标，不是酿造配方或实测酒；投影只认证所列缺陷坐标（0.9×阈值余量），不认证可酿造性与感官接受度。风格诊断整体很弱（LOO≈多数类基线），因此 Tripel/Lambic 的迁移只能表述为“化学判据通过的候选”，不是“已成功迁移为该风格”。
