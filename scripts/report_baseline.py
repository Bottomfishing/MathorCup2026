"""Generate paper-support figures from saved results only.

AI-assisted: OpenAI Codex (GPT-6), OpenAI, 2026-10-23.
"""
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

LABELS={'mean':'Mean baseline','ridge':'Ridge (inner CV)','pls':'PLS (inner CV)',
        'extra_trees':'Extra Trees','gradient_boosting':'Gradient Boosting',
        'ridge_asinh':'Robust Ridge (inner CV)','pls_asinh':'Robust PLS (inner CV)'}
COLORS={'mean':'#888888','ridge':'#8b5a2b','pls':'#56877a','extra_trees':'#2463a7','gradient_boosting':'#b84a45',
        'ridge_asinh':'#b98a55','pls_asinh':'#80a89d'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--additional',type=Path)
    parser.add_argument('--report-output',type=Path)
    args=parser.parse_args();sources=[args.output]
    if args.additional:
        sources.append(args.additional)
        pd.testing.assert_frame_equal(pd.read_csv(sources[0]/'split_manifest.csv'),pd.read_csv(sources[1]/'split_manifest.csv'))
    out=args.report_output or args.output
    out.mkdir(parents=True,exist_ok=True)
    summary=pd.concat([pd.read_csv(p/'metrics_summary.csv') for p in sources],ignore_index=True)
    target=pd.concat([pd.read_csv(p/'metrics_by_target.csv') for p in sources],ignore_index=True)
    stress=pd.concat([pd.read_csv(p/'missingness_robustness.csv') for p in sources],ignore_index=True)
    if summary.duplicated(['model','scope']).any():raise ValueError('Overlapping model identities in comparison')
    if args.report_output:
        summary.to_csv(out/'metrics_summary.csv',index=False,encoding='utf-8-sig')
        target.to_csv(out/'metrics_by_target.csv',index=False,encoding='utf-8-sig')
        stress.to_csv(out/'missingness_robustness.csv',index=False,encoding='utf-8-sig')
    (out/'report_sources.json').write_text(json.dumps([str(p.resolve()) for p in sources],indent=2),encoding='utf-8')
    primary=summary.loc[summary.scope.eq('primary_45')].copy()
    figures=out/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    labels=[LABELS[m] for m in primary.model]
    for ax,metric,title in zip(axes,['macro_mae','macro_r2'],['Mean absolute error (lower is better)','Mean R² across targets (higher is better)']):
        values=primary[metric].to_numpy()
        ax.barh(labels,values,color=[COLORS[m] for m in primary.model])
        ax.invert_yaxis();ax.axvline(0,color='#333333',lw=.7)
        ax.set_title(title,fontsize=11)
        ax.set_xlabel('Provided standardized score units' if metric=='macro_mae' else 'Dimensionless R²')
        ax.set_ylabel('Model')
        for i,v in enumerate(values):ax.text(v+(.006 if metric=='macro_mae' else .012),i,f'{v:.3f}',va='center',fontsize=9)
        ax.set_xlim(min(0,values.min()*1.3),values.max()+.13*(values.max()-min(0,values.min())))
    fig.suptitle('Exploratory baseline: 3-fold out-of-fold predictions, 45 Aroma/Taste targets',fontsize=12)
    for ext in ['png','svg']:fig.savefig(figures/f'model_comparison.{ext}',dpi=180)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,5))
    fig.subplots_adjust(left=.10,right=.68,top=.89,bottom=.22)
    selected=stress.loc[stress.scope.eq('primary_45')]
    for model in primary.model:
        base=float(primary.loc[primary.model.eq(model),'macro_mae'].iloc[0])
        group=selected.loc[selected.model.eq(model)].groupby('additional_missing_rate')['macro_mae'].agg(['mean','std'])
        xs=np.r_[0,group.index.to_numpy()]*100
        ys=np.r_[base,group['mean'].to_numpy()]
        errs=np.r_[0,group['std'].to_numpy()]
        ax.errorbar(xs,ys,yerr=errs,marker='o',capsize=3,label=LABELS[model],color=COLORS[model])
    ax.set_title('Additional random missingness in held-out features',fontsize=12)
    ax.set_xlabel('Fraction of originally observed test cells masked (%)')
    ax.set_ylabel('Macro MAE, 45 targets (standardized score units)')
    ax.legend(fontsize=9,loc='upper left',bbox_to_anchor=(1.02,1),frameon=False);ax.grid(axis='y',alpha=.2)
    fig.text(.5,.025,'Same masks across models; error bars = SD over 3 masks, not a confidence interval.',ha='center',fontsize=8)
    for ext in ['png','svg']:fig.savefig(figures/f'missingness_robustness.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

    # Full target coverage, not only the best-performing outcomes.
    pivot=target.pivot(index='target',columns='model',values='r2')
    pivot=pivot[[m for m in LABELS if m!='mean' and m in pivot.columns]]
    fig,ax=plt.subplots(figsize=(10,13),layout='constrained')
    low=min(-.5,float(pivot.min().min()))
    im=ax.imshow(pivot.to_numpy(),aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=low,vcenter=0,vmax=1))
    ax.set_yticks(range(len(pivot)),pivot.index,fontsize=8)
    ax.set_xticks(range(len(pivot.columns)),[LABELS[m] for m in pivot.columns],rotation=20,ha='right',fontsize=9)
    ax.set_ylabel('All 50 supplied sensory targets');ax.set_xlabel('Model')
    ax.set_title('Out-of-fold R² by target; negative values are retained')
    fig.colorbar(im,ax=ax,label='R² (dimensionless)',shrink=.65)
    for ext in ['png','svg']:fig.savefig(figures/f'target_r2.{ext}',dpi=160)
    plt.close(fig)
    base=float(primary.loc[primary.model.eq('mean'),'macro_mae'].iloc[0])
    lines=[]
    for row in primary.itertuples():
        lines.append(f'- {LABELS[row.model]}：MAE={row.macro_mae:.4f}，RMSE={row.macro_rmse:.4f}，宏平均R²={row.macro_r2:.4f}；相对均值基线MAE降低 {(1-row.macro_mae/base)*100:.1f}%。')
    below=target.loc[target.model.ne('mean')&target.primary&target.r2.lt(0)].groupby('model').size().to_dict()
    report='''# P1 初步基线结果

本结果是标准化目标上的探索性模型比较，不是正式0～6分评分，不是获奖效果保证。

## 验证协议

250款啤酒、231个输入特征、50个目标；主评价范围为名称表中45个Aroma/Taste候选目标。
风格分层三折外层验证，每个样本只在未参与该模型拟合的外层验证折获得一次预测。
缺失值中位数填补、Ridge缩放和PLS内部缩放均在训练折执行。
Ridge及PLS的参数由外层训练集内部两折验证选择（选择指标为全部50维的平均MAE）；树模型配置在本轮运行前固定。
所有模型使用同一外层划分。风格和ID不作为预测输入。

## 主目标结果

'''+ '\n'.join(lines)+f'''

主目标中R²为负的维度数量（未隐藏）：{json.dumps(below,ensure_ascii=False)}。
逐目标指标见 metrics_by_target.csv；所有样本外预测见 oof_predictions.csv。

## 缺失压力测试

仅在外层验证样本中，对原有观测特征额外遮蔽5%、10%、20%，每档3次。
同折、同重复、同缺失率的各模型使用相同遮蔽位置。没有用验证数据重新填补或训练。
误差条为三种遮蔽的标准差，不是置信区间；随机缺失实验不能证明真实缺失机制。

## 解读限制

- 原始标签已由上游按品评员标准化，缺少原始数据时无法消除这一步潜在的全数据处理影响。
- MAE的单位是附件提供的标准化评分，不能说成“0～6分上的误差”。
- 这是一次外层划分下的探索比较。依据本轮比较挑选出的最佳模型，不拥有额外独立测试集上的成绩。
- 若图中包含Robust模型，它们是观察首轮外推问题后追加的探索实验，不能视为独立确认；变换仅在各训练折拟合，不删除样本。
- 多维宏平均可能掩盖难预测的维度，应同时交付逐维度指标和负R²结果。
- 风格内数据很少，分风格结果只作诊断。当前验证覆盖已有风格中的样本，不证明能预测未知风格。

## 图表

figures/model_comparison.png：45维主目标模型比较。
figures/missingness_robustness.png：额外缺失下的误差变化。
figures/target_r2.png：全部50维R²；PNG与SVG均由保存的结果表生成。
所有图表均为AI辅助程序生成，需队员独立复核后决定论文采纳范围。
'''
    report+='\n实验来源目录（完整OOF预测和模型保存在这些目录中）：\n'+''.join(f'- `{p.resolve()}`\n' for p in sources)
    (out/'baseline_report.md').write_text(report,encoding='utf-8')
    print('Saved report and 3 figures in PNG/SVG:',out)

if __name__=='__main__':main()
