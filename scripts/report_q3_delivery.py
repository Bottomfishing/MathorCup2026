"""Plot existing Q3 delivery tables without running new experiments."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run(delivery, q3, audit):
    table = pd.read_csv(delivery / 'sensory_fingerprints.csv')
    mapping = pd.read_csv(audit / 'sensory_mapping.csv')
    targets = mapping.loc[mapping.scope.eq('aroma_flavour_primary_candidate'), 'R_name'].tolist()
    styles = ['Tripel', 'Kriek', 'Lambic']
    generated = table.pivot(index='target', columns='style', values='generated_prediction_mean').loc[targets, styles].to_numpy()
    observed = table.pivot(index='target', columns='style', values='real_observed_mean').loc[targets, styles].to_numpy()
    limit = max(np.abs(generated).max(), np.abs(observed).max())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 12), sharey=True)
    for ax, values, title in zip(axes, [generated, observed], ['Generated profiles: predicted means', 'Real style samples: observed means']):
        im = ax.imshow(values, vmin=-limit, vmax=limit, cmap='RdBu_r', aspect='auto')
        ax.set_xticks(range(3), styles); ax.set_yticks(range(len(targets)), targets, fontsize=8)
        ax.set_title(title, fontsize=10)
        for i in range(len(targets)):
            for j in range(3):
                ax.text(j, i, f'{values[i,j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if abs(values[i,j]) > .6 * limit else '#202020')
    fig.suptitle('Sensory fingerprints in standardized score coordinates', fontsize=12)
    fig.subplots_adjust(left=.21, right=.89, top=.94, bottom=.07, wspace=.12)
    cax = fig.add_axes([.92, .2, .017, .6]); fig.colorbar(im, cax=cax, label='Standardized sensory score')
    fig.text(.21, .025, '100 generated profiles per style. Predictions are not panel measurements or recovered 0-6 ratings.', fontsize=8)
    for ext in ['png', 'svg']:
        fig.savefig(delivery / ('sensory_fingerprints.' + ext), dpi=170)
    plt.close(fig)
    transfer = pd.read_csv(q3 / 'transfer_results.csv', dtype={'beer': str})
    transfer = transfer.loc[transfer.selected].copy()
    fig, ax = plt.subplots(figsize=(8, 5.8)); y = np.arange(len(transfer)); height = .35
    ax.barh(y - height / 2, transfer.source_target_distance, height, label='Source', color='#9ba9b2')
    ax.barh(y + height / 2, transfer.target_distance, height, label='Selected model design', color='#2b7a78')
    ax.set_yticks(y, ['Beer ' + b for b in transfer.beer]); ax.invert_yaxis()
    ax.set_xlabel('RMS distance to Tripel center in fitted asinh coordinates')
    ax.set_title('Pils/Lager to Tripel: distance reduction under model constraints', pad=36, fontsize=11)
    ax.legend(loc='lower right', bbox_to_anchor=(1, 1.01), ncol=2, frameon=False, fontsize=8)
    ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    fig.text(.14, .025, '9 designs pass proxy constraints; 2 sources have missing defect fields. Style is not authenticated.', fontsize=8)
    fig.subplots_adjust(left=.14, right=.97, top=.83, bottom=.14)
    for ext in ['png', 'svg']:
        fig.savefig(delivery / ('transfer_distances.' + ext), dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--delivery', type=Path, default=Path('outputs/q3_delivery_20261024'))
    p.add_argument('--q3', type=Path, default=Path('outputs/final_20261024/q3'))
    p.add_argument('--audit', type=Path, default=Path('outputs/p0_audit_20261023_final'))
    args = p.parse_args(); run(args.delivery, args.q3, args.audit)
