#!/usr/bin/env python3
"""Render grouped P14 percentage bars from frozen mouse-level statistics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'frozen_preprocessing/if_replicates_20261014'))
from audit_regional_statistics import record, verified

COLORS = {'DMSO': '#37788F', 'RARi': '#B05B49'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    release = json.loads(verified({'path': str(args.manifest), 'sha256': args.manifest_sha256}).read_text())
    assert release['status'] == 'frozen_post_removal_p14_composition_statistics'
    files = {Path(r['path']).name: verified(r) for r in release['outputs']}
    plan = json.loads(files['analysis_plan.json'].read_text())
    mice = pd.read_csv(files['per_mouse_composition.csv'])
    stats = pd.read_csv(files['p14_composition_statistics.csv'])
    gates = plan['gates']
    primary = stats.loc[stats.analysis.eq('all_mice')].set_index('gate').loc[gates]
    assert len(mice) == 45 and primary.n_dmso.eq(4).all() and primary.n_rari.eq(5).all()
    for weight in ['normal', 'bold']:
        findfont(FontProperties(family='Arial', weight=weight), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.linewidth': .7, 'savefig.facecolor': 'white'})
    args.output.mkdir(parents=True)
    outputs = []
    def save(fig, stem):
        for extension in ['pdf', 'png']:
            path = args.output/f'{stem}.{extension}'
            fig.savefig(path, dpi=300)
            outputs.append(record(path))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    fig.subplots_adjust(left=.10, right=.98, bottom=.24, top=.78)
    fig.suptitle('P14 distribution across anatomical gates', y=.955, fontsize=15, fontweight='bold')
    positions = np.arange(len(gates))
    width = .32
    for j, condition in enumerate(['DMSO', 'RARi']):
        key = condition.lower()
        centers = positions + (-.18 if j == 0 else .18)
        ax.bar(centers, primary[key+'_mean'], width, yerr=primary[key+'_sem'],
               color=COLORS[condition], alpha=.38, edgecolor=COLORS[condition], linewidth=.8,
               error_kw={'elinewidth': 1, 'capsize': 3, 'capthick': 1, 'ecolor': '#333333'},
               label=f'{condition} (n = {4 if j == 0 else 5} mice)', zorder=2)
        for i, gate in enumerate(gates):
            group = mice.loc[mice.gate.eq(gate) & mice.condition.eq(condition)].sort_values('mouse_id')
            for offset, row in zip(np.linspace(-.09, .09, len(group)), group.itertuples()):
                original = row.cohort == 'original'
                ax.scatter(centers[i]+offset, row.p14_composition_percent, s=31, marker='s' if original else 'o',
                           facecolor='white' if original else COLORS[condition], edgecolor=COLORS[condition],
                           linewidth=.9, zorder=4)
    maximum = float(mice.p14_composition_percent.max())
    for i, gate in enumerate(gates):
        gate_max = mice.loc[mice.gate.eq(gate), 'p14_composition_percent'].max()
        y, height = gate_max + maximum*.075, maximum*.025
        ax.plot([i-.18, i-.18, i+.18, i+.18], [y, y+height, y+height, y], color='#444444', lw=.8)
        ax.text(i, y+height+maximum*.025, f"P = {primary.loc[gate, 'holm_adjusted_p']:.3f}", ha='center', va='bottom', fontsize=9)
    ax.set(xticks=positions, xticklabels=gates, ylabel='P14 cells in gate (% of all P14 cells)',
           xlim=(-.6, 4.6), ylim=(0, min(100, math.ceil(maximum*1.24/10)*10)))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e7e7e7', lw=.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker='s', color='#555555', markerfacecolor='white', ls='', markersize=5))
    labels.append('Original mouse')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.54, .90), frameon=False, ncol=3, fontsize=9.5)
    for y, text in [(.14, 'Bars: mouse mean ± SEM. Points: individual mice.'),
                    (.095, 'DMSO versus RARi within each gate: two-sided exact permutation; P values Holm-adjusted across five gates.'),
                    (.05, 'Percentages use each mouse’s total retained P14 count as the denominator. All tissue removals applied.')]:
        fig.text(.54, y, text, ha='center', fontsize=8)
    save(fig, 'Additional_08_P14_composition_bars')

    fig = plt.figure(figsize=(10.5, 9.8))
    fig.text(.5, .96, 'P14 gate percentages: treatment comparisons', ha='center', fontsize=15, fontweight='bold')
    for bottom, analysis, title in [(.65, 'all_mice', 'All mice: 4 DMSO, 5 RARi'),
                                     (.39, 'new_only', 'New mice only: 3 DMSO, 4 RARi'),
                                     (.13, 'cohort_stratified', 'All mice, stratified by original/new cohort')]:
        fig.text(.07, bottom+.225, title, fontsize=11, fontweight='bold')
        group = stats.loc[stats.analysis.eq(analysis)].set_index('gate').loc[gates]
        rows = [[gate, f'{r.dmso_mean:.1f} ± {r.dmso_sem:.1f}', f'{r.rari_mean:.1f} ± {r.rari_sem:.1f}',
                 f'{r.test_statistic:+.2f}', f'{r.raw_p:.3f}', f'{r.holm_adjusted_p:.3f}'] for gate, r in group.iterrows()]
        axis = fig.add_axes([.07, bottom, .86, .20]); axis.axis('off')
        table = axis.table(cellText=rows, colLabels=['Gate', 'DMSO mean ± SEM', 'RARi mean ± SEM',
                          'Test statistic (pp)', 'Raw P', 'Adjusted P'], cellLoc='center', bbox=[0, 0, 1, 1],
                          colWidths=[.15, .21, .21, .19, .12, .12])
        table.auto_set_font_size(False); table.set_fontsize(9)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_facecolor('#e6ecef' if row == 0 else ('#f4f6f7' if row % 2 else 'white'))
            if row == 0:
                cell.set_text_props(weight='bold')
    for y, text in [(.085, 'Exact mouse-label permutations: 126 all-mice, 35 new-only and 70 cohort-stratified assignments. Holm covers five gates per analysis.'),
                    (.058, 'Stratified statistic: cohort-size-weighted DMSO − RARi difference. Group means are unadjusted; pp = percentage points.'),
                    (.031, 'Exploratory comparisons. Randomization and removal-annotation blinding are undocumented; original mice contributed to model development.')]:
        fig.text(.5, y, text, ha='center', fontsize=7.8)
    save(fig, 'Additional_09_P14_composition_statistics')
    provenance = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
                  'python': platform.python_version(), 'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'matplotlib']},
                  'git_state': 'Not a Git worktree', 'random_seed': None, 'point_offsets': 'Deterministic by sorted mouse ID',
                  'inputs': [record(args.manifest), *release['outputs'], record(__file__),
                             record(ROOT/'frozen_preprocessing/if_replicates_20261014/audit_regional_statistics.py')],
                  'outputs': outputs, 'warnings': release['warnings'], 'tests_computed_during_render': False,
                  'paper_figures_modified': False}
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'pdfs': [r['path'] for r in outputs if r['path'].endswith('.pdf')]}))


if __name__ == '__main__':
    main()
