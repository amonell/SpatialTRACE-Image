#!/usr/bin/env python3
"""Render density fold-change bars from frozen experiment-blocked statistics."""
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'frozen_preprocessing/if_replicates_20261014'))
from audit_regional_statistics import record, verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    release = json.loads(verified({'path': str(args.manifest), 'sha256': args.manifest_sha256}).read_text())
    assert release['status'] == 'frozen_post_removal_p14_density_fold_change'
    files = {Path(rec['path']).name: verified(rec) for rec in release['outputs']}
    plan = json.loads(files['analysis_plan.json'].read_text())
    gates = plan['gates']
    stats = pd.read_csv(files['density_fold_change.csv']).set_index('gate').loc[gates]
    experiments = pd.read_csv(files['experiment_fold_change.csv'])
    for weight in ['normal', 'bold']:
        findfont(FontProperties(family='Arial', weight=weight), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.linewidth': .7, 'savefig.facecolor': 'white'})
    positions = np.arange(len(gates))
    fig, ax = plt.subplots(figsize=(8.4, 5.5))
    fig.subplots_adjust(left=.12, right=.97, bottom=.255, top=.77)
    fig.suptitle('P14 density: RARi relative to DMSO', y=.955, fontsize=15, fontweight='bold')
    color = '#B05B49'
    ax.bar(positions, stats.fold_change_rari_over_dmso, width=.55,
           facecolor='#DFC2BB', edgecolor=color, linewidth=1, zorder=2)
    for cohort, marker, offset, face in [('original', 's', -.11, 'white'), ('new', 'o', .11, '#343434')]:
        points = experiments.loc[experiments.cohort.eq(cohort)].set_index('gate').loc[gates]
        ax.scatter(positions+offset, points.fold_change_rari_over_dmso, marker=marker, s=34,
                   facecolor=face, edgecolor='#343434', linewidth=1, zorder=4)
    ymax = max(1.0, float(experiments.fold_change_rari_over_dmso.max()), float(stats.fold_change_rari_over_dmso.max()))
    ylim = math.ceil(ymax*1.3/.2)*.2
    ax.axhline(1, color='#626262', lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(4.5, 1+.025*ylim, 'Equal density', ha='right', va='bottom', fontsize=8, color='#555555')
    for i, gate in enumerate(gates):
        value = stats.loc[gate, 'fold_change_rari_over_dmso']
        top = max(value, experiments.loc[experiments.gate.eq(gate), 'fold_change_rari_over_dmso'].max())
        ax.text(i, top+.045*ylim, f'{value:.2f}×', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(i, .96, f"P = {stats.loc[gate, 'holm_adjusted_p']:.3f}",
                transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8.5)
    ax.set(xticks=positions, xticklabels=gates, ylabel='P14 density fold change (RARi / DMSO)',
           xlim=(-.55, 4.55), ylim=(0, ylim))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e8e8e8', lw=.6)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor='#DFC2BB', edgecolor=color, label='Batch-adjusted ratio'),
               Line2D([], [], ls='', marker='s', color='#343434', markerfacecolor='white', markersize=5,
                      label='Original experiment'),
               Line2D([], [], ls='', marker='o', color='#343434', markersize=5, label='New experiment')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .89), frameon=False, ncol=3, fontsize=9)
    fig.text(.54, .17, 'Density = P14 cells per 10,000 non-P14 cells in the same gate.', ha='center', fontsize=9)
    fig.text(.54, .12, 'Bars: experiment-standardized mean ratios. Points: ratios within each experiment.', ha='center', fontsize=8)
    fig.text(.54, .075, '4 DMSO and 5 RARi mice; all tissue removals applied.', ha='center', fontsize=8)
    fig.text(.54, .035, 'P: exact within-experiment permutation, Holm-adjusted across five gates.', ha='center', fontsize=8)
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = []
    for extension in ['pdf', 'png']:
        path = args.output/f'Additional_10_P14_density_fold_change.{extension}'
        fig.savefig(path, dpi=300)
        outputs.append(record(path))
    plt.close(fig)
    provenance = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
                  'python': platform.python_version(),
                  'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'matplotlib']},
                  'git_state': 'Not a Git worktree', 'random_seed': None,
                  'inputs': [record(args.manifest), *release['outputs'], record(__file__),
                             record(ROOT/'frozen_preprocessing/if_replicates_20261014/audit_regional_statistics.py')],
                  'outputs': outputs, 'warnings': release['warnings'],
                  'tests_computed_during_render': False, 'paper_figures_modified': False}
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'outputs': outputs}))


if __name__ == '__main__':
    main()
