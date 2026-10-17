#!/usr/bin/env python3
"""Render a nine-SI, five-gate heatmap from frozen row-normalized densities."""
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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.font_manager import FontProperties, findfont
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
    assert release['status'] == 'frozen_post_removal_p14_density_heatmap'
    files = {Path(r['path']).name: verified(r) for r in release['outputs']}
    definition = json.loads(files['definition.json'].read_text())
    table = pd.read_csv(files['density_row_normalized.csv'], index_col='mouse_id').loc[definition['row_order'], definition['column_order']]
    rows = pd.read_csv(files['rows.csv']).set_index('mouse_id').loc[table.index]
    values = table.to_numpy()
    assert values.shape == (9, 5) and np.isfinite(values).all()
    np.testing.assert_allclose(values.mean(axis=1), 1)
    for weight in ['normal', 'bold']:
        findfont(FontProperties(family='Arial', weight=weight), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.linewidth': .7, 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(9.4, 6.8))
    ax = fig.add_axes([.24, .205, .57, .625])
    cmap = plt.get_cmap('RdBu_r')
    vmax = max(1.5, math.ceil(float(values.max())*2)/2)
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=vmax)
    tiles = ax.pcolormesh(np.arange(6), np.arange(10), values, cmap=cmap, norm=norm,
                         edgecolors='white', linewidth=.9, rasterized=False)
    ax.set(xlim=(0, 5), ylim=(9, 0), xticks=np.arange(5)+.5, yticks=np.arange(9)+.5,
           xticklabels=table.columns, yticklabels=rows.display_label)
    ax.xaxis.tick_top()
    ax.tick_params(axis='both', length=0, pad=9)
    for tick, condition in zip(ax.get_yticklabels(), rows.condition):
        tick.set_color({'DMSO': '#37788F', 'RARi': '#B05B49'}[condition])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(4, color='white', lw=5, zorder=4)
    for i in range(9):
        for j in range(5):
            value = values[i, j]
            rgb = np.asarray(cmap(norm(value))[:3])
            linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
            luminance = linear @ np.array([.2126, .7152, .0722])
            ax.text(j+.5, i+.5, f'{value:.2f}', ha='center', va='center', fontsize=10,
                    color='white' if luminance < .35 else '#202020')
    cax = fig.add_axes([.845, .205, .022, .625])
    cb = fig.colorbar(tiles, cax=cax, ticks=np.arange(0, vmax+.01, .5))
    cb.set_label('Density / row mean', labelpad=10)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=3, width=.6)
    fig.suptitle('P14 density across gates in individual SIs', y=.955, fontsize=15, fontweight='bold')
    fig.text(.53, .132, 'Density = P14 cells per 10,000 non-P14 cells in the same gate.', ha='center', fontsize=9)
    fig.text(.53, .085, 'Each row is divided by its mean across the five gates; 1 = that SI’s mean gate density.', ha='center', fontsize=8.7)
    fig.text(.53, .040, 'Original: earlier experiment. Numbered SIs: new experiment. All tissue removals applied.', ha='center', fontsize=8.2)
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = []
    for extension in ['pdf', 'png']:
        path = args.output/f'Additional_11_P14_density_row_normalized_heatmap.{extension}'
        fig.savefig(path, dpi=300)
        outputs.append(record(path))
    plt.close(fig)
    provenance = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
                  'python': platform.python_version(),
                  'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'matplotlib']},
                  'git_state': 'Not a Git worktree', 'random_seed': None,
                  'inputs': [record(args.manifest), *release['outputs'], record(__file__),
                             record(ROOT/'frozen_preprocessing/if_replicates_20261014/audit_regional_statistics.py')],
                  'outputs': outputs, 'color_scale': {'minimum': 0, 'center': 1, 'maximum': vmax, 'clipped_values': 0},
                  'warnings': release['warnings'], 'tests_computed_during_render': False, 'paper_figures_modified': False}
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps({'outputs': outputs, 'color_max': vmax}))


if __name__ == '__main__':
    main()
