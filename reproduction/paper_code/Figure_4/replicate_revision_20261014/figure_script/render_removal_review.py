#!/usr/bin/env python3
"""Render a single-tissue removal review from frozen masks and microscopy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HOME = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HOME/'frozen_preprocessing/if_replicates_20261014'))
from audit_regional_statistics import record, verified
from check_alignment import composite


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--removals', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    release = json.loads(a.removals.read_text()); sid = release['source_id']
    if release['status'] != 'frozen_single_tissue_removals_applied':
        raise ValueError('Removals not frozen')
    inputs = [record(a.removals), record(__file__), record(HOME/'frozen_preprocessing/if_replicates_20261014/check_alignment.py'),
              record(HOME/'frozen_preprocessing/if_replicates_20261014/audit_regional_statistics.py')]
    files = {Path(r['path']).name: r for r in release['outputs']}
    def read(name):
        rec = files[name]; inputs.append(rec)
        return pd.read_csv(verified(rec))
    flags, vertices, regions = [read(n) for n in ['all_cells_annotation_flags.csv', 'removal_vertices.csv', 'region_counts.csv']]
    base_rec = next(r for r in release['inputs'] if '/nine_mouse_release_v2/manifest.json' in r['path'])
    inputs.append(base_rec); base = json.loads(verified(base_rec).read_text())
    base_files = {Path(r['path']).name: r for r in base['outputs']}
    field_rec = base_files['display_fields.json']; inputs.append(field_rec)
    field = json.loads(verified(field_rec).read_text())[sid]
    arrays = {}
    for channel in ['dapi', 'ecad']:
        rec = next(r for r in base['inputs'] if f'/images/{sid}/{channel}.npy' in r['path'])
        inputs.append(rec); arrays[channel] = np.load(verified(rec), mmap_mode='r')
    height, width = field['shape_yx']; levels = field['levels']
    assert arrays['dapi'].shape == arrays['ecad'].shape == (height, width)
    def image(x0=0, y0=0, x1=width, y1=height, stride=1):
        return composite(arrays['dapi'][y0:y1:stride, x0:x1:stride], arrays['ecad'][y0:y1:stride, x0:x1:stride],
                         levels['dapi_upper'], levels['ecad_upper'])
    def outlines(ax, table, labels=False):
        for (region, _, _), ring in table.groupby(['region_index', 'polygon_index', 'ring_index']):
            xy = ring.sort_values('point_index')[['x_px', 'y_px']].to_numpy()
            xy = np.vstack([xy, xy[0]])
            ax.plot(xy[:, 0], xy[:, 1], c='#ffdf00', lw=.8)
            if labels:
                ax.text(xy[:, 0].mean(), xy[:, 1].mean(), str(region+1), ha='center', va='center',
                        fontsize=8, color='black', bbox={'facecolor': '#ffdf00', 'edgecolor': 'none', 'pad': 1.5})
    findfont(FontProperties(family='Arial'), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42, 'font.size': 9})
    a.output.mkdir(parents=True)
    destination = a.output/'Tissue01_removal_review.pdf'
    outputs = []
    with PdfPages(destination) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')
        overview = image(stride=24)
        for j, ax in enumerate(axes):
            if j < 2:
                ax.imshow(overview, extent=(0, width, height, 0))
            if j == 1:
                removed = flags.loc[flags.remove_by_annotation]
                ax.scatter(removed.centroid_x_fullres_px, removed.centroid_y_fullres_px, c='#ff5050', s=.25,
                           linewidths=0, rasterized=True)
                outlines(ax, vertices, labels=True)
            if j == 2:
                retained = flags.loc[flags.keep_for_analysis]
                for value, color, size in [(0, '#738596', .2), (1, '#bd1e7b', 1.5)]:
                    sub = retained.loc[retained.is_p14.eq(value)]
                    ax.scatter(sub.centroid_x_fullres_px, sub.centroid_y_fullres_px, c=color, s=size,
                               linewidths=0, rasterized=True)
            ax.set(xlim=(0, width), ylim=(height, 0), aspect='equal'); ax.axis('off')
            ax.set_title(['DAPI / E-cadherin', 'Removal regions and excluded cells', 'Retained cells'][j])
        c = release['counts']
        fig.suptitle(f"RARi tissue 01 — {c['removed_cells']:,} cells excluded; {c['after']['n_cells']:,} retained", fontsize=13)
        handles = [Line2D([], [], marker='o', ls='', color=col, label=name, markersize=5)
                   for col, name in [('#ff5050', 'Excluded cells'), ('#738596', 'Retained non-P14'), ('#bd1e7b', 'Retained P14')]]
        fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False)
        pdf.savefig(fig, dpi=300)
        path = a.output/'Tissue01_removal_overview.png'; fig.savefig(path, dpi=180); outputs.append(record(path)); plt.close(fig)
        fig, axes = plt.subplots(2, 5, figsize=(16, 9), layout='constrained')
        fig.suptitle('All 10 annotated removal regions — native DAPI / E-cadherin fields', fontsize=13)
        for ax, (region, table) in zip(axes.flat, vertices.groupby('region_index', sort=True)):
            x0 = max(0, int(table.x_px.min())-150); x1 = min(width, int(table.x_px.max())+150)
            y0 = max(0, int(table.y_px.min())-150); y1 = min(height, int(table.y_px.max())+150)
            stride = max(1, int(np.ceil(max(x1-x0, y1-y0)/1600)))
            ax.imshow(image(x0, y0, x1, y1, stride), extent=(x0, x1, y1, y0))
            outlines(ax, table)
            row = regions.loc[regions.region_index.eq(region)].iloc[0]
            ax.set_title(f"Region {region+1}\n{int(row.n_cells):,} cells; {int(row.n_p14):,} P14", fontsize=9)
            ax.set(xlim=(x0, x1), ylim=(y1, y0), aspect='equal'); ax.axis('off')
        fig.text(.5, .005, 'Region counts overlap; the filtered table counts each excluded cell only once.', ha='center', fontsize=9)
        pdf.savefig(fig, dpi=300)
        path = a.output/'Tissue01_removal_regions.png'; fig.savefig(path, dpi=160); outputs.append(record(path)); plt.close(fig)
    outputs.append(record(destination))
    (a.output/'provenance.json').write_text(json.dumps({
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
        'python': platform.python_version(), 'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'matplotlib']},
        'git_state': 'Project directory is not a Git worktree', 'random_seed': None,
        'inputs': inputs, 'outputs': outputs, 'warnings': ['Review only; does not replace any manuscript panel.'],
        'display': 'Unchanged frozen slide levels; cyan E-cadherin and grayscale DAPI; no image masking or prediction changes.'
    }, indent=2)+'\n')
    print(destination)


if __name__ == '__main__':
    main()
