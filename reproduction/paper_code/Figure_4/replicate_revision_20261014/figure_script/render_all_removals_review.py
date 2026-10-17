#!/usr/bin/env python3
"""Review the frozen post-removal endpoint, changes and anatomical exclusions."""
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
from audit_regional_statistics import GATES, record, verified
from check_alignment import composite

COLORS = {'DMSO': '#4F7F8A', 'RARi': '#A55D54'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort', type=Path, required=True)
    p.add_argument('--statistics', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    inputs = [record(a.cohort), record(a.statistics), record(__file__),
              record(HOME/'frozen_preprocessing/if_replicates_20261014/check_alignment.py'),
              record(HOME/'frozen_preprocessing/if_replicates_20261014/audit_regional_statistics.py')]
    release = json.loads(a.cohort.read_text()); stat_release = json.loads(a.statistics.read_text())
    def read(rec, kind='csv'):
        inputs.append(rec); path = verified(rec)
        if kind == 'json':
            return json.loads(path.read_text())
        if kind == 'parquet':
            return pd.read_parquet(path)
        if kind == 'npz':
            return np.load(path)
        return pd.read_csv(path)
    files = {Path(r['path']).name: r for r in release['outputs']}
    stat_files = {Path(r['path']).name: r for r in stat_release['outputs']}
    plan = read(stat_files['analysis_plan.json'], 'json')
    if plan['input_manifest_sha256'] != record(a.cohort)['sha256']:
        raise ValueError('Statistics and filtered cohort differ')
    stats = read(stat_files['density_over_expected_statistics.csv'])
    mice = read(stat_files['per_mouse_observed_expected.csv'])
    primary = stats.loc[stats.analysis.eq('all_mice')].set_index('gate').loc[GATES]
    old_manifest = read(plan['previous_statistics'], 'json')
    old_files = {Path(r['path']).name: r for r in old_manifest['outputs']}
    old_stats = read(old_files['density_over_expected_statistics.csv'])
    old_primary = old_stats.loc[old_stats.analysis.eq('all_mice')].set_index('gate').loc[GATES]
    removals = read(files['removal_summary.csv'])
    fields = read(files['display_fields.json'], 'json')
    batch_rec = next(r for r in release['inputs'] if r['path'].endswith('/batch02_07_v1/manifest.json'))
    batch = read(batch_rec, 'json')
    findfont(FontProperties(family='Arial'), fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'pdf.fonttype': 42, 'axes.titlesize': 10})
    a.output.mkdir(parents=True)
    outputs, titles = [], []
    pdf_path = a.output/'IF_removals_and_density_over_expected.pdf'
    def save(pdf, fig, title):
        number = len(titles)+1; titles.append(title)
        pdf.savefig(fig, dpi=300)
        target = a.output/f'page_{number:02d}.png'
        fig.savefig(target, dpi=150); outputs.append(record(target)); plt.close(fig)
    def table(ax, rows, labels, fontsize=10):
        ax.axis('off')
        artist = ax.table(cellText=rows, colLabels=labels, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
        artist.auto_set_font_size(False); artist.set_fontsize(fontsize)
        for (r, c), cell in artist.get_celld().items():
            cell.set_edgecolor('white')
            cell.set_facecolor('#e9edef' if r == 0 else ('#f5f6f7' if r % 2 else 'white'))
            if r == 0:
                cell.set_text_props(weight='bold')
        return artist
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(12, 8.5))
        fig.text(.5, .955, 'P14 observed/expected density after tissue removals', ha='center', fontsize=17)
        fig.text(.5, .915, '4 DMSO mice · 5 RARi mice · Original pair retained · Tissue 08 excluded', ha='center', fontsize=10)
        fig.text(.5, .88, 'Observed/expected = fraction of P14 cells in a gate ÷ fraction of non-P14 cells in that gate', ha='center', fontsize=10)
        for j, gate in enumerate(GATES):
            ax = fig.add_axes([.065+j*.19, .52, .15, .28])
            ax.axhline(1, color='#777777', lw=.7, ls='--', zorder=0)
            sub = mice.loc[mice.gate.eq(gate)]
            for x, condition in enumerate(['DMSO', 'RARi']):
                group = sub.loc[sub.condition.eq(condition)].sort_values('mouse_id')
                for offset, row in zip(np.linspace(-.16, .16, len(group)), group.itertuples()):
                    original = row.cohort == 'original'
                    ax.scatter(x+offset, row.observed_over_expected_p14_density, marker='s' if original else 'o', s=30,
                               facecolor='white' if original else COLORS[condition], edgecolor=COLORS[condition], zorder=3)
                row = primary.loc[gate]; key = condition.lower()
                ax.errorbar(x+.27, row[key+'_mean'], yerr=row[key+'_sem'], fmt='_', ms=10, c='black', capsize=3, lw=1, zorder=4)
            ax.set(xticks=[0, 1], xticklabels=['DMSO', 'RARi'], xlim=(-.4, 1.5), ylim=(0, max(1.15, sub.observed_over_expected_p14_density.max()*1.18)), title=gate)
            ax.spines[['top', 'right']].set_visible(False)
            if j == 0:
                ax.set_ylabel('Observed / expected')
        fig.text(.5, .455, 'Points: individual mice; open squares: original mice; black marks: mean ± SEM. Dashed line: expected abundance.', ha='center', fontsize=9)
        rows = [[gate, f'{r.dmso_mean:.2f}', f'{r.rari_mean:.2f}', f'{r.raw_p:.3f}', f'{r.holm_adjusted_p:.3f}'] for gate, r in primary.iterrows()]
        table(fig.add_axes([.13, .17, .74, .23]), rows, ['Gate', 'DMSO mean', 'RARi mean', 'Raw P', 'Holm-adjusted P'])
        fig.text(.5, .115, 'DMSO versus RARi: exact mouse-label permutation, doubled-tail two-sided P; Holm correction across five gates.', ha='center', fontsize=9)
        fig.text(.5, .077, 'Background-cell counts account for compartment availability; physical compartment areas were not measured.', ha='center', fontsize=9)
        fig.text(.5, .04, 'Exploratory analysis. Removal annotations were supplied after earlier results were inspected; annotation blinding is undocumented.', ha='center', fontsize=8.5)
        save(pdf, fig, 'Post-removal observed/expected comparisons')
        fig = plt.figure(figsize=(12, 8.5))
        fig.text(.5, .95, 'Changes introduced by the removal annotations', ha='center', fontsize=17)
        new = removals.loc[removals.cohort.eq('new')]
        rows = [[r.source_id[-2:], r.condition, str(r.n_regions), f'{r.removed_cells:,}', f'{r.removed_p14:,}', f'{100*r.removed_p14/r.before_p14:.1f}%', f'{r.after_p14:,}'] for r in new.itertuples()]
        table(fig.add_axes([.06, .55, .88, .31]), rows, ['Tissue', 'Condition', 'Regions', 'Cells removed', 'P14 removed', '% P14 removed', 'P14 retained'], 10)
        fig.text(.5, .505, f"Total removed: {release['n_removed_cells']:,} cells, including {release['n_removed_p14']:,} P14. Original mice are unchanged.", ha='center', fontsize=10)
        rows = []
        for gate, r in primary.iterrows():
            old = old_primary.loc[gate]
            rows.append([gate, f'{old.dmso_mean:.2f} → {r.dmso_mean:.2f}', f'{old.rari_mean:.2f} → {r.rari_mean:.2f}', f'{old.holm_adjusted_p:.3f} → {r.holm_adjusted_p:.3f}'])
        table(fig.add_axes([.1, .22, .8, .23]), rows, ['Gate', 'DMSO mean O/E', 'RARi mean O/E', 'Adjusted P: before → after'], 10)
        fig.text(.5, .14, 'All exclusions use the same native-pixel centroid rule for P14 and non-P14 cells. Overlaps count once.', ha='center', fontsize=10)
        fig.text(.5, .1, 'No changes to retained predictions, gate boundaries, model checkpoints or mouse inclusion.', ha='center', fontsize=10)
        save(pdf, fig, 'Removal counts and before/after group results')
        fig = plt.figure(figsize=(12, 8.5))
        fig.text(.5, .95, 'Cohort sensitivity checks', ha='center', fontsize=17)
        for bottom, analysis, title in [(.53, 'new_only', 'New mice only: 3 DMSO and 4 RARi'), (.16, 'cohort_stratified', 'All nine mice: permutations restricted within original/new cohorts')]:
            fig.text(.5, bottom+.28, title, ha='center', fontsize=12)
            group = stats.loc[stats.analysis.eq(analysis)].set_index('gate').loc[GATES]
            rows = [[gate, f'{r.test_statistic:+.3f}', f'{r.raw_p:.3f}', f'{r.holm_adjusted_p:.3f}'] for gate, r in group.iterrows()]
            table(fig.add_axes([.16, bottom, .68, .23]), rows, ['Gate', 'DMSO − RARi statistic', 'Raw P', 'Holm-adjusted P'])
        fig.text(.5, .075, 'Stratified statistic: within-cohort mean differences weighted by cohort size. The endpoint and test definitions are unchanged.', ha='center', fontsize=9)
        save(pdf, fig, 'New-only and cohort-stratified tests')
        for sid, removal_rec in sorted(batch['releases'].items()):
            removal = read(removal_rec, 'json'); rf = {Path(r['path']).name: r for r in removal['outputs']}
            flags = read(rf['all_cells_annotation_flags.csv']); vertices = read(rf['removal_vertices.csv'])
            field = fields[sid]; assets = read(files[sid+'_microscopy.npz'], 'npz')
            image = composite(assets['dapi_overview'], assets['ecad_overview'], field['levels']['dapi_upper'], field['levels']['ecad_upper'])
            height, width = field['shape_yx']; c = removal['counts']
            fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')
            for j, ax in enumerate(axes):
                if j < 2:
                    ax.imshow(image, extent=(0, width, height, 0))
                if j == 1:
                    excluded = flags.loc[flags.remove_by_annotation]
                    ax.scatter(excluded.centroid_x_fullres_px, excluded.centroid_y_fullres_px, s=.25, c='#ff5050', linewidths=0, rasterized=True)
                    for _, ring in vertices.groupby(['region_index', 'polygon_index', 'ring_index']):
                        xy = ring.sort_values('point_index')[['x_px', 'y_px']].to_numpy(); xy = np.vstack([xy, xy[0]])
                        ax.plot(xy[:, 0], xy[:, 1], c='#ffdf00', lw=.8)
                if j == 2:
                    retained = flags.loc[flags.keep_for_analysis]
                    for value, color, size in [(0, '#738596', .2), (1, '#bd1e7b', 1.5)]:
                        sub = retained.loc[retained.is_p14.eq(value)]
                        ax.scatter(sub.centroid_x_fullres_px, sub.centroid_y_fullres_px, s=size, c=color, linewidths=0, rasterized=True)
                ax.set(xlim=(0, width), ylim=(height, 0), aspect='equal'); ax.axis('off')
                ax.set_title(['DAPI / E-cadherin', 'Removal regions and excluded cells', 'Retained cells'][j])
            fig.suptitle(f"{c['after']['condition']} tissue {sid[-2:]} — {c['removed_cells']:,} cells excluded; {c['after']['n_cells']:,} retained", fontsize=13)
            handles = [Line2D([], [], marker='o', ls='', color=col, label=name, markersize=5) for col, name in
                       [('#ff5050', 'Excluded cells'), ('#738596', 'Retained non-P14'), ('#bd1e7b', 'Retained P14')]]
            fig.legend(handles=handles, loc='outside lower center', ncol=3, frameon=False)
            save(pdf, fig, f"{c['after']['condition']} tissue {sid[-2:]} removal review")
    outputs.append(record(pdf_path))
    provenance = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
                  'python': platform.python_version(), 'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'matplotlib']},
                  'git_state': 'Project directory is not a Git worktree', 'random_seed': None, 'inputs': inputs, 'outputs': outputs,
                  'page_titles': titles, 'warnings': ['Review PDF only; manuscript figure PDFs remain unchanged.']}
    (a.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(pdf_path)


if __name__ == '__main__':
    main()
