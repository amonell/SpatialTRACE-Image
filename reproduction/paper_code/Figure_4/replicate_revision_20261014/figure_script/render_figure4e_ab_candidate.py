#!/usr/bin/env python3
"""Render only the selected abundance/enrichment plots as a Figure 4e candidate."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib.metadata
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'frozen_preprocessing/if_replicates_20261014'))
from export_cell_plotting_data import record, verified
from render_user_rari_positioning import setup

GATES = ['Top IE', 'Top LP', 'Crypt IE', 'Crypt LP', 'Muscularis']
CONDITIONS = ['DMSO', 'RARi']


def treatment_palette(source):
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == 'CONDITION_COLORS' for target in node.targets):
            colors = ast.literal_eval(node.value)
            assert set(colors) == set(CONDITIONS)
            return colors
    raise RuntimeError('No authoritative Figure 4e condition palette found')


def points(ax, table, value, colors, center=0., width=.36):
    rng = np.random.default_rng(1)
    for i, condition in enumerate(CONDITIONS):
        sub = table.loc[table.condition.eq(condition)]
        y = sub[value].to_numpy()
        x = center + (i - .5) * width
        jitter = rng.uniform(-.06, .06, len(y)) * width / .32
        ax.scatter(x + jitter, y, s=11.5, marker='o', color=colors[condition],
                   edgecolor='white', linewidth=.35, zorder=4)
        mean = np.mean(y)
        sem = np.std(y, ddof=1) / np.sqrt(len(y))
        ax.hlines(mean, x-width*.27, x+width*.27, color='#222222', linewidth=1.0, zorder=5)
        ax.vlines(x, mean-sem, mean+sem, color='#222222', linewidth=.65, zorder=5)


def bracket(ax, center, width, y, h, text):
    ax.plot([center-width/2, center-width/2, center+width/2, center+width/2],
            [y, y+h, y+h, y], color='#666666', linewidth=.55)
    ax.annotate(text, (center, y+h), xytext=(0, 2.3), textcoords='offset points',
                ha='center', va='bottom', fontsize=6.1, color='#333333')


def significance_label(value, kind):
    """Use the same P/q for both the stars and significance cutoff."""
    if not np.isfinite(value):
        raise ValueError('Missing significance value')
    for cutoff, stars in [(.0001, '****'), (.001, '***'), (.01, '**'), (.05, '*')]:
        if value < cutoff:
            number = f'{kind} < 0.001' if value < .001 else f'{kind} = {value:.3f}'
            return f'{stars}  {number}'
    return None


def draw(metrics, gates, stats, colors):
    setup()
    plt.rcParams.update({'font.size': 6.7, 'axes.labelsize': 6.7, 'axes.titlesize': 7.6,
                         'xtick.labelsize': 6.2, 'ytick.labelsize': 6.2, 'axes.linewidth': .65,
                         'text.color': '#222222', 'axes.labelcolor': '#222222',
                         'axes.edgecolor': '#222222', 'xtick.color': '#222222', 'ytick.color': '#222222',
                         'xtick.major.size': 2.2, 'ytick.major.size': 2.2})
    # 360 × 190 pt: approximately the existing Figure 4e chart footprint.
    fig = plt.figure(figsize=(5, 190/72))
    left = fig.add_axes([.128, .25, .18, .56])
    right = fig.add_axes([.474, .25, .503, .56])
    left.set(yscale='log', xlim=(-.8, .8), ylim=(40, 1500),
             xticks=[-.425, .425], xticklabels=CONDITIONS,
             ylabel='P14 per 10,000\nnon-P14 cells')
    left.yaxis.set_major_locator(FixedLocator([50, 100, 250, 500, 1000]))
    left.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    left.yaxis.set_minor_formatter(NullFormatter())
    left.tick_params(axis='y', which='minor', length=0)
    points(left, metrics, 'p14_per10k', colors, width=.85)
    abundance_test = stats.loc[stats.metric.eq('log2_p14_per10k')].iloc[0]
    annotation = significance_label(abundance_test.p_ols_batch, 'P')
    if annotation is not None:
        bracket(left, 0, .85, 810, 60, annotation)
    left.set_title('P14 abundance', loc='left', pad=13, fontweight='bold')

    right.set(xlim=(-.55, 4.55), ylim=(-2.4, 2.5), xticks=range(5),
              xticklabels=['Top IE', 'Top LP', 'Crypt IE', 'Crypt LP', 'Muscularis'],
              yticks=[-2, -1, 0, 1, 2], ylabel='P14 enrichment\nlog$_2$(P14 share / non-P14 share)')
    right.axhline(0, color='#A3A3A3', linewidth=.65, linestyle=(0, (3, 2)), zorder=0)
    for i, gate in enumerate(GATES):
        sub = gates.loc[gates.imap_gate.eq(gate)]
        points(right, sub, 'log2_enrich', colors, center=i)
        test = stats.loc[stats.metric.eq('log2_enrich') & stats.gate.eq(gate)].iloc[0]
        annotation = significance_label(test.q_ols_batch, 'q')
        if annotation is not None:
            bracket(right, i, .36, sub.log2_enrich.max()+.22, .08, annotation)
    right.set_title('P14 enrichment by compartment', loc='left', pad=13, fontweight='bold')
    # Temporary a/b identifiers aid review; final integration will use the page's e.
    fig.text(.025, .88, 'a', fontsize=9, weight='bold', va='bottom')
    fig.text(.37, .88, 'b', fontsize=9, weight='bold', va='bottom')
    handles = [Line2D([], [], ls='', marker='o', color=colors[c], markersize=3.5,
                      label=f'{c} (n = {n})') for c, n in [('DMSO', 4), ('RARi', 5)]]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.52, .10), ncol=2, frameon=False,
               fontsize=6.1, handletextpad=.35, handlelength=.8, columnspacing=1.15)

    # Directly check every dot against the frozen selected endpoints.
    for axis, expected in [(left, metrics.p14_per10k), (right, gates.log2_enrich)]:
        plotted = np.concatenate([c.get_offsets()[:, 1] for c in axis.collections if isinstance(c, PathCollection)])
        np.testing.assert_array_equal(np.sort(plotted), np.sort(expected))
        assert axis.get_ylim()[0] < plotted.min() <= plotted.max() < axis.get_ylim()[1]
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--style-provenance', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    release = json.loads(verified({'path': str(a.manifest), 'sha256': a.manifest_sha256}).read_text())
    assert release['status'] == 'frozen_user_rari_positioning'
    files = {Path(r['path']).name: r for r in release['outputs']}
    needed = ['per_mouse_metrics.csv', 'per_mouse_gate_counts.csv', 'statistics.csv']
    metrics, gates, stats = [pd.read_csv(verified(files[n]), float_precision='round_trip') for n in needed]
    stats = stats.fillna({'gate': ''})
    style = json.loads(a.style_provenance.read_text())
    palette_record = next(r for r in style['inputs'] if r['path'].endswith('/replicate_revision_20261014/final_panels.py'))
    colors = treatment_palette(verified(palette_record))
    current = Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Regenerated_Figures/Current')
    before = {p.name: record(p)['sha256'] for p in current.glob('*.pdf')}
    assert len(before) == 8
    a.output.mkdir(parents=True)
    fig = draw(metrics, gates, stats, colors)
    outputs = []
    for ext in ['pdf', 'png']:
        target = a.output / f'Figure4e_candidate_AB.{ext}'
        fig.savefig(target, dpi=600)
        outputs.append(record(target))
    plt.close(fig)
    text = '''# Figure 4e candidate: abundance and compartment enrichment

Only panels A and B from the uploaded P14-positioning analysis are retained.
The current Figure 4e treatment palette is used exactly: DMSO #4F7F8A and RARi #A55D54.
The 360 × 190 pt layout is sized approximately for the current Figure 4e chart area.
Each circle represents one biological replicate. Both plots retain all individual
mice and mean ± SEM; the legend identifies treatment only.

Left: whole-section P14 per 10,000 non-P14 cells, on a logarithmic axis.
Its P value is from experiment-adjusted OLS on log2 abundance.
Right: per-mouse gate enrichment, using the supplied script's 0.5 P14-count
pseudocount. q values are BH-adjusted across the five gate comparisons.
The exploratory analysis, cells, and statistics are unchanged from the selected
panels. These tests differ from those in the current manuscript Figure 4e.
The experiment covariate remains in the tests; only its visual encoding was removed.
Only comparisons with P < 0.05 (left) or q < 0.05 (right) have brackets and labels.
Asterisks use that same P/q value: * < 0.05, ** < 0.01, *** < 0.001, **** < 0.0001.
Nonsignificant comparisons retain all points but have no significance annotation.

This is a review candidate. Current Figure 4, panel exports, Methods, and legends
remain unchanged. The a/b labels are review identifiers; final integration will
use the main figure's e label. No final integration has been performed.

The PDF is vector artwork with embedded Arial; the companion PNG is 600 dpi.
The provenance file records exact inputs, code, palette source, and regeneration command.
'''
    (a.output / 'README.md').write_text(text)
    outputs.append(record(a.output / 'README.md'))
    assert before == {p.name: record(p)['sha256'] for p in current.glob('*.pdf')}
    provenance = {
        'status': 'candidate_awaiting_author_review', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, *sys.argv], 'python': platform.python_version(),
        'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'matplotlib']},
        'git_state': 'Not a Git worktree', 'random_seed': 1, 'randomness': 'Deterministic point jitter only',
        'inputs': [record(a.manifest), *[files[n] for n in needed], record(a.style_provenance), palette_record,
                   record(__file__), record(Path(__file__).with_name('render_user_rari_positioning.py')),
                   record(ROOT / 'frozen_preprocessing/if_replicates_20261014/export_cell_plotting_data.py')],
        'outputs': outputs, 'condition_palette': colors, 'canvas_pt': [360, 190],
        'plotted_mouse_values': {'A': 9, 'B': 45, 'exact_frozen_value_match': True},
        'replicate_marker': 'Circle for every mouse; treatment-only legend',
        'significance_display': {
            'cutoff': .05, 'A_value': 'p_ols_batch', 'B_value': 'q_ols_batch',
            'stars': {'*': '<0.05', '**': '<0.01', '***': '<0.001', '****': '<0.0001'},
            'nonsignificant_brackets_and_labels': False,
            'experiment_adjustment_retained': True,
        },
        'current_figure_hashes_unchanged': before, 'paper_figures_modified': False,
        'statistics_recomputed': False, 'warnings': release['warnings'],
    }
    (a.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'palette': colors, 'outputs': outputs}, indent=2))


if __name__ == '__main__':
    main()
