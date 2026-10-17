#!/usr/bin/env python3
"""Polish the uploaded P14 figures from frozen tables; no fitting or inference."""
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'frozen_preprocessing/if_replicates_20261014'))
from export_cell_plotting_data import record, verified

CONDITIONS = ['DMSO', 'RARi']
COLORS = {'DMSO': '#37788F', 'RARi': '#B05B49'}
GATES = ['Top IE', 'Top LP', 'Crypt IE', 'Crypt LP', 'Muscularis']
INK = '#28333B'
MUTED = '#637078'
ENRICHMENT = LinearSegmentedColormap.from_list('enrichment',
    ['#234E70', '#78A9BD', '#F7F6F2', '#D9A77E', '#A74638'])
DIFFERENCE = LinearSegmentedColormap.from_list('difference',
    ['#276B82', '#91B8C2', '#F7F6F2', '#D9A17E', '#AA513D'])


def setup():
    for weight in ['normal', 'bold']:
        findfont(FontProperties(family='Arial', weight=weight), fallback_to_default=False)
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 9, 'axes.labelsize': 9,
        'axes.titlesize': 11, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK,
        'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': .75,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': .7, 'ytick.major.width': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
        'figure.facecolor': 'white', 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Arial', 'mathtext.it': 'Arial:italic',
        'mathtext.bf': 'Arial:bold', 'mathtext.sf': 'Arial',
        'mathtext.default': 'regular', 'mathtext.fallback': None,
    })


def legend(fig, y):
    handles = [Patch(facecolor=COLORS[c], edgecolor='none', label=f'{c} (n = {n})')
               for c, n in [('DMSO', 4), ('RARi', 5)]]
    handles += [Line2D([], [], linestyle='', marker=marker, color=MUTED,
                       markersize=5, label=label)
                for marker, label in [('o', 'New experiment'), ('D', 'Original experiment')]]
    fig.legend(handles=handles, loc='center', bbox_to_anchor=(.52, y), ncol=4,
               frameon=False, fontsize=8.5, handlelength=1.1, handletextpad=.5, columnspacing=1.8)


def dots(ax, table, column, center=0., width=.4, size=30):
    rng = np.random.default_rng(1)
    positions = {}
    for i, condition in enumerate(CONDITIONS):
        selected = table.loc[table.condition.eq(condition)]
        values = selected[column].to_numpy(float)
        x = center + (i - .5) * width
        positions[condition] = x
        jitter = rng.uniform(-.06, .06, len(values)) * width / .32
        for original, marker in [(False, 'o'), (True, 'D')]:
            mask = selected.experiment.eq('original').to_numpy() == original
            ax.scatter(x + jitter[mask], values[mask], s=size, marker=marker,
                       color=COLORS[condition], edgecolor='white', linewidth=.65, zorder=4)
        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values))
        ax.hlines(mean, x - width * .28, x + width * .28, color=INK, linewidth=1.5, zorder=5)
        ax.vlines(x, mean - sem, mean + sem, color=INK, linewidth=1., zorder=5)
    return positions


def bracket(ax, x1, x2, y, text, h=.06):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color=MUTED, linewidth=.75, clip_on=False)
    ax.annotate(text, ((x1+x2)/2, y+h), xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom', fontsize=8, color=INK)


def label(fig, x, y, letter):
    fig.text(x, y, letter, fontsize=13, fontweight='bold', ha='left', va='bottom')


def probability(value, letter='P'):
    return f'{letter} < 0.001' if value < .001 else f'{letter} = {value:.3f}'


def positioning(metrics, gates, profiles, results):
    fig = plt.figure(figsize=(10.6, 7.3))
    fig.text(.52, .982, 'P14 abundance and anatomical positioning', fontsize=15,
             fontweight='bold', ha='center', va='top')
    legend(fig, .928)
    stats = results.set_index(['metric', 'gate'])
    ax = fig.add_axes([.085, .55, .18, .285])
    positions = dots(ax, metrics, 'p14_per10k', width=.85, size=34)
    ax.set(yscale='log', ylim=(40, 1500), xlim=(-.8, .8),
           xticks=list(positions.values()), xticklabels=CONDITIONS,
           ylabel='P14 per 10,000\nnon-P14 cells')
    ax.yaxis.set_major_locator(FixedLocator([50, 100, 250, 500, 1000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='y', which='minor', length=0)
    ax.set_title('P14 abundance', pad=25, loc='left')
    result = stats.loc[('log2_p14_per10k', '')]
    bracket(ax, positions['DMSO'], positions['RARi'], 810, probability(result.p_ols_batch), h=60)
    fig.text(.085, .847, f'RARi / DMSO: {2**result.diff_RARi_minus_DMSO:.2f}×',
             fontsize=8.5, color=MUTED)
    label(fig, .025, .872, 'a')

    ax = fig.add_axes([.365, .55, .605, .285])
    ax.axhline(0, color='#A0AAAF', linewidth=.8, linestyle=(0, (3, 3)), zorder=0)
    for i, gate in enumerate(GATES):
        selected = gates.loc[gates.imap_gate.eq(gate)]
        positions = dots(ax, selected, 'log2_enrich', center=i, width=.36, size=28)
        result = stats.loc[('log2_enrich', gate)]
        bracket(ax, positions['DMSO'], positions['RARi'], selected.log2_enrich.max()+.22,
                probability(result.q_ols_batch, 'q'), h=.08)
    ax.set(xlim=(-.55, 4.55), ylim=(-2.4, 2.5), xticks=range(5), xticklabels=GATES,
           ylabel='P14 enrichment\nlog₂(P14 share / non-P14 share)', yticks=[-2, -1, 0, 1, 2])
    ax.set_title('P14 enrichment by compartment', pad=25, loc='left')
    fig.text(.365, .847, 'Normalized to the non-P14 distribution in each section', fontsize=8.5, color=MUTED)
    label(fig, .315, .872, 'b')

    for j, comp in enumerate(['LP', 'IE']):
        left = [.085, .405][j]
        ax = fig.add_axes([left, .16, .225, .265])
        selected = profiles.loc[profiles.compartment.eq(comp)]
        ax.axvline(0, color='#A0AAAF', linewidth=.8, linestyle=(0, (3, 3)), zorder=0)
        for boundary in ([.25, .4] if comp == 'LP' else [.4]):
            ax.axhline(boundary, color='#D1D7DA', linewidth=.75, zorder=0)
        for condition in CONDITIONS:
            subset = selected.loc[selected.condition.eq(condition)]
            for _, mouse in subset.groupby('mouse_id'):
                ax.plot(mouse.log2_enrich, mouse.cv, color=COLORS[condition], linewidth=.65, alpha=.27)
            summary = subset.groupby('cv').log2_enrich.agg(['mean', 'sem'])
            ax.fill_betweenx(summary.index, summary['mean']-summary['sem'], summary['mean']+summary['sem'],
                             color=COLORS[condition], alpha=.15, linewidth=0)
            ax.plot(summary['mean'], summary.index, color=COLORS[condition], linewidth=2.2)
        ax.set(xlim=(-3.8, 1.4), ylim=(0, 1), xticks=[-3, -2, -1, 0, 1], yticks=[0, .25, .5, .75, 1],
               xlabel='P14 enrichment, log₂', ylabel='Crypt–villus position')
        ax.set_title('LP + muscularis' if comp == 'LP' else 'Intraepithelial', loc='left', pad=25)
        fig.text(left, .437, 'Epithelial distance > 0.28' if comp == 'LP' else 'Epithelial distance ≤ 0.28',
                 fontsize=8.5, color=MUTED)
        label(fig, left-.06, .462, 'cd'[j])

    ax = fig.add_axes([.755, .16, .215, .265])
    ax.axhline(.5, color='#A0AAAF', linewidth=.8, linestyle=(0, (3, 3)), zorder=0)
    for i, name in enumerate(['auc_cv_LP', 'auc_cv_IE']):
        positions = dots(ax, metrics, name, center=i, width=.42, size=28)
        result = stats.loc[(name, '')]
        bracket(ax, positions['DMSO'], positions['RARi'], metrics[name].max()+.014,
                probability(result.p_ols_batch), h=.006)
    ax.set(xlim=(-.5, 1.5), ylim=(.34, .735), xticks=[0, 1], xticklabels=['LP + muscularis', 'IE'],
           yticks=[.4, .5, .6, .7], ylabel='CV-position AUC\nP14 relative to non-P14')
    ax.tick_params(axis='x', labelsize=7.5)
    ax.set_title('Villus-ward bias', loc='left', pad=25)
    fig.text(.755, .437, '0.5 = no positional bias', fontsize=8.5, color=MUTED)
    label(fig, .695, .462, 'e')
    fig.text(.085, .065, 'Points: individual mice; summaries: mean ± SEM. Profiles: individual mice with mean ± SEM.',
             color=MUTED, fontsize=8)
    fig.text(.085, .037, 'P: experiment-adjusted OLS. q: BH-adjusted across five gates. All tissue removals applied.',
             color=MUTED, fontsize=8)
    return fig


def gate_overlay(ax):
    style = {'color': '#46525A', 'linewidth': .8, 'linestyle': (0, (4, 3)), 'alpha': .85}
    ax.axvline(.28, **style)
    ax.axhline(.4, **style)
    ax.plot([.28, .8], [.25, .25], **style)
    for x, y, name in [(.14, .85, 'Top IE'), (.56, .85, 'Top LP'),
                        (.14, .2, 'Crypt IE'), (.58, .325, 'Crypt LP'), (.66, .125, 'Musc')]:
        ax.text(x, y, name, ha='center', va='center', fontsize=8, color=INK,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .88, 'pad': 1.8})


def imaps(grids, gates, results):
    fig = plt.figure(figsize=(12, 5.15))
    fig.text(.52, .984, 'P14 enrichment across anatomical coordinates', ha='center', va='top',
             fontsize=15, fontweight='bold')
    legend(fig, .917)
    axes = [fig.add_axes([left, .29, .18, .49]) for left in [.065, .29, .515]]
    meshes = []
    for i, (ax, data, title, cmap, limit) in enumerate(zip(
            axes, [*grids['group_means'], grids['difference']],
            ['DMSO', 'RARi', 'RARi − DMSO'], [ENRICHMENT, ENRICHMENT, DIFFERENCE], [2.5, 2.5, 2.])):
        cmap = cmap.copy()
        cmap.set_bad('#FFFFFF')
        mesh = ax.pcolormesh(grids['epi_edges'], grids['cv_edges'], np.ma.masked_invalid(data),
                             shading='flat', cmap=cmap, norm=TwoSlopeNorm(0, -limit, limit),
                             linewidth=0, antialiased=False, rasterized=False)
        meshes.append(mesh)
        gate_overlay(ax)
        ax.set(xlim=(0, .8), ylim=(0, 1), xticks=[0, .4, .8], yticks=[0, .2, .4, .6, .8, 1],
               xlabel='Epithelial distance')
        if i == 0:
            ax.set_ylabel('Crypt–villus position')
        ax.set_title(title, fontsize=11, pad=12, color=COLORS[title] if title in COLORS else INK)
        label(fig, [.03, .255, .48][i], .812, 'abc'[i])
    cb = fig.colorbar(meshes[0], cax=fig.add_axes([.085, .145, .365, .024]),
                      orientation='horizontal', ticks=[-2.5, 0, 2.5], extend='both')
    cb.set_label('P14 enrichment: log₂(P14 density / non-P14 density)', fontsize=8.5, labelpad=4)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=2)
    cb = fig.colorbar(meshes[2], cax=fig.add_axes([.525, .145, .16, .024]),
                      orientation='horizontal', ticks=[-2, 0, 2], extend='both')
    cb.set_label('Difference in log₂ enrichment', fontsize=8.5, labelpad=4)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=2)

    ax = fig.add_axes([.775, .29, .205, .49])
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#EBEEF0', linewidth=.65)
    for i, gate in enumerate(GATES):
        dots(ax, gates.loc[gates.imap_gate.eq(gate)], 'bg_share', center=i, width=.42, size=24)
    ax.set(xlim=(-.6, 4.6), ylim=(0, 40), xticks=range(5), xticklabels=GATES,
           ylabel='Non-P14 cells in gate (%)', yticks=[0, 10, 20, 30, 40])
    plt.setp(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor', fontsize=8)
    ax.set_title('Background composition', fontsize=11, pad=12)
    q_min = results.loc[results.metric.eq('bg_share'), 'q_ols_batch'].min()
    ax.text(.98, .97, f'All gates q ≥ {q_min:.3f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color=MUTED)
    label(fig, .735, .812, 'd')
    fig.text(.065, .035,
             'Maps: mean per-mouse log₂ enrichment; sparse regions masked. Dashed lines: fixed gates. Difference map is descriptive.',
             fontsize=8, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    release = json.loads(verified({'path': str(args.manifest), 'sha256': args.manifest_sha256}).read_text())
    assert release['status'] == 'frozen_user_rari_positioning'
    files = {Path(r['path']).name: r for r in release['outputs']}
    needed = ['per_mouse_metrics.csv', 'per_mouse_gate_counts.csv', 'per_mouse_cv_profiles.csv', 'statistics.csv', 'imap_maps.npz']
    paths = {name: verified(files[name]) for name in needed}
    metrics = pd.read_csv(paths[needed[0]], float_precision='round_trip')
    gates = pd.read_csv(paths[needed[1]], float_precision='round_trip')
    profiles = pd.read_csv(paths[needed[2]], float_precision='round_trip')
    results = pd.read_csv(paths[needed[3]], float_precision='round_trip').fillna({'gate': ''})
    with np.load(paths['imap_maps.npz']) as src:
        grids = {k: src[k] for k in src.files}
    setup()
    args.output.mkdir(parents=True)
    outputs, axes_geometry = [], []
    for stem, fig in [
        ('Additional_14_P14_positioning_polished', positioning(metrics, gates, profiles, results)),
        ('Additional_15_IMAP_enrichment_polished', imaps(grids, gates, results)),
    ]:
        # This Arial build has no Unicode subscript-two glyph. Typeset the
        # subscript with Arial digits instead of silently using a fallback font.
        for text in fig.findobj(match=matplotlib.text.Text):
            if '₂' in text.get_text():
                text.set_text(text.get_text().replace('₂', r'$_2$'))
        fig.canvas.draw()
        axes_geometry.append({'figure': stem, 'axes': [
            {'position': list(ax.get_position().bounds), 'xlim': list(ax.get_xlim()),
             'ylim': list(ax.get_ylim()), 'xlabel': ax.get_xlabel(), 'ylabel': ax.get_ylabel()}
            for ax in fig.axes]})
        for extension in ['pdf', 'png']:
            path = args.output / f'{stem}.{extension}'
            fig.savefig(path, dpi=450)
            outputs.append(record(path))
        plt.close(fig)
    readme = '''# P14 positioning figures

Polished versions of the two figures in the supplied `analyze_rari_p14.py`.
The numerical analysis is unchanged. Both figures are supplied as vector PDFs
and 450-dpi PNGs. Arial is embedded in the PDFs.

The first figure shows P14 abundance, gate enrichment, CV profiles, and
background-relative CV-position AUC. The second shows the two treatment IMAPs,
their difference, and non-P14 gate composition.

Teal denotes DMSO and rust denotes RARi in point/profile plots. Circles show the
new experiment and diamonds the original experiment. Black marks show mean ± SEM.
In the heatmaps, blue means lower enrichment and rust means higher enrichment.
The two treatment maps share a scale. Colorbar extensions identify saturation;
white outside the tissue-shaped support marks unavailable values.

Statistics follow the supplied script: mouse-level OLS with an experiment
covariate. Panel 1b now prints BH-adjusted q values directly rather than raw
P values followed by significance stars. The five gate comparisons form one
correction family. The mouse-level P values in panels 1a/1e are unadjusted for
multiple testing. These exploratory tests differ from the earlier permutation/
Holm analyses and do not replace them.

The 0.28 epithelial threshold separates IE from LP + muscularis in profiles
and AUC summaries. AUC is P(random P14 CV coordinate > random non-P14 CV
coordinate), counting ties as one half. It is a positional summary, not
classifier performance. The abundance fold change is the unadjusted ratio of
group geometric means, as in the supplied script.

The original gate pseudocount (0.5), profile bins/masking, map smoothing,
0–0.8 map epithelial range, and color limits (±2.5 enrichment, ±2 difference)
are retained. Maps average available per-mouse values without experiment
adjustment; the number of contributing mice varies across bins. They show
relative distribution, not absolute cell counts or statistical significance.
A nonsignificant background-composition test is not evidence of equivalence.

The original uploaded script, its unchanged reference figures, numeric tables,
and map arrays are preserved in the frozen input release. Re-rendering reads
those tables and arrays; it does not fit models or recompute tests.
'''
    (args.output / 'README.md').write_text(readme, encoding='utf-8')
    outputs.append(record(args.output / 'README.md'))
    provenance = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
        'python': platform.python_version(),
        'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'matplotlib']},
        'git_state': 'Not a Git worktree', 'random_seed': 1, 'randomness': 'Plot jitter only',
        'inputs': [record(args.manifest), *[files[n] for n in needed], record(__file__),
                   record(ROOT / 'frozen_preprocessing/if_replicates_20261014/export_cell_plotting_data.py')],
        'outputs': outputs, 'palette': COLORS, 'axes_geometry': axes_geometry,
        'statistics_recomputed_during_render': False, 'warnings': release['warnings'],
        'display_changes': ['Arial typography', 'teal/rust treatment colors', 'shared treatment-map colorbar',
                            'direct BH q annotations', 'shared profile x limits', 'colorbar saturation extensions'],
        'paper_figures_modified': False,
    }
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'outputs': outputs}, indent=2))


if __name__ == '__main__':
    main()
