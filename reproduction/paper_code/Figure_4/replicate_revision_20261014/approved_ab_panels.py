"""Final Figure 4E/G: render the author-approved frozen A/B analysis."""
from __future__ import annotations
import argparse
import json
from paper_paths import Path, lock_record, rendering_script
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

from Figure_4.assembly.common import HOME, PAGE, PageInputs, sha256
from Figure_4.assembly import design

HERE = Path(__file__).resolve().parent
POINTER = HERE / 'approved_ab_release.json'


def points(axis, table, value, colors, center=0., width=.36):
    """Approved deterministic replicate display, kept local for portable rendering."""
    rng = np.random.default_rng(1)
    for i, condition in enumerate(['DMSO', 'RARi']):
        sub = table.loc[table.condition.eq(condition)]
        y = sub[value].to_numpy()
        x = center + (i - .5) * width
        jitter = rng.uniform(-.06, .06, len(y)) * width / .32
        axis.scatter(x + jitter, y, s=11.5, marker='o', color=colors[condition],
                     edgecolor='white', linewidth=.35, zorder=4)
        mean = np.mean(y)
        sem = np.std(y, ddof=1) / np.sqrt(len(y))
        axis.hlines(mean, x-width*.27, x+width*.27, color='#222222', linewidth=1.0, zorder=5)
        axis.vlines(x, mean-sem, mean+sem, color='#222222', linewidth=.65, zorder=5)


def bracket(axis, center, width, y, height, label):
    axis.plot([center-width/2, center-width/2, center+width/2, center+width/2],
              [y, y+height, y+height, y], color='#666666', linewidth=.55)
    axis.annotate(label, (center, y+height), xytext=(0, 2.3), textcoords='offset points',
                  ha='center', va='bottom', fontsize=6.1, color='#333333')


def significance_label(value, kind):
    if not np.isfinite(value):
        raise ValueError('Missing significance value')
    for cutoff, stars in [(.0001, '****'), (.001, '***'), (.01, '**'), (.05, '*')]:
        if value < cutoff:
            number = f'{kind} < 0.001' if value < .001 else f'{kind} = {value:.3f}'
            return f'{stars}  {number}'
    return None


def load(page, manifest=None, expected=None):
    pointer = json.loads(page.read(POINTER, role='approved_figure4_EF_release').read_text())
    assert pointer['status'] == 'author_approved_for_main_figure4'
    rec = pointer['analysis_manifest']
    if manifest is not None:
        assert Path(manifest).resolve() == Path(rec['path']).resolve() and expected == rec['sha256']
    release = json.loads(page.read(rec['path'], rec['sha256'], 'frozen_approved_EF_analysis').read_text())
    assert release['status'] == 'frozen_user_rari_positioning'
    candidate = pointer['approved_candidate_provenance']
    approved = json.loads(page.read(candidate['path'], candidate['sha256'], 'author_selected_candidate').read_text())
    # Reuse the approved dot/bracket functions without changing their code.
    for name in ['render_figure4e_ab_candidate.py', 'render_user_rari_positioning.py']:
        source = next(r for r in approved['inputs'] if Path(r['path']).name == name)
        page.read(source['path'], source['sha256'], 'approved_plot_helper')
    files = {Path(r['path']).name: r for r in release['outputs']}
    def table(name):
        r = files[name]
        return pd.read_csv(page.read(r['path'], r['sha256'], name), float_precision='round_trip')
    data = {'metrics': table('per_mouse_metrics.csv'), 'gates': table('per_mouse_gate_counts.csv'),
            'stats': table('statistics.csv').fillna({'gate': ''}), 'colors': pointer['condition_palette']}
    assert len(data['metrics']) == 9 and len(data['gates']) == 45
    assert data['metrics'].condition.value_counts().to_dict() == {'RARi': 5, 'DMSO': 4}
    assert data['metrics'].n_cells.sum() == 1568177 and data['metrics'].n_p14.sum() == 43424
    page.read(Path(__file__), role='approved_EF_final_renderer')
    page.warnings.extend(release['warnings'])
    page.details['approved_EF'] = {'analysis_manifest': rec, 'candidate': candidate,
        'all_mouse_values': data['metrics'].to_dict('records'),
        'all_gate_values': data['gates'].to_dict('records'),
        'statistics': data['stats'].loc[data['stats'].metric.isin(['log2_p14_per10k', 'log2_enrich'])].to_dict('records'),
        'palette': data['colors'], 'summary': 'Arithmetic mouse mean ± SEM',
        'points': 'One circle per biological replicate; four DMSO and five RARi mice',
        'P': 'Two-sided condition coefficient in OLS on log2 whole-section abundance, adjusted for experiment',
        'q': 'BH correction of the five experiment-adjusted gate OLS P values',
        'significance_display': 'Only P/q < 0.05; stars use the same P/q value',
        'random_seed': 1, 'randomness': 'Approved plotting jitter only'}
    return data


def style(axis):
    axis.tick_params(labelsize=7, width=.65, length=2.2, pad=2.5)
    axis.spines[['top', 'right']].set_visible(False)
    for spine in axis.spines.values():
        spine.set_linewidth(.65)


def enlarge_replicate_dots(axis):
    """Resize the approved dots without altering their values or jitter."""
    for artist in axis.collections:
        if isinstance(artist, PathCollection):
            artist.set_sizes([design.REPLICATE_MARKER_AREA])


def abundance(page, data):
    figure = page.figure()
    axis = page.axis(figure, 'E_abundance', design.ABUNDANCE_RECT)
    axis.set(yscale='log', xlim=(-.8, .8), ylim=(40, 1500),
             xticks=[-.425, .425], xticklabels=['DMSO', 'RARi'])
    axis.set_ylabel('P14 per 10,000\nnon-P14 cells', fontsize=7.5)
    axis.yaxis.set_major_locator(FixedLocator([50, 100, 250, 500, 1000]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(axis='y', which='minor', length=0)
    style(axis)
    points(axis, data['metrics'], 'p14_per10k', data['colors'], width=.85)
    test = data['stats'].loc[data['stats'].metric.eq('log2_p14_per10k')].iloc[0]
    text = significance_label(test.p_ols_batch, 'P')
    if text:
        bracket(axis, 0, .85, 810, 60, text)
    for label in axis.texts:
        label.set_fontsize(7)
    enlarge_replicate_dots(axis)
    page.save(figure, 'panel_E_P14_abundance')


def enrichment(page, data):
    figure = page.figure()
    axis = page.axis(figure, 'G_enrichment', design.ENRICHMENT_RECT)
    gates = ['Top IE', 'Top LP', 'Crypt IE', 'Crypt LP', 'Muscularis']
    axis.set(xlim=(-.55, 4.55), ylim=(-2.4, 2.5), xticks=range(5),
             xticklabels=[g.replace(' ', '\n') for g in gates],
             yticks=[-2, -1, 0, 1, 2])
    axis.set_ylabel('P14 enrichment\nlog$_2$(P14 share / non-P14 share)', fontsize=7.5)
    style(axis)
    axis.axhline(0, color='#A3A3A3', linewidth=.65, linestyle=(0, (3, 2)), zorder=0)
    for i, gate in enumerate(gates):
        sub = data['gates'].loc[data['gates'].imap_gate.eq(gate)]
        points(axis, sub, 'log2_enrich', data['colors'], center=i)
        test = data['stats'].loc[data['stats'].metric.eq('log2_enrich') & data['stats'].gate.eq(gate)].iloc[0]
        text = significance_label(test.q_ols_batch, 'q')
        if text:
            bracket(axis, i, .36, sub.log2_enrich.max()+.22, .08, text)
    handles = [Line2D([], [], ls='', marker='o', color=data['colors'][c], markersize=np.sqrt(design.REPLICATE_MARKER_AREA),
                      label=c) for c in ['DMSO', 'RARi']]
    figure.legend(handles=handles, loc='center', bbox_to_anchor=(design.CONDITION_LEGEND[0]/PAGE[0], 1-design.CONDITION_LEGEND[1]/PAGE[1]),
                  ncol=2, frameon=False, fontsize=7, handletextpad=.35, handlelength=.8, columnspacing=1.15)
    for label in axis.texts:
        label.set_fontsize(7)
    enlarge_replicate_dots(axis)
    page.save(figure, 'panel_G_P14_enrichment')


def component_main(panel):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--analysis-manifest', type=Path, required=True)
    parser.add_argument('--analysis-sha256', required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    stem = {'E': 'panel_E_P14_abundance', 'G': 'panel_G_P14_enrichment'}[panel]
    for name in (stem+'.pdf', stem+'.png', 'assembly_inputs_provenance.json'):
        if (args.output_dir/name).exists() and not args.overwrite:
            raise FileExistsError(args.output_dir/name)
    page = PageInputs(args.output_dir, args.overwrite)
    data = load(page, args.analysis_manifest, args.analysis_sha256)
    {'E': abundance, 'G': enrichment}[panel](page, data)
    for output in list(page.outputs):
        pdf = Path(output['path'])
        subprocess.run(['pdftoppm', '-singlefile', '-r', '600', '-png', str(pdf), str(pdf.with_suffix(''))], check=True)
        png = pdf.with_suffix('.png')
        page.outputs.append({'path': str(png), 'sha256': sha256(png)})
    page.read(Path(__file__), role='approved_EF_renderer')
    page.read(Path(design.__file__), role='author_approved_display_geometry')
    page.finish(HOME/'Figure_4/assembly/layout.json', Path(sys.argv[0]).resolve(),
                command=[sys.executable, *sys.argv], random_seed=1)
