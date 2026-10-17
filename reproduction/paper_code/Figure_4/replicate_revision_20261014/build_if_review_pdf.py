#!/usr/bin/env python3
"""Bundle the current IF figures with an index and frozen P-value audit.

This is a review compilation, not panel regeneration. Source PDF pages are
embedded intact at their native dimensions, without rasterization or rescaling.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import sys

import pandas as pd
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph, Table, TableStyle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from figure_assembly.compositor import file_sha256, register_fonts

SOURCES = [
    ('Figure4.pdf', 'Main Figure 4', 'Current manuscript layout; nine-mouse summaries in E/F.'),
    ('Figure4_extended.pdf', 'Extended Figure 4', 'Original gate-overlay fields; nine-mouse non-P14 IMAPs.'),
    ('Additional_01_regional_consistency.pdf', 'Regional consistency', 'Individual mice: composition, abundance and regional enrichment.'),
    ('Additional_03_regional_composition.pdf', 'P14 and background composition', 'Compare the regional distributions of P14 and non-P14 cells.'),
    ('Additional_02_individual_P14_IMAPs.pdf', 'Individual P14 IMAPs', 'Inspect each mouse rather than only the treatment averages.'),
    ('Additional_06_individual_non_P14_IMAPs.pdf', 'Individual non-P14 IMAPs', 'Background coordinate distributions for each mouse.'),
    ('Additional_04_new_section_maps.pdf', 'Seven new tissue maps', 'Microscopy and both predicted coordinates across each section.'),
    ('Additional_05_new_gate_backprojections.pdf', 'Seven new anatomical overlays', 'Native DAPI/E-cadherin beside cells colored by the fixed gates.'),
    ('Additional_07_regional_statistics.pdf', 'Statistical summary', 'Original test convention, with new-only and cohort-stratified checks.')
]


def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def verify(rec):
    if file_sha256(rec['path']) != rec['sha256']:
        raise RuntimeError(f"Hash mismatch: {rec['path']}")
    return Path(rec['path'])


def paragraph(canvas, text, x, y, width, *, size=10, leading=14, bold=False, color='#222222'):
    style = ParagraphStyle('body', fontName='Arial-Bold' if bold else 'Arial', fontSize=size,
                           leading=leading, textColor=colors.HexColor(color))
    flow = Paragraph(text, style)
    _, height = flow.wrap(width, 1000)
    if y-height < 30:
        raise ValueError('Review text exceeds its page')
    flow.drawOn(canvas, x, y-height)
    return y-height


def make_front_matter(path, entries, audit):
    width, height = A4
    canvas = Canvas(str(path), pagesize=A4, initialFontName='Arial')
    canvas.setTitle('SpatialTRACE IF quantification — review guide and statistical audit')
    canvas.setAuthor('SpatialTRACE')
    x, body_width = 42, width-84
    y = paragraph(canvas, 'IF quantification: figure review', x, height-48, body_width,
                  size=21, leading=26, bold=True)
    y = paragraph(canvas, 'SpatialTRACE · 14 October 2026', x, y-12, body_width, size=11)
    y = paragraph(canvas, 'Nine mice: four DMSO and five RARi, including the original pair. '
                  'Tissue08 is excluded. These are the latest complete figures and all seven companion pages.',
                  x, y-23, body_width, size=10)
    y = paragraph(canvas, 'Read the P-value audit before interpreting panel E. The repeated 0.040 values reproduce '
                  'for the original convention, but the default SciPy two-sided convention gives 0.079.',
                  x, y-17, body_width, size=10, bold=True, color='#824534')
    y = paragraph(canvas, 'Contents', x, y-26, body_width, size=12, bold=True)
    contents = [(2, 'P-value audit', 'Independent count, gate, permutation and correction checks.'), *entries]
    for page, title, description in contents:
        y -= 12
        canvas.setFont('Arial-Bold', 10)
        canvas.drawString(x, y-9, f'{page:02d}')
        y = paragraph(canvas, title, x+30, y, body_width-30, size=10, leading=12, bold=True)
        y = paragraph(canvas, description, x+30, y-3, body_width-30, size=9, leading=11, color='#555555')
    paragraph(canvas, 'Use the PDF bookmarks to jump between figures. Original page sizes, labels and vector '
              'artwork are preserved. This review copy does not replace the manuscript figures.',
              x, y-22, body_width, size=9, leading=12)
    canvas.showPage()
    y = paragraph(canvas, 'Figure 4e: P-value audit', x, height-45, body_width, size=19, leading=23, bold=True)
    y = paragraph(canvas, 'The repeated P values are in the left subplot of Figure 4e. Main Figure 3e contains '
                  'coordinate maps and has no P values.', x, y-13, body_width)
    y = paragraph(canvas, 'An independent implementation reproduced the gate assignments, cell counts, '
                  'denominators and mouse identities, then enumerated every label allocation. '
                  'All 30 original tests and Holm corrections matched. No source figures were changed.',
                  x, y-13, body_width)
    selected = audit.loc[audit.analysis.eq('all_mice') & audit.metric.eq('p14_per_10000_non_p14')]
    headers = ['Gate', 'Absolute\nraw P', 'Absolute\nadjusted P', 'Doubled-tail\nraw P', 'Doubled-tail\nadjusted P']
    rows = [headers]
    for row in selected.itertuples():
        rows.append([row.gate, f'{row.absolute_raw_p:.5f}', f'{row.absolute_holm_p:.5f}',
                     f'{row.doubled_tail_raw_p:.5f}', f'{row.doubled_tail_holm_p:.5f}'])
    table = Table(rows, colWidths=[99, 93, 104, 102, body_width-398], rowHeights=[36]+[27]*5)
    table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Arial'), ('FONTSIZE', (0,0), (-1,-1), 9),
        ('FONTNAME', (0,0), (-1,0), 'Arial-Bold'), ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eaf0f2')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f6f6f6')]),
        ('GRID', (0,0), (-1,-1), .4, colors.HexColor('#d5d5d5')), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (-1,-1), 'CENTER')]))
    _, th = table.wrap(body_width, 500); y -= 18+th
    table.drawOn(canvas, x, y)
    y = paragraph(canvas, 'Both columns use the same four DMSO and five RARi mice, mean-difference statistic, '
                  '126 exact allocations and Holm correction across five gates.', x, y-12, body_width, size=9, leading=12)
    y = paragraph(canvas, 'Why all five values are equal', x, y-21, body_width, size=11, bold=True)
    y = paragraph(canvas, 'The first four raw absolute-tail P values are 1/126; muscularis is 4/126. '
                  'Holm’s step-down adjustment maps all five to 5/126 = 0.03968. This is not a copied label.',
                  x, y-7, body_width)
    y = paragraph(canvas, 'What changes the interpretation', x, y-18, body_width, size=11, bold=True)
    y = paragraph(canvas, 'The current test counts absolute differences at least as large as observed. '
                  'SciPy’s default two-sided test doubles the smaller signed-tail P value. These are different '
                  'conventions, not an arithmetic error; an asymmetric permutation distribution can give different answers. '
                  'Here the latter convention gives adjusted P = 0.07937 for every gate.', x, y-7, body_width)
    y = paragraph(canvas, 'The result therefore does not support a robust claim of P &lt; 0.05. '
                  'Treatment randomization is undocumented, the original mice contributed to model development, '
                  'and tissue08 was excluded after interim review. The cohort checks are retained in the final page. '
                  'Choose and state the inferential design before finalizing manuscript significance claims; '
                  'do not choose a convention because it gives a smaller P value.', x, y-13, body_width)
    y = paragraph(canvas, 'Definitions: <link href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html" color="#245a80">'
                  'SciPy permutation_test</link>; '
                  '<link href="https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html" color="#245a80">'
                  'statsmodels Holm correction</link>.', x, y-17, body_width, size=9, leading=12)
    canvas.save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--publication', type=Path, required=True)
    p.add_argument('--audit-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    publication = json.loads(a.publication.read_text())
    if publication['status'] != 'REVIEWED_AND_PUBLISHED':
        raise RuntimeError('Expected the visually reviewed current IF publication')
    copies = {Path(r['copy']['path']).name: r for r in publication['copies']}
    audit_manifest = json.loads(a.audit_manifest.read_text())
    if audit_manifest['status'] != 'frozen_independent_pvalue_audit':
        raise RuntimeError('Run the independent statistics audit before compilation')
    audit_files = {Path(r['path']).name: verify(r) for r in audit_manifest['outputs']}
    audit = pd.read_csv(audit_files['pvalue_audit.csv'])
    inputs, entries, pages, next_page = [], [], [], 3
    for name, title, description in SOURCES:
        item = copies[name]
        source = verify(item['copy'])
        verify(item['source'])
        reader = PdfReader(source)
        entries.append((next_page, title, description))
        pages.append({'path': str(source), 'title': title, 'first_page': next_page,
                      'page_count': len(reader.pages), 'source_sha256': item['copy']['sha256'],
                      'page_sizes_pt': [[float(pg.mediabox.width), float(pg.mediabox.height)] for pg in reader.pages]})
        next_page += len(reader.pages)
        inputs.append(record(source))
    a.output.mkdir(parents=True)
    fonts = register_fonts()
    front = a.output/'review_guide_and_pvalue_audit.pdf'
    make_front_matter(front, entries, audit)
    writer = PdfWriter()
    writer.append(front, import_outline=False)
    writer.add_outline_item('Contents', 0)
    writer.add_outline_item('P-value audit — read before interpreting panel E', 1)
    for entry in pages:
        writer.append(entry['path'], outline_item=f"{entry['first_page']:02d} · {entry['title']}", import_outline=False)
    writer.page_mode = '/UseOutlines'
    writer.add_metadata({'/Title': 'SpatialTRACE IF quantification — figure review',
                         '/Author': 'SpatialTRACE', '/Subject': 'Nine-mouse IF figure review with independent statistical audit'})
    target = a.output/'IF_quantification_review.pdf'
    with target.open('xb') as handle:
        writer.write(handle)
    reader = PdfReader(target)
    assert len(reader.pages) == next_page-1
    for entry in pages:
        for i, dims in enumerate(entry['page_sizes_pt']):
            pg = reader.pages[entry['first_page']-1+i]
            assert [float(pg.mediabox.width), float(pg.mediabox.height)] == dims
    manifest = {'status': 'COMPILED_REVIEW_PDF', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, *sys.argv], 'python': platform.python_version(),
        'packages': {p: importlib.metadata.version(p) for p in ['pypdf','reportlab','pandas']},
        'git_state': 'Not a Git worktree', 'random_seed': None, 'fonts': fonts,
        'inputs': [record(a.publication), record(a.audit_manifest), record(__file__), *inputs],
        'outputs': [record(front), record(target)], 'total_pages': len(reader.pages),
        'contents': [{'first_page': 1, 'title': 'Contents', 'page_count': 1},
                     {'first_page': 2, 'title': 'Independent P-value audit', 'page_count': 1}, *pages],
        'source_figures_modified': False, 'native_page_dimensions_preserved': True,
        'compiled_only_not_panel_regeneration': True,
        'warnings': ['Source figures retain original P values; their convention sensitivity is explained on page 2.']}
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output': str(target), 'pages': len(reader.pages), 'bytes': target.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
