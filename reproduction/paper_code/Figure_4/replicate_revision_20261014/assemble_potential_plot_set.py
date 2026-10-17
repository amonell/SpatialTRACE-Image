#!/usr/bin/env python3
"""Collect reviewed current IF PDFs and compile a bookmarked candidate book."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from figure_assembly.compositor import file_sha256
from figure_assembly.qa import font_check


def record(path):
    return {'path': str(path), 'sha256': file_sha256(path)}


def verified(rec):
    if file_sha256(rec['path']) != rec['sha256']:
        raise ValueError(f"Hash mismatch: {rec['path']}")
    return Path(rec['path'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-set', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--removal-review', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    publication = args.plot_set/'plot_set_manifest.json'
    book = args.plot_set/'All_IF_candidate_plots.pdf'
    if publication.exists() or book.exists():
        raise FileExistsError(publication)
    rendering = json.loads((args.plot_set/'rendering_manifest.json').read_text())
    candidate = json.loads((args.candidate/'provenance.json').read_text())
    review = json.loads((args.removal_review/'provenance.json').read_text())
    for manifest in [rendering, candidate, review]:
        for rec in manifest['inputs'] + manifest['outputs']:
            verified(rec)
    current = args.data_root/'Regenerated_Figures/Current'
    before = {p.name: file_sha256(p) for p in current.iterdir()}
    # Copies are review assets, never a substitute for regeneration.
    copies = []
    for package, name in [('Figure_4', 'Figure4.pdf'), ('Figure_4_Extended', 'Figure4_extended.pdf')]:
        index = json.loads((args.data_root/package/'final_panel_manifest.json').read_text())
        source = verified({'path': index['complete_page'], 'sha256': index['complete_page_sha256']})
        assert source == current/name
        copies.append((source, args.plot_set/'pdfs'/name))
    for rec in candidate['outputs']:
        if rec['path'].endswith('.pdf'):
            source = verified(rec)
            copies.append((source, args.plot_set/'pdfs'/source.name))
    review_pdf = args.removal_review/'IF_removals_and_density_over_expected.pdf'
    copies.append((review_pdf, args.plot_set/'pdfs/Additional_07_removals_and_enrichment_review.pdf'))
    if any(destination.exists() for _, destination in copies):
        raise FileExistsError('Refusing to replace existing review copies')
    sources = [source for source, _ in copies] + sorted((args.plot_set/'pdfs').glob('*.pdf'))
    qa = []
    for source in sources:
        fonts = font_check(source)
        if fonts['errors']:
            raise RuntimeError(fonts['errors'])
        document = ET.fromstring(subprocess.check_output(['pdftotext', '-bbox', str(source), '-']))
        outside = []
        for page in document.iter():
            if page.tag.rsplit('}', 1)[-1] != 'page':
                continue
            width, height = float(page.attrib['width']), float(page.attrib['height'])
            for word in page.iter():
                if word.tag.rsplit('}', 1)[-1] == 'word':
                    x0, y0, x1, y1 = [float(word.attrib[k]) for k in ['xMin', 'yMin', 'xMax', 'yMax']]
                    if x0 < -.5 or y0 < -.5 or x1 > width+.5 or y1 > height+.5:
                        outside.append(word.text)
        if outside:
            raise RuntimeError(f'Text outside page in {source}: {outside}')
        qa.append({'source': record(source), 'embedded_Arial': True, 'text_outside_page': outside})
    copied = []
    for source, destination in copies:
        shutil.copy2(source, destination)
        assert file_sha256(source) == file_sha256(destination)
        copied.append({'source': record(source), 'copy': record(destination)})
    names = ['Additional_08_P14_composition_bars.pdf', 'Additional_09_P14_composition_statistics.pdf',
             'Figure4.pdf', 'Figure4_extended.pdf',
             'Additional_01_regional_consistency.pdf', 'Additional_02_individual_P14_IMAPs.pdf',
             'Additional_03_regional_composition.pdf', 'Additional_04_new_section_maps.pdf',
             'Additional_05_new_gate_backprojections.pdf', 'Additional_06_individual_non_P14_IMAPs.pdf',
             'Additional_07_removals_and_enrichment_review.pdf']
    assert {p.name for p in (args.plot_set/'pdfs').iterdir()} == set(names)
    writer = PdfWriter()
    order = []
    for name in names:
        source = args.plot_set/'pdfs'/name
        first = len(writer.pages) + 1
        reader = PdfReader(source)
        writer.append(reader, outline_item=name.removesuffix('.pdf').replace('_', ' '))
        order.append({'file': name, 'first_page': first, 'pages': len(reader.pages), 'source': record(source)})
    writer.add_metadata({'/Title': 'IF candidate plots after tissue removals', '/Author': 'SpatialTRACE'})
    with book.open('xb') as handle:
        writer.write(handle)
    after = {p.name: file_sha256(p) for p in current.iterdir()}
    assert before == after, 'Paper figures changed during review compilation'
    manifest = {'status': 'COMPILED_PENDING_FINAL_VISUAL_QA', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'command': [sys.executable, *sys.argv], 'code': record(Path(__file__)),
                'rendering_manifest': record(args.plot_set/'rendering_manifest.json'),
                'candidate_provenance': record(args.candidate/'provenance.json'),
                'removal_review_provenance': record(args.removal_review/'provenance.json'),
                'copies': copied, 'pdf_checks': qa, 'page_order': order, 'combined_pdf': record(book),
                'current_figures_unchanged': before, 'description': 'Post-removal candidates; new percentage comparison first. Current paper PDFs are copies.'}
    publication.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'pdf_folder': str(args.plot_set/'pdfs'), 'combined_pdf': str(book), 'pages': len(writer.pages)}, indent=2))


if __name__ == '__main__':
    main()
