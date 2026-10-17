#!/usr/bin/env python3
"""Copy a reviewed IF plot set and prepend a newly rendered candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from paper_paths import Path, lock_record, rendering_script
import platform
import shutil
import sys

import pymupdf
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'frozen_preprocessing/if_replicates_20261014'))
from audit_regional_statistics import record, verified
from figure_assembly.qa import font_check


def check_pdf(path):
    fonts = font_check(path)
    if fonts['errors']:
        raise RuntimeError(fonts['errors'])
    with pymupdf.open(path) as doc:
        for page in doc:
            for word in page.get_text('words'):
                rect = pymupdf.Rect(word[:4])
                if not (page.rect + (-.5, -.5, .5, .5)).contains(rect):
                    raise RuntimeError(f'Off-page text in {path}: {word}')
    return {'source': record(path), 'embedded_Arial': True, 'text_outside_page': []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-manifest', type=Path, required=True)
    parser.add_argument('--candidate-provenance', type=Path, required=True)
    parser.add_argument('--current-figures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--new-page-visually-reviewed', action='store_true', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    previous = json.loads(args.previous_manifest.read_text())
    candidate = json.loads(args.candidate_provenance.read_text())
    before = {p.name: record(p)['sha256'] for p in args.current_figures.iterdir()}
    assert len(before) == 8 and all(name.endswith('.pdf') for name in before)
    verified(previous['combined_pdf'])
    for rec in candidate['inputs'] + candidate['outputs']:
        verified(rec)
    sources = [verified(rec) for rec in candidate['outputs'] if rec['path'].endswith('.pdf')]
    sources.extend(verified(entry['source']) for entry in previous['page_order'])
    assert len({p.name for p in sources}) == len(sources)
    for name in ['Figure4.pdf', 'Figure4_extended.pdf']:
        entry = next(e for e in previous['page_order'] if e['file'] == name)
        assert entry['source']['sha256'] == before[name], 'Current figure changed since prior plot set'
    checks = [check_pdf(path) for path in sources]
    pdfs = args.output/'pdfs'
    pdfs.mkdir(parents=True, exist_ok=False)
    copies, order = [], []
    writer = PdfWriter()
    for source in sources:
        target = pdfs/source.name
        shutil.copy2(source, target)
        assert record(target)['sha256'] == record(source)['sha256']
        copies.append({'source': record(source), 'copy': record(target)})
        reader = PdfReader(target)
        order.append({'file': target.name, 'first_page': len(writer.pages)+1,
                      'pages': len(reader.pages), 'source': record(target)})
        writer.append(reader, outline_item=target.stem.replace('_', ' '))
    writer.add_metadata({'/Title': 'IF candidate plots after tissue removals', '/Author': 'SpatialTRACE'})
    book_path = args.output/'All_IF_candidate_plots.pdf'
    with book_path.open('xb') as handle:
        writer.write(handle)
    comparisons = []
    with pymupdf.open(book_path) as book:
        assert len(book) == sum(e['pages'] for e in order)
        assert len(book.get_toc()) == len(order)
        for entry in order:
            with pymupdf.open(entry['source']['path']) as source:
                for i, page in enumerate(source):
                    index = entry['first_page']-1+i
                    original = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                    copied = book[index].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
                    assert (original.width, original.height, original.samples) == (copied.width, copied.height, copied.samples)
                    comparisons.append({'book_page': index+1, 'source': entry['file'],
                                        'source_page': i+1, 'status': 'PIXEL_IDENTICAL'})
    assert before == {p.name: record(p)['sha256'] for p in args.current_figures.iterdir()}
    manifest = {'status': 'REVIEWED_AND_VERIFIED', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'command': [sys.executable, *sys.argv], 'code': record(__file__),
                'python': platform.python_version(),
                'packages': {n: importlib.metadata.version(n) for n in ['PyMuPDF', 'pypdf']},
                'git_state': 'Not a Git worktree', 'random_seed': None,
                'previous_manifest': record(args.previous_manifest),
                'candidate_provenance': record(args.candidate_provenance),
                'copies': copies, 'pdf_checks': checks, 'page_order': order,
                'combined_pdf': record(book_path), 'current_figures_unchanged': before,
                'description': 'Density fold-change plot first; preceding candidate pages retained unchanged.',
                'warnings': ['Previous composition and paper-figure P values retain their earlier analysis definitions. Only the new density-fold-change candidate uses the newly confirmed experiment-blocked design.']}
    manifest_path = args.output/'plot_set_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    report = {'status': 'REVIEWED_AND_VERIFIED', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': record(manifest_path)['sha256'], 'comparisons': comparisons,
              'visual_review': 'New density-fold-change PNG inspected for labels, values and spacing. Prior pages are unchanged copies.',
              'font_and_bounds_checks': f'All {len(sources)} PDFs passed embedded-Arial and page-bound checks.',
              'paper_figures_modified': False}
    (args.output/'visual_qa.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'pdf_folder': str(pdfs), 'combined_pdf': str(book_path),
                      'pdfs': len(sources), 'pages_verified': len(comparisons)}, indent=2))


if __name__ == '__main__':
    main()
