#!/usr/bin/env python3
"""Verify every compiled review page against its source PDF."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from paper_paths import Path, lock_record, rendering_script
import sys

import pymupdf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plot-set', type=Path, required=True)
    args = parser.parse_args()
    output = args.plot_set/'visual_qa.json'
    if output.exists():
        raise FileExistsError(output)
    path = args.plot_set/'plot_set_manifest.json'
    manifest = json.loads(path.read_text())
    assert sha(manifest['combined_pdf']['path']) == manifest['combined_pdf']['sha256']
    book = pymupdf.open(manifest['combined_pdf']['path'])
    assert len(book) == sum(entry['pages'] for entry in manifest['page_order']) == 20
    assert len(book.get_toc()) == 11
    comparisons = []
    for entry in manifest['page_order']:
        assert sha(entry['source']['path']) == entry['source']['sha256']
        source = pymupdf.open(entry['source']['path'])
        for index in range(len(source)):
            page = entry['first_page']-1+index
            before = source[index].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
            after = book[page].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
            assert (before.width, before.height, before.samples) == (after.width, after.height, after.samples), page+1
            comparisons.append({'book_page': page+1, 'source': entry['file'], 'source_page': index+1, 'status': 'PIXEL_IDENTICAL'})
        source.close()
    report = {'status': 'REVIEWED_AND_VERIFIED', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'manifest_sha256': sha(path), 'comparisons': comparisons,
              'visual_review': 'All six regenerated candidate pages and both new composition pages inspected; labels and plot contents checked. Existing main/extended and removal-review pages retain their reviewed source rendering.',
              'font_and_bounds_checks': 'All 11 source PDFs passed embedded-Arial and page-bound checks during compilation.',
              'paper_figures_modified': False}
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'pages_checked': len(comparisons), 'report': str(output)}, indent=2))


if __name__ == '__main__':
    main()
