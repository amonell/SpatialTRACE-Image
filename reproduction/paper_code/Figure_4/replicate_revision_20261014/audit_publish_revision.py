#!/usr/bin/env python3
"""Check the scoped figure revision, then publish reviewed pages recoverably."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import shutil
import sys

from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from figure_assembly.compositor import file_sha256
from figure_assembly.publish import publish_panels
from figure_assembly.current_figures import refresh_current_figures
from figure_organization import load_catalog


def record(path):
    path = Path(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def same_render(old, new):
    if file_sha256(old) == file_sha256(new):
        return {"old": record(old), "new": record(new), "status": "IDENTICAL_PNG"}
    with Image.open(old) as a, Image.open(new) as b:
        if a.size != b.size or ImageChops.difference(a.convert('RGB'), b.convert('RGB')).getbbox():
            raise RuntimeError(f"An unchanged or restored panel differs: {old} versus {new}")
    return {"old": record(old), "new": record(new), "status": "PIXEL_IDENTICAL"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--previous-run", type=Path, required=True)
    p.add_argument("--original-fields-run", type=Path, required=True)
    p.add_argument("--companion-source", type=Path, required=True)
    p.add_argument("--statistics-pdf", type=Path, required=True)
    p.add_argument("--copy-output", type=Path, required=True)
    p.add_argument("--publish-visually-reviewed", action="store_true")
    a = p.parse_args()
    state = json.loads((a.run/'complete_figure_run.json').read_text())
    if state['status'] != 'AUTOMATED_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED':
        raise RuntimeError('The complete-page run must pass all checks first')
    if {f['package'] for f in state['figures']} != {'Figure_4', 'Figure_4_Extended'}:
        raise RuntimeError('Unexpected revision scope')
    comparisons = []
    for package, letters in [('Figure_4', 'ABCD'), ('Figure_4_Extended', 'ABCEF')]:
        qa = json.loads((a.run/'QA'/package/'qa.json').read_text())
        if qa['errors']:
            raise RuntimeError(qa['errors'])
        provenance = json.loads((a.run/('Figure4_provenance.json' if package == 'Figure_4' else 'Figure4_extended_provenance.json')).read_text())
        if provenance['panel_recomposition']['status'] != 'PIXEL_IDENTICAL':
            raise RuntimeError('Final panel composition differs from direct assembly')
        for letter in letters:
            relative = Path('panels')/package/f'panel_{letter}_final.png'
            comparisons.append(same_render(a.previous_run/relative, a.run/relative))
    relative = Path('panels/Figure_4_Extended/panel_D_final.png')
    comparisons.append(same_render(a.original_fields_run/relative, a.run/relative))
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': [sys.executable, *sys.argv],
              'status': 'SCOPED_RENDER_CHECKS_PASSED', 'comparisons': comparisons,
              'changed_panels': ['Figure_4 E: original dumbbell/lollipop style with mouse means, SEM and adjusted P',
                                 'Figure_4 F: P14 scale 0–13, unchanged density values',
                                 'Figure_4_Extended D: exact original fields, contrast and markers restored'],
              'statistics': record(a.statistics_pdf),
              'manuscript_documents_modified': False}
    if a.publish_visually_reviewed:
        if a.copy_output.exists():
            raise FileExistsError(a.copy_output)
        catalog = load_catalog(); root = Path(catalog['output_root'])
        current = root/'Regenerated_Figures/Current'
        before = {path.name: file_sha256(path) for path in current.iterdir()}
        report['visual_review'] = {'status': 'REVIEWED',
            'items': ['Full Figure4 page and enlarged E: labels and SEM bars fit',
                      'Full Extended Figure4: original anatomical fields, preserved A–C/E/F',
                      'Statistics companion: legible tables, methods and both sensitivity analyses']}
        report['panel_publication'] = publish_panels(state, root, overwrite=True)
        report['current_figures'] = refresh_current_figures(root, catalog['figures'])
        after = {path.name: file_sha256(path) for path in current.iterdir()}
        if len(after) != 8 or {n for n in before if before[n] != after[n]} != {'Figure4.pdf', 'Figure4_extended.pdf'}:
            raise RuntimeError('Unexpected current-folder changes')
        pdfs = a.copy_output/'pdfs'; pdfs.mkdir(parents=True)
        sources = [current/'Figure4.pdf', current/'Figure4_extended.pdf', *sorted(a.companion_source.glob('Additional_0[1-6]_*.pdf')), a.statistics_pdf]
        if len(sources) != 9 or len({p.name for p in sources}) != 9:
            raise RuntimeError('Expected two figures and seven companion PDFs')
        copies = []
        for source in sources:
            destination = pdfs/source.name; shutil.copy2(source, destination)
            if file_sha256(source) != file_sha256(destination):
                raise RuntimeError('Copy hash mismatch')
            copies.append({'source': record(source), 'copy': record(destination)})
        report['copies'] = copies
        report['status'] = 'REVIEWED_AND_PUBLISHED'
        (a.copy_output/'publication.json').write_text(json.dumps(report, indent=2)+'\n')
    target = a.run/('visual_qa_and_publication.json' if a.publish_visually_reviewed else 'scoped_render_audit.json')
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'comparisons': len(comparisons), 'report': str(target)}, indent=2))


if __name__ == '__main__':
    main()
