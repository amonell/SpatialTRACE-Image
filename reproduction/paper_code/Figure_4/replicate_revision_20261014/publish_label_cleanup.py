#!/usr/bin/env python3
"""Publish the Figure 4 title/sample-size cleanup after a scoped pixel audit."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import re
import sys
from PIL import Image, ImageChops, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Figure_4.replicate_revision_20261014.publish_post_removal_figures import compare, record
from figure_assembly.compositor import file_sha256
from figure_assembly.current_figures import refresh_current_figures
from figure_assembly.publish import publish_panels
from figure_organization import load_catalog

TEXT_REGIONS = {
    'E_title': [40, 655, 125, 668],
    'F_title': [165, 655, 355, 668],
    'EF_treatment_legend': [115, 802, 260, 820],
    'G_DMSO_heading': [400, 448, 585, 463],
    'G_RARi_heading': [400, 583, 585, 597],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous-run', type=Path, required=True)
    parser.add_argument('--publish-visually-reviewed', action='store_true')
    args = parser.parse_args()
    target = args.run / ('visual_qa_and_publication.json' if args.publish_visually_reviewed else 'scoped_render_audit.json')
    if target.exists():
        raise FileExistsError(target)
    state = json.loads((args.run/'complete_figure_run.json').read_text())
    assert state['status'] == 'AUTOMATED_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED'
    assert [f['package'] for f in state['figures']] == ['Figure_4']
    assert [p['panel'] for p in state['figures'][0]['final_panels']] == list('ABCDEFG')
    assert not json.loads((args.run/'QA/Figure_4/qa.json').read_text())['errors']
    new = json.loads((args.run/'Figure4_provenance.json').read_text())
    old = json.loads((args.previous_run/'Figure4_provenance.json').read_text())
    assert new['panel_recomposition']['status'] == 'PIXEL_IDENTICAL'
    prep = new['preparation']
    assert not any(prep[k] for k in ('model_inference_run', 'model_fitting_run', 'metric_tables_created'))
    assert json.dumps(prep['details'], sort_keys=True) == json.dumps(old['preparation']['details'], sort_keys=True)
    for item in prep['inputs'] + new['outputs']:
        assert file_sha256(item['path']) == item['sha256'], item['path']
    pdf_text = '\n'.join(p.extract_text() for p in PdfReader(args.run/'Figure4.pdf').pages)
    assert not re.search(r'\bn\s*=', pdf_text, re.I)
    for title in ('P14 abundance', 'P14 enrichment by compartment'):
        assert title not in pdf_text
    for retained in ('P = 0.004', 'q = 0.046', 'q = 0.016', 'DMSO P14', 'RARi P14', 'Probability density'):
        assert retained in pdf_text, retained
    comparisons = []
    for letter in 'ABCD':
        rel = Path('panels/Figure_4')/f'panel_{letter}_final.png'
        result = compare(args.previous_run/rel, args.run/rel)
        assert result['pixel_identical'], letter
        comparisons.append({'panel': letter, **result})
    with Image.open(args.previous_run/'Figure4.png') as previous, Image.open(args.run/'Figure4.png') as current:
        assert previous.size == current.size
        difference = ImageChops.difference(previous.convert('RGB'), current.convert('RGB'))
        bbox = difference.getbbox()
        assert bbox is not None
        draw = ImageDraw.Draw(difference)
        w, h = difference.size
        for rect in TEXT_REGIONS.values():
            pixels = tuple(round(v*(w/595.276 if i%2==0 else h/841.89)) for i,v in enumerate(rect))
            draw.rectangle(pixels, fill='black')
        assert difference.getbbox() is None, ('Change outside approved text regions', difference.getbbox())
    report = {'status': 'SCOPED_RENDER_CHECKS_PASSED', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'code': record(__file__),
              'unchanged_panels': comparisons, 'allowed_text_regions_pt': TEXT_REGIONS,
              'changed_pixel_bbox': bbox, 'all_pixels_outside_text_regions': 'IDENTICAL',
              'scientific_details': 'IDENTICAL', 'sample_size_labels_and_EF_titles': 'ABSENT',
              'new_analysis_or_inference': False, 'documents_modified': False}
    if args.publish_visually_reviewed:
        catalog = load_catalog()
        root = Path(catalog['output_root'])
        current = root/'Regenerated_Figures/Current'
        before = {p.name: file_sha256(p) for p in current.iterdir()}
        assert before['Figure4.pdf'] == file_sha256(args.previous_run/'Figure4.pdf')
        report['visual_review'] = 'Complete page inspected: titles and sample sizes removed; panel letters, axis labels, treatment legend and significance annotations retained.'
        report['panel_publication'] = publish_panels(state, root, overwrite=True)
        report['current_figures'] = refresh_current_figures(root, catalog['figures'])
        after = {p.name: file_sha256(p) for p in current.iterdir()}
        assert set(after) == set(before) and len(after) == 8
        assert {n for n in before if before[n] != after[n]} == {'Figure4.pdf'}
        report.update(status='REVIEWED_AND_PUBLISHED', current_hashes_before=before, current_hashes_after=after)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'report': str(target)}, indent=2))


if __name__ == '__main__':
    main()
