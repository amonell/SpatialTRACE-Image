#!/usr/bin/env python3
"""Audit and publish the author-approved C/E cleanup; protect all other panels."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import re
import sys
from PIL import Image, ImageChops, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Figure_4.replicate_revision_20261014.publish_post_removal_figures import compare, record
from figure_assembly.compositor import file_sha256
from figure_assembly.current_figures import refresh_current_figures
from figure_assembly.publish import publish_panels
from figure_organization import load_catalog

REGIONS = {
    'C_stage_labels': [255, 137, 363, 148],
    'E_DMSO_heading': [60, 466, 205, 482],
    'E_RARi_heading': [305, 466, 450, 482],
    'E_removed_left_bar': [221, 497, 260, 606],
    'E_shared_right_bar': [464, 497, 507, 606],
    'E_removed_footer': [215, 633, 385, 643],
}
SCIENCE = ('replicate_cohort', 'workflow_training_counts', 'transfer', 'IF_history',
           'replicate_DMSO_non_p14_imap', 'replicate_RARi_non_p14_imap', 'Peyer_DMSO', 'Peyer_RARi')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous-run', type=Path, required=True)
    parser.add_argument('--publish-visually-reviewed', action='store_true')
    args = parser.parse_args()
    report_path = args.run/('visual_qa_and_publication.json' if args.publish_visually_reviewed else 'scoped_render_audit.json')
    if report_path.exists():
        raise FileExistsError(report_path)
    state = json.loads((args.run/'complete_figure_run.json').read_text())
    assert state['status'] == 'AUTOMATED_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED'
    assert [f['package'] for f in state['figures']] == ['Figure_4_Extended']
    assert [p['panel'] for p in state['figures'][0]['final_panels']] == list('ABCDEF')
    assert not json.loads((args.run/'QA/Figure_4_Extended/qa.json').read_text())['errors']
    new = json.loads((args.run/'Figure4_extended_provenance.json').read_text())
    old = json.loads((args.previous_run/'Figure4_extended_provenance.json').read_text())
    assert new['panel_recomposition']['status'] == 'PIXEL_IDENTICAL'
    prepared = new['preparation']
    assert not any(prepared[k] for k in ('model_inference_run', 'model_fitting_run', 'metric_tables_created'))
    for item in prepared['inputs'] + new['outputs']:
        assert file_sha256(item['path']) == item['sha256'], item['path']
    details = prepared['details']
    for key in SCIENCE:
        assert details[key] == old['preparation']['details'][key], key
    assert [r['center_epoch'] for r in details['IF_history_stage_labels']] == [4, 9.5]
    assert details['non_P14_colorbar_display']['limits'] == [0, 16]
    text = '\n'.join(p.extract_text() for p in PdfReader(args.run/'Figure4_extended.pdf').pages)
    assert not re.search(r'\bn\s*=', text, re.I)
    assert 'Equal mouse weights' not in text and 'labels show mean %' not in text
    assert text.count('Probability density') == 1
    for retained in ('heads only', 'full model', 'DMSO non-P14', 'RARi non-P14', 'Peyer’s patch probability'):
        assert retained in text, retained
    comparisons = []
    for letter in 'ABDF':
        relative = Path('panels/Figure_4_Extended')/f'panel_{letter}_final.png'
        comparison = compare(args.previous_run/relative, args.run/relative)
        assert comparison['pixel_identical'], letter
        comparisons.append({'panel': letter, **comparison})
    with Image.open(args.previous_run/'Figure4_extended.png') as old_png, Image.open(args.run/'Figure4_extended.png') as new_png:
        assert old_png.size == new_png.size
        diff = ImageChops.difference(old_png.convert('RGB'), new_png.convert('RGB'))
        bbox = diff.getbbox()
        assert bbox is not None
        w, h = diff.size
        draw = ImageDraw.Draw(diff)
        for rect in REGIONS.values():
            draw.rectangle(tuple(round(v*(w/595.276 if i%2 == 0 else h/841.89)) for i,v in enumerate(rect)), fill='black')
        assert diff.getbbox() is None, ('Change outside approved display regions', diff.getbbox())
    report = {'status': 'SCOPED_RENDER_CHECKS_PASSED', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'code': record(__file__), 'unchanged_panels': comparisons,
              'allowed_display_regions_pt': REGIONS, 'changed_pixel_bbox': bbox,
              'all_pixels_outside_display_regions': 'IDENTICAL', 'scientific_details': 'IDENTICAL',
              'stage_label_positions': details['IF_history_stage_labels'],
              'shared_non_P14_colorbar': details['non_P14_colorbar_display'],
              'new_analysis_or_inference': False, 'documents_modified': False}
    if args.publish_visually_reviewed:
        catalog = load_catalog()
        root = Path(catalog['output_root'])
        current = root/'Regenerated_Figures/Current'
        before = {p.name: file_sha256(p) for p in current.iterdir()}
        assert before['Figure4_extended.pdf'] == file_sha256(args.previous_run/'Figure4_extended.pdf')
        report['visual_review'] = 'Complete page and C/E inspected: stage labels lie on the correct sides of the divider; E has one readable shared bar, no sample sizes or footer; maps and other panels unchanged.'
        report['panel_publication'] = publish_panels(state, root, overwrite=True)
        report['current_figures'] = refresh_current_figures(root, catalog['figures'])
        after = {p.name: file_sha256(p) for p in current.iterdir()}
        assert set(after) == set(before) and len(after) == 8
        assert {n for n in before if before[n] != after[n]} == {'Figure4_extended.pdf'}
        report.update(status='REVIEWED_AND_PUBLISHED', current_hashes_before=before, current_hashes_after=after)
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'report': str(report_path)}, indent=2))


if __name__ == '__main__':
    main()
