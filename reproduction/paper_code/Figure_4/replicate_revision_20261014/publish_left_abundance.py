#!/usr/bin/env python3
"""Verify that only Figure 4E's left subplot changed, then publish recoverably."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import sys

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Figure_4.replicate_revision_20261014.publish_post_removal_figures import compare, record
from figure_assembly.compositor import file_sha256
from figure_assembly.current_figures import refresh_current_figures
from figure_assembly.publish import publish_panels
from figure_organization import load_catalog


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
    qa = json.loads((args.run/'QA/Figure_4/qa.json').read_text())
    assert not qa['errors'], qa['errors']
    provenance = json.loads((args.run/'Figure4_provenance.json').read_text())
    assert provenance['panel_recomposition']['status'] == 'PIXEL_IDENTICAL'
    prepared = provenance['preparation']
    assert not prepared['model_inference_run'] and not prepared['metric_tables_created']
    for rec in prepared['inputs'] + provenance['outputs']:
        assert file_sha256(rec['path']) == rec['sha256'], rec['path']
    details = prepared['details']['replicate_regional_summary']
    assert details['left_endpoint'] == 'p14_per_10000_non_p14'
    assert details['right_endpoint'] == 'observed_over_expected_p14_density'
    previous = json.loads((args.previous_run/'Figure4_provenance.json').read_text())
    assert details['displayed_statistics'] == previous['preparation']['details']['replicate_regional_summary']['displayed_statistics']
    comparisons = []
    for letter in 'ABCDEF':
        relative = Path('panels/Figure_4') / f'panel_{letter}_final.png'
        result = compare(args.previous_run/relative, args.run/relative)
        assert result['pixel_identical'] == (letter != 'E'), letter
        comparisons.append({'panel': letter, **result})
    # Full-page pixel comparison also protects the right subplot and shared legend.
    with Image.open(args.previous_run/'Figure4.png') as old, Image.open(args.run/'Figure4.png') as new:
        assert old.size == new.size
        box = ImageChops.difference(old.convert('RGB'), new.convert('RGB')).getbbox()
        assert box is not None
        width, height = old.size
        allowed = (0, int(648/841.89*height), int(205/595.276*width), height)
        assert box[0] >= allowed[0] and box[1] >= allowed[1] and box[2] <= allowed[2] and box[3] <= allowed[3], box
    report = {'status': 'SCOPED_RENDER_CHECKS_PASSED', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'code': record(Path(__file__)),
              'comparisons': comparisons, 'changed_pixel_bbox': box, 'allowed_left_subplot_bbox': allowed,
              'right_subplot_and_rest_of_page': 'PIXEL_IDENTICAL',
              'change': 'Left E restored to P14 per 10,000 non-P14 cells in the same gate, using filtered mouse means and SEM',
              'manuscript_documents_modified': False}
    if args.publish_visually_reviewed:
        catalog = load_catalog()
        root = Path(catalog['output_root'])
        current = root/'Regenerated_Figures/Current'
        before = {p.name: file_sha256(p) for p in current.iterdir()}
        assert before['Figure4.pdf'] == file_sha256(args.previous_run/'Figure4.pdf')
        report['visual_review'] = 'Full page and enlarged E reviewed: labels, error bars and spacing are clear.'
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
