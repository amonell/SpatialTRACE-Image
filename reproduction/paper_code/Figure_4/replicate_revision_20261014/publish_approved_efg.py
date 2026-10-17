#!/usr/bin/env python3
"""Validate the authorized E/F replacement and G cleanup before publication."""
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
    assert [p['panel'] for p in state['figures'][0]['final_panels']] == list('ABCDEFG')
    qa = json.loads((args.run/'QA/Figure_4/qa.json').read_text())
    assert not qa['errors'], qa['errors']
    provenance = json.loads((args.run/'Figure4_provenance.json').read_text())
    assert provenance['panel_recomposition']['status'] == 'PIXEL_IDENTICAL'
    prepared = provenance['preparation']
    assert not any(prepared[k] for k in ('model_inference_run', 'model_fitting_run', 'metric_tables_created'))
    for item in prepared['inputs'] + provenance['outputs']:
        assert file_sha256(item['path']) == item['sha256'], item['path']
    details = prepared['details']
    previous = json.loads((args.previous_run/'Figure4_provenance.json').read_text())['preparation']
    for name in ('replicate_cohort', 'replicate_DMSO_p14_imap', 'replicate_RARi_p14_imap'):
        assert details[name] == previous['details'][name], name
    approved = details['approved_EF']
    assert len(approved['all_mouse_values']) == 9 and len(approved['all_gate_values']) == 45
    assert len(approved['statistics']) == 6
    assert approved['palette'] == {'DMSO': '#4F7F8A', 'RARi': '#A55D54'}
    significant = {(r['metric'], r['gate']) for r in approved['statistics']
                   if r['p_ols_batch' if r['metric']=='log2_p14_per10k' else 'q_ols_batch'] < .05}
    assert significant == {('log2_p14_per10k', ''), ('log2_enrich', 'Top LP'), ('log2_enrich', 'Muscularis')}
    comparisons = []
    for letter in 'ABCD':
        relative = Path('panels/Figure_4') / f'panel_{letter}_final.png'
        comparison = compare(args.previous_run/relative, args.run/relative)
        assert comparison['pixel_identical'], letter
        comparisons.append({'panel': letter, **comparison})
    # Between the letter and removed footer, the entire IMAP must remain identical.
    with Image.open(args.previous_run/'panels/Figure_4/panel_F_final.png') as old, \
         Image.open(args.run/'panels/Figure_4/panel_G_final.png') as new:
        assert old.size == new.size
        w, h = old.size
        box = (0, round(448/841.89*h), w, round(765/841.89*h))
        bands = ImageChops.difference(old.convert('RGBA').crop(box), new.convert('RGBA').crop(box)).split()
        assert all(b.getbbox() is None for b in bands), 'IMAP map/axes/colorbar changed'
    report = {'status': 'SCOPED_RENDER_CHECKS_PASSED', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'code': record(Path(__file__)),
              'comparisons': comparisons, 'imap_map_axes_colorbar': 'PIXEL_IDENTICAL',
              'imap_comparison_box_px': box, 'changed_panels': ['E', 'F', 'G'],
              'approved_analysis': approved['analysis_manifest'], 'new_analysis_or_inference': False}
    if args.publish_visually_reviewed:
        catalog = load_catalog()
        root = Path(catalog['output_root'])
        current = root/'Regenerated_Figures/Current'
        before = {p.name: file_sha256(p) for p in current.iterdir()}
        assert before['Figure4.pdf'] == file_sha256(args.previous_run/'Figure4.pdf')
        report['visual_review'] = 'Full page and E/F checked: readable labels, matched approved colors, no collisions; G has no explanatory footer.'
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
