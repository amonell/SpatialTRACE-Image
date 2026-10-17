#!/usr/bin/env python3
"""Audit the display-only E/F/G size and marker refinement before publication."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import subprocess
import sys
import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from figure_assembly.compositor import file_sha256
from figure_assembly.current_figures import refresh_current_figures
from figure_assembly.publish import publish_panels
from figure_organization import load_catalog


def equal(left, right):
    assert json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def audit_science(old, new):
    a, b = old['preparation'], new['preparation']
    assert not any(b[k] for k in ('model_inference_run', 'model_fitting_run', 'metric_tables_created'))
    for key in ('approved_EF', 'replicate_cohort', 'strict_test'):
        equal(a['details'][key], b['details'][key])
    before = {v['path']: v['sha256'] for v in a['inputs'] if v['path'].startswith('/mnt/sata1/')}
    after = {v['path']: v['sha256'] for v in b['inputs'] if v['path'].startswith('/mnt/sata1/')}
    equal(before, after)
    aa = {r['id']: np.asarray(r['rect_pt']) for r in a['rendered_anchors']}
    bb = {r['id']: np.asarray(r['rect_pt']) for r in b['rendered_anchors']}
    for condition in ('DMSO', 'RARi'):
        old_image, new_image = a['details']['microscopy_'+condition], b['details']['microscopy_'+condition]
        for key in old_image:
            if key != 'bar_length_pt':
                equal(old_image[key], new_image[key])
        np.testing.assert_allclose(old_image['bar_length_pt']/aa[f'A_{condition}_image'][2],
                                   new_image['bar_length_pt']/bb[f'A_{condition}_image'][2], atol=1e-12)
        key = f'replicate_{condition}_p14_imap'
        for name in a['details'][key]:
            if name != 'gate_labels':
                equal(a['details'][key][name], b['details'][key][name])
        old_labels = a['details'][key]['gate_labels']
        new_labels = b['details'][key]['gate_labels']
        equal(old_labels['source'], new_labels['source'])
        old_axis, new_axis = aa[f'F_{condition}'], bb[f'F_{condition}']
        scale = new_axis[2:]/old_axis[2:]
        np.testing.assert_allclose(new_labels['font_size_pt'], 5.7, atol=1e-10)
        def transformed(rect):
            return [*(new_axis[:2] + (np.asarray(rect[:2])-old_axis[:2])*scale),
                    *(np.asarray(rect[2:])*scale if len(rect) > 2 else [])]
        for x, y in zip(old_labels['boxes'], new_labels['boxes']):
            assert x['gate'] == y['gate']
            np.testing.assert_allclose(y['rect_pt'], transformed(x['rect_pt']), atol=1e-8)
        for x, y in zip(old_labels['text_baselines'], new_labels['text_baselines']):
            assert x['text'] == y['text']
            np.testing.assert_allclose([y['x_pt'], y['y_pt']], transformed([x['x_pt'], x['y_pt']]), atol=1e-8)
    for key in a['details']['sparse_example']:
        if key not in ('authorized_image_height_pt', 'bar_length_pt'):
            equal(a['details']['sparse_example'][key], b['details']['sparse_example'][key])
    np.testing.assert_allclose(a['details']['sparse_example']['bar_length_pt']/aa['B_image'][2],
                               b['details']['sparse_example']['bar_length_pt']/bb['B_image'][2], atol=1e-12)
    for i in range(4):
        key = f'whole_map_{i}'
        for field in a['details'][key]:
            if field not in ('frame_rect_pt', 'contained_image_rect_pt'):
                equal(a['details'][key][field], b['details'][key][field])
    return {'frozen_SATA_inputs': len(before), 'all_mouse_values_tests_and_gate_percentages': 'IDENTICAL',
            'images_and_magnified_fields': 'IDENTICAL', 'physical_scale_calibrations': 'IDENTICAL',
            'IMAP_labels': 'Same relative positions in narrower axes; 5.7-pt Arial text; values unchanged',
            'sparse_image_linear_scale_factor': float(bb['B_image'][2]/aa['B_image'][2])}


def audit_alignment(prep):
    a = {v['id']: np.asarray(v['rect_pt']) for v in prep['rendered_anchors']}
    def same(names, fields):
        values = np.stack([a[n][fields] for n in names])
        np.testing.assert_allclose(values, np.broadcast_to(values[0], values.shape), atol=1e-7)
    same(['A_DMSO_image', 'A_RARi_image', 'B_image'], [1, 3])
    same(['C_crypt_villus', 'C_epithelial_distance', 'C_crypt_villus_bar', 'C_epithelial_distance_bar'], [1, 3])
    same([f'D_zoom_frame_{i}' for i in range(4)], [1, 2, 3])
    same([f'D_colorbar_{i}' for i in range(4)], [1, 2, 3])
    same(['F_DMSO', 'F_RARi'], [1, 2, 3])
    np.testing.assert_allclose(a['E_abundance'][0]+a['E_abundance'][2]/2,
                               a['G_enrichment'][0]+a['G_enrichment'][2]/2, atol=1e-7)
    np.testing.assert_allclose(a['E_abundance'][2:], [128, 120], rtol=0, atol=1e-7)
    np.testing.assert_allclose(a['G_enrichment'][2], 176, rtol=0, atol=1e-7)
    np.testing.assert_allclose([a['F_DMSO'][2], a['F_RARi'][2]], [152, 152], rtol=0, atol=1e-7)
    assert prep['details']['display_alignment']['replicate_marker_area_pt2'] == 25.0
    same(['F_DMSO', 'G_enrichment'], [1, 3])
    assert a['E_abundance'][1]+a['E_abundance'][3] < a['G_enrichment'][1]
    assert a['F_RARi'][0]+a['F_RARi'][2] < a['G_enrichment'][0]
    for i in range(4):
        zoom, bar = a[f'D_zoom_frame_{i}'], a[f'D_colorbar_{i}']
        np.testing.assert_allclose(zoom[0]+zoom[2]/2, bar[0]+bar[2]/2, atol=1e-7)
    left, right, bar = a['F_DMSO'], a['F_RARi'], a['F_shared_density_bar']
    np.testing.assert_allclose((left[0]+right[0]+right[2])/2, bar[0]+bar[2]/2, atol=1e-7)
    letters = {e['panel']: e for e in prep['elements'] if e['kind'] == 'text'}
    for left, right in (('A', 'B'), ('D', 'E'), ('F', 'G')):
        assert letters[left]['y_pt'] == letters[right]['y_pt']
    return {'tolerance_pt': 1e-7, 'status': 'ALIGNED',
            'checked': ['image tops/bottoms', 'scatter axes/colorbars', 'zoom frames/colorbars',
                        'side-by-side IMAP axes/shared bar center', 'E above G in same column', 'F/G row baselines', 'panel-letter row baselines']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--previous-run', type=Path, required=True)
    p.add_argument('--publish-visually-reviewed', action='store_true')
    args = p.parse_args()
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
    for letter in 'ABCD':
        relative = f'panels/Figure_4/panel_{letter}_final.png'
        with Image.open(args.previous_run/relative) as previous, Image.open(args.run/relative) as current:
            assert ImageChops.difference(previous.convert('RGB'), current.convert('RGB')).getbbox() is None, letter
    science = audit_science(old, new)
    alignment = audit_alignment(new['preparation'])
    for item in new['preparation']['inputs'] + new['outputs']:
        assert file_sha256(item['path']) == item['sha256'], item['path']
    text = subprocess.run(['pdftotext', '-raw', str(args.run/'Figure4.pdf'), '-'],
                          capture_output=True, text=True, check=True).stdout
    for phrase in ('P = 0.004', 'q = 0.046', 'q = 0.016', 'DMSO P14', 'RARi P14', 'Probability density'):
        assert phrase in text, phrase
    report = {'status': 'DISPLAY_ONLY_CHECKS_PASSED', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'code_sha256': file_sha256(__file__),
              'science': science, 'alignment': alignment, 'documents_modified': False,
              'unchanged_panels': 'A-D pixel-identical',
              'panel_order': 'UNCHANGED: E abundance upper right, F IMAP lower left, G enrichment lower right',
              'display_changes': {'replicate_diameter_factor': float(np.sqrt(25/11.5)),
                                  'E_width_factor': 128/156, 'E_height_factor': 120/136.5,
                                  'G_width_factor': 176/156, 'F_width_factor': 152/160}}
    if args.publish_visually_reviewed:
        catalog = load_catalog()
        root = Path(catalog['output_root'])
        current = root/'Regenerated_Figures/Current'
        before = {p.name: file_sha256(p) for p in current.iterdir()}
        assert before['Figure4.pdf'] == file_sha256(args.previous_run/'Figure4.pdf')
        report['visual_review'] = 'Complete PDF and enlarged regions inspected: aligned images, headings, axes, zooms, colorbars and panel letters; no clipped or overlapping labels.'
        report['panel_publication'] = publish_panels(state, root, overwrite=True)
        report['current_figures'] = refresh_current_figures(root, catalog['figures'])
        after = {p.name: file_sha256(p) for p in current.iterdir()}
        assert set(after) == set(before) and len(after) == 8
        assert {name for name in before if before[name] != after[name]} == {'Figure4.pdf'}
        report.update(status='REVIEWED_AND_PUBLISHED', current_hashes_before=before, current_hashes_after=after)
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'report': str(target)}, indent=2))


if __name__ == '__main__':
    main()
