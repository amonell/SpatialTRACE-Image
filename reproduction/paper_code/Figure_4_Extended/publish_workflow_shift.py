#!/usr/bin/env python3
"""Verify a pure panel-B workflow translation before publishing the page."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import sys

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Figure_4.publish_selected_fields import normalized_run_paths
from figure_assembly.compositor import file_sha256
from figure_assembly.current_figures import refresh_current_figures
from figure_assembly.publish import publish_panels
from figure_organization import load_catalog


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_translation(old, new, delta):
    """Check every element, allowing only the requested vertical translation."""
    check(len(old) == len(new), 'Workflow element count changed')
    for a, b in zip(old, new, strict=True):
        expected = dict(a)
        if a['kind'] == 'pdf':
            rect = list(a['rect_pt'])
            rect[1] += delta
            expected['rect_pt'] = rect
        elif a['kind'] == 'text':
            if a['text'].lower().rstrip('.') != 'b':
                expected['y_pt'] += delta
        else:
            raise RuntimeError('Unexpected workflow element type')
        check(expected == b, 'Change beyond translation: '+a['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous-run', type=Path, required=True)
    parser.add_argument('--shift-pt', type=float, required=True)
    parser.add_argument('--publish-visually-reviewed', action='store_true')
    args = parser.parse_args()
    report_path = args.run / ('visual_qa_and_publication.json' if args.publish_visually_reviewed
                              else 'scoped_render_audit.json')
    if report_path.exists():
        raise FileExistsError(report_path)
    catalog = load_catalog()
    root = Path(catalog['output_root'])
    current = root/'Regenerated_Figures/Current'
    before = {p.name:file_sha256(p) for p in current.iterdir()}
    originals = {p.name:file_sha256(p) for p in (root/'Figures').glob('*.pdf')}
    check(before['Figure4_extended.pdf'] == file_sha256(args.previous_run/'Figure4_extended.pdf'),
          'Current baseline has changed')
    state = json.loads((args.run/'complete_figure_run.json').read_text())
    check(state['status'] == 'AUTOMATED_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED', 'Run QA incomplete')
    check([f['package'] for f in state['figures']] == ['Figure_4_Extended'], 'Unexpected figure scope')
    figure = state['figures'][0]
    check([p['panel'] for p in figure['final_panels']] == list('ABCDEFG'), 'Unexpected panel order')
    qa = json.loads((args.run/'QA/Figure_4_Extended/qa.json').read_text())
    check(not qa['errors'], 'Page QA failed')
    old = json.loads((args.previous_run/'Figure4_extended_provenance.json').read_text())
    new = json.loads((args.run/'Figure4_extended_provenance.json').read_text())
    check(new['panel_recomposition']['status'] == 'PIXEL_IDENTICAL', 'Panel recomposition differs')
    prepared = new['preparation']
    check(not any(prepared[k] for k in ('model_inference_run','model_fitting_run','metric_tables_created')),
          'Unexpected scientific computation')
    old_command, new_command = old['prepare_command'], new['prepare_command']
    old_root = old_command[old_command.index('--output-dir')+1]
    new_root = new_command[new_command.index('--output-dir')+1]
    old_details = normalized_run_paths(old['preparation']['details'], old_root)
    new_details = normalized_run_paths(prepared['details'], new_root)
    command = list(new_details['IEL_CCR9']['command'])
    check(command[-2] == '--renderer-source-dir', 'Missing explicit frozen renderer copies')
    new_details['IEL_CCR9']['command'] = command[:-2]
    check(old_details == new_details, 'Scientific details changed')
    old_sources = {Path(r['path']).name:r['sha256'] for r in old['preparation']['inputs']
                   if r.get('role') == 'approved_flow_renderer'}
    source_copies = [r for r in prepared['inputs'] if r.get('role') == 'approved_flow_renderer']
    check({Path(r['path']).name:r['sha256'] for r in source_copies} == old_sources,
          'Archived renderer copies do not match the prior release')
    for record in prepared['inputs'] + new['outputs']:
        check(file_sha256(record['path']) == record['sha256'], 'Changed input/output: '+record['path'])
    comparisons = {}
    for panel in figure['final_panels']:
        letter = panel['panel']
        previous = args.previous_run/f'panels/Figure_4_Extended/panel_{letter}_final.png'
        with Image.open(previous) as a, Image.open(panel['png']) as b:
            check(a.size == b.size, 'Panel canvas changed')
            box = ImageChops.difference(a.convert('RGBA'), b.convert('RGBA')).getbbox(alpha_only=False)
            if letter == 'B':
                check(box is not None, 'Workflow was not moved')
                comparisons[letter] = {'changed_bbox_px':box,'vertical_translation_pt':args.shift_pt}
            else:
                check(box is None, 'Unrequested change to panel '+letter)
                comparisons[letter] = 'PIXEL_IDENTICAL'
        previous_info = json.loads((args.previous_run/
            f'panels/Figure_4_Extended/panel_{letter}_final_provenance.json').read_text())
        new_info = json.loads(Path(panel['provenance']).read_text())
        if letter == 'B':
            check_translation(previous_info['elements'], new_info['elements'], args.shift_pt)
        else:
            check(previous_info['elements'] == new_info['elements'], 'Unrequested layout change: '+letter)
    report = {'status':'SCOPED_RENDER_CHECKS_PASSED',
              'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'command':[sys.executable,*sys.argv], 'code_sha256':file_sha256(__file__),
              'baseline_run':str(args.previous_run), 'panels':comparisons,
              'workflow':'Original vectors and all internal text translated together; panel letter unchanged',
              'scientific_details':'IDENTICAL',
              'identical_renderer_source_copies':source_copies,
              'selected_fields':prepared['details']['selected_anatomical_fields'],
              'new_analysis_or_inference':False, 'documents_modified':False}
    if args.publish_visually_reviewed:
        check(before == {p.name:file_sha256(p) for p in current.iterdir()}, 'Current changed during audit')
        report['visual_review'] = 'Full page and enlarged workflow inspected: balanced spacing, aligned text and arrows, no clipping; option 32 retained.'
        report['panel_publication'] = publish_panels(state, root, overwrite=True)
        report['current_figures'] = refresh_current_figures(root, catalog['figures'])
        after = {p.name:file_sha256(p) for p in current.iterdir()}
        check(set(before) == set(after) and len(after) == 8, 'Current must contain eight PDFs')
        check({name for name in before if before[name] != after[name]} == {'Figure4_extended.pdf'},
              'Unexpected changes to other figures')
        check(originals == {p.name:file_sha256(p) for p in (root/'Figures').glob('*.pdf')},
              'Original artwork changed')
        report.update(status='REVIEWED_AND_PUBLISHED', current_hashes_before=before,
                      current_hashes_after=after)
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status':report['status'],'report':str(report_path)},indent=2))


if __name__ == '__main__':
    main()
