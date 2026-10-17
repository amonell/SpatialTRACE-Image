#!/usr/bin/env python3
"""Check the approved shared fields and publish after visual review."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import sys

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from figure_assembly.compositor import file_sha256
from figure_assembly.current_figures import refresh_current_figures
from figure_assembly.publish import publish_panels
from figure_organization import load_catalog
from Figure_4.field_selection import load_fields

BASELINES = {
    'Figure_4': ('Figure4', 'if_replicates_20261015_EFG_size_polish_v2',
                 'ec64ccec6ada4fc41f228de2dbc860d18c85f5ae013440d749d4d0e54a5725c5'),
    'Figure_4_Extended': ('Figure4_extended', 'if_replicates_20261015_extended_cleanup_v1',
                          '613adcb1d9d1b8c8281276df5f2b0f76127799d89880f392664513dba71cc721'),
}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def normalized_run_paths(value, root):
    """Ignore only the fresh output directory, retaining all scientific values."""
    if isinstance(value, dict):
        return {k: normalized_run_paths(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [normalized_run_paths(v, root) for v in value]
    if isinstance(value, str) and value.startswith(root + '/'):
        return '<PREPARED_RUN>/' + value[len(root)+1:]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline-run', action='append', default=[], metavar='PACKAGE=PATH',
                        help='Explicit previous run for each figure; must match the current published PDFs')
    parser.add_argument('--expected-rari-id', default='R1')
    parser.add_argument('--publish-visually-reviewed', action='store_true')
    args = parser.parse_args()
    report_path = args.run / ('visual_qa_and_publication.json' if args.publish_visually_reviewed
                              else 'scoped_render_audit.json')
    if report_path.exists():
        raise FileExistsError(report_path)
    catalog = load_catalog()
    data = Path(catalog['output_root'])
    current = data/'Regenerated_Figures/Current'
    before = {p.name: file_sha256(p) for p in current.iterdir()}
    original = {p.name: file_sha256(p) for p in (data/'Figures').glob('*.pdf')}
    state = json.loads((args.run/'complete_figure_run.json').read_text())
    check(state['status'] == 'AUTOMATED_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED', 'Run has not passed QA')
    check([f['package'] for f in state['figures']] == list(BASELINES), 'Unexpected figure scope')
    baseline_runs = {}
    for spec in args.baseline_run:
        package, path = spec.split('=', 1)
        check(package in BASELINES and package not in baseline_runs, 'Invalid or duplicate baseline package')
        baseline_runs[package] = Path(path).resolve()
    if baseline_runs:
        check(set(baseline_runs) == set(BASELINES), 'Specify both baseline runs')
    manifest_path, fields = load_fields()
    selections, panel_checks, details_checks = [], {}, {}
    already_current = []
    for figure in state['figures']:
        package = figure['package']
        name, previous_id, expected_hash = BASELINES[package]
        previous = baseline_runs.get(package, data/'Figure_Run_Archive'/previous_id)
        if baseline_runs:
            prior_provenance = json.loads((previous/(name+'_provenance.json')).read_text())
            expected_hash = next(r['sha256'] for r in prior_provenance['outputs']
                                 if Path(r['path']) == previous/(name+'.pdf'))
        check(before[name+'.pdf'] == expected_hash == file_sha256(previous/(name+'.pdf')),
              'Published baseline changed: '+package)
        old = json.loads((previous/(name+'_provenance.json')).read_text())
        new = json.loads((args.run/(name+'_provenance.json')).read_text())
        page_is_current = before[name+'.pdf'] == file_sha256(args.run/(name+'.pdf'))
        if page_is_current:
            already_current.append(package)
        qa = json.loads((args.run/'QA'/package/'qa.json').read_text())
        check(not qa['errors'], 'Page QA errors: '+package)
        check(new['panel_recomposition']['status'] == 'PIXEL_IDENTICAL', 'Panel recomposition differs')
        prep = new['preparation']
        check(not any(prep[k] for k in ('model_inference_run', 'model_fitting_run', 'metric_tables_created')),
              'Unexpected scientific computation')
        a, b = old['preparation']['details'], prep['details']
        selections.append(b['selected_anatomical_fields'])
        old_command, new_command = old['prepare_command'], new['prepare_command']
        old_root = old_command[old_command.index('--output-dir')+1]
        new_root = new_command[new_command.index('--output-dir')+1]
        check(set(a) == set(b), 'Preparation detail keys changed: '+package)
        unchanged = []
        for key in a:
            if key == 'selected_anatomical_fields':
                continue
            left = normalized_run_paths(a[key], old_root)
            right = normalized_run_paths(b[key], new_root)
            if key in ('whole_map_2', 'whole_map_3'):
                left = {k:v for k,v in left.items() if k not in ('zoom_cells', 'bounds_xxyy_px')}
                right = {k:v for k,v in right.items() if k not in ('zoom_cells', 'bounds_xxyy_px')}
            if key == 'manual_vector_extraction':
                left = {k:v for k,v in left.items() if k != 'output'}
                right = {k:v for k,v in right.items() if k != 'output'}
            check(left == right, 'Unexpected detail change: '+package+'/'+key)
            unchanged.append(key)
        details_checks[package] = unchanged
        checks = {}
        selected_panel = 'D' if package == 'Figure_4' else ('E' if len(figure['final_panels']) == 7 else 'D')
        for panel in figure['final_panels']:
            letter = panel['panel']
            with Image.open(previous/f'panels/{package}/panel_{letter}_final.png') as left_img, \
                    Image.open(panel['png']) as right_img:
                left, right = left_img.convert('RGBA'), right_img.convert('RGBA')
                check(left.size == right.size, 'Changed panel canvas')
                difference = ImageChops.difference(left, right)
                bbox = difference.getbbox(alpha_only=False)
                if letter != selected_panel or page_is_current:
                    check(bbox is None, 'Changed unselected panel: '+package+'/'+letter)
                    checks[letter] = 'PIXEL_IDENTICAL'
                else:
                    check(bbox is not None, 'Selected field was not updated')
                    if package == 'Figure_4':
                        check(bbox[0] > left.width * 196/595.276, 'Original DMSO pixels changed')
                    elif selected_panel == 'E':
                        check(bbox[0] > left.width * 300/595.276, 'Original DMSO backprojection pixels changed')
                    checks[letter] = {'changed_bbox_px': bbox}
        panel_checks[package] = checks
        for item in prep['inputs'] + new['outputs']:
            check(file_sha256(item['path']) == item['sha256'], 'Changed input/output: '+item['path'])
        if page_is_current:
            check(a['selected_anatomical_fields'] == b['selected_anatomical_fields'],
                  'Already-current page has different field provenance')
    check(selections[0] == selections[1], 'Main and extended field selections differ')
    selection = selections[0]
    check(selection['manifest'] == str(manifest_path) and selection['same_main_and_extended'],
          'Incorrect shared manifest')
    check(selection['selected'] == {'DMSO':'Current_DMSO','RARi':args.expected_rari_id}, 'Incorrect author choice')
    check(selection['selected'] == fields['selection']['selected'], 'Choice differs from frozen author approval')
    check(selection['bounds_xxyy_px'] == fields['selection']['bounds_xxyy_px'], 'Different field bounds')
    report = {
        'status': 'DISPLAY_ONLY_CHECKS_PASSED',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, *sys.argv], 'code_sha256': file_sha256(__file__),
        'selected_fields': selection, 'panels': panel_checks, 'unchanged_details': details_checks,
        'baseline_runs': {package:str(path) for package,path in baseline_runs.items()},
        'already_current_packages': already_current,
        'original_DMSO_main_panel': 'PIXEL_IDENTICAL',
        'new_inference': False, 'new_model_fitting': False, 'new_statistics': False,
        'documents_modified': False,
    }
    if args.publish_visually_reviewed:
        report['visual_review'] = ('Complete PDFs and enlarged selected-field panels inspected: matching fields, '
                                  'undistorted microscopy, readable labels and scale bars; no clipping.')
        check(before == {p.name: file_sha256(p) for p in current.iterdir()},
              'Current figures changed during audit')
        changed_state = dict(state, figures=[f for f in state['figures']
                                             if f['package'] not in already_current])
        report['panel_publication'] = publish_panels(changed_state, data, overwrite=True)
        report['current_figures'] = refresh_current_figures(data, catalog['figures'])
        after = {p.name: file_sha256(p) for p in current.iterdir()}
        check(set(before) == set(after) and len(after) == 8, 'Current must contain eight PDFs')
        expected_changed = {BASELINES[f['package']][0]+'.pdf' for f in changed_state['figures']}
        check({k for k in before if before[k] != after[k]} == expected_changed,
              'Unexpected published figure changes')
        check(original == {p.name:file_sha256(p) for p in (data/'Figures').glob('*.pdf')},
              'Original assembled artwork changed')
        report.update(status='REVIEWED_AND_PUBLISHED', current_hashes_before=before, current_hashes_after=after)
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status':report['status'], 'report':str(report_path)}, indent=2))


if __name__ == '__main__':
    main()
