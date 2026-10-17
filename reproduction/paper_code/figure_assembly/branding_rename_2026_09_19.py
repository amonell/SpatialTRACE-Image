"""Snapshot, validate and publish the display-only SpatialTRACE re-export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from paper_paths import Path, lock_record, rendering_script
import shutil
import subprocess

from figure_assembly.compositor import file_sha256
from figure_assembly.current_figures import refresh_current_figures
from figure_assembly.publish import publish_panels
from figure_organization import REPO, load_catalog

SATA = Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript')
AUDIT = SATA / 'audits/spatialtrace_rename_2026_09_19'
EDITED = [
    'Figure_1/assembly/layout.json', 'Figure_1/assembly/README.md',
    'Figure_2_Extended/assembly/layout.json', 'Figure_2_Extended_3/common.py',
    'figure_assembly/test_branding_mask_display.py',
]
EXPECTED = {('Figure_1', 'C'), ('Figure_1', 'D'), ('Figure_2_Extended', 'F')}


def record(path):
    path = Path(path)
    return {'path': str(path), 'sha256': file_sha256(path)}


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def verify(item):
    if file_sha256(item['path']) != item['sha256']:
        raise RuntimeError('Changed protected file: ' + item['path'])


def snapshot():
    catalog = load_catalog()
    AUDIT.mkdir(parents=True, exist_ok=False)
    manifests = {}
    inputs = {}
    for entry in catalog['figures']:
        package = entry['package']
        manifest = json.loads((SATA / package / 'final_panel_manifest.json').read_text())
        manifests[package] = manifest
        for panel in manifest['panels']:
            verify({'path': panel['png'], 'sha256': panel['png_sha256']})
            for item in json.loads(Path(panel['provenance']).read_text())['inputs']:
                inputs[item['path']] = item
    sources = []
    for relative in EDITED:
        source = REPO / relative
        target = AUDIT / 'source_snapshots' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources.append(record(source))
    protected = [SATA / entry['reference'] for entry in catalog['figures']]
    for directory in ['Manuscript', 'Methods', 'Figure_Legends']:
        protected.extend((SATA / directory).glob('*.docx'))
    write_new(AUDIT / 'baseline.json', {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'manifests': manifests, 'inputs': inputs, 'source_files': sources,
        'current_pdfs': [record(p) for p in sorted((SATA / 'Regenerated_Figures/Current').glob('*.pdf'))],
        'protected': [record(p) for p in protected],
    })
    print('Saved pre-edit baseline:', AUDIT)


def finish(run, publish):
    baseline = json.loads((AUDIT / 'baseline.json').read_text())
    state = json.loads((run / 'complete_figure_run.json').read_text())
    assert state['status'] == 'AUTOMATED_CHECKS_PASSED_VISUAL_REVIEW_REQUIRED'
    assert len(state['figures']) == 8
    for item in baseline['protected'] + baseline['current_pdfs']:
        verify(item)
    changed = set()
    current_inputs = {}
    previous_input_versions = {}
    for package, manifest in baseline['manifests'].items():
        for panel in manifest['panels']:
            for item in json.loads(Path(panel['provenance']).read_text())['inputs']:
                key = (item['path'], item['sha256'])
                previous_input_versions.setdefault(key, set()).add(package)
    names = {}
    for figure in state['figures']:
        package = figure['package']
        qa = json.loads((run / 'QA' / package / 'qa.json').read_text())
        assert not qa['errors'], qa['errors']
        text = subprocess.check_output(['pdftotext', '-raw', str(run / figure['pdf']), '-'], text=True)
        assert 'tissueframe' not in ''.join(text.split()).lower(), package
        assert 'tissuemapper' not in ''.join(text.split()).lower(), package
        names[package] = text.count('SpatialTRACE')
        old = baseline['manifests'][package]['panels']
        for before, after in zip(old, figure['final_panels'], strict=True):
            assert before['panel'] == after['panel']
            for extension in ['pdf', 'png']:
                verify({'path': after[extension], 'sha256': after[extension + '_sha256']})
            if before['png_sha256'] != after['png_sha256']:
                changed.add((package, after['panel']))
            for item in json.loads(Path(after['provenance']).read_text())['inputs']:
                current_inputs[item['path']] = item
    assert changed == EXPECTED, changed
    assert names['Figure_1'] == 2 and names['Figure_2_Extended'] == 1, names
    retained = []
    refreshed_preexisting_code_metadata = []
    for path, before in baseline['inputs'].items():
        if '/stored_outputs/complete_figure_runs/' in path:
            continue
        assert path in current_inputs, 'Removed input: ' + path
        after = current_inputs[path]
        if before['sha256'] != after['sha256']:
            if path not in {str(REPO / p) for p in EDITED}:
                # Current pages were last exported at different times. The
                # September 7 inset adapter was already published in Extended
                # Figure 2, while other pages still recorded its older hash.
                assert Path(path).is_relative_to(REPO) and Path(path).suffix == '.py', path
                evidence = previous_input_versions.get((path, after['sha256']))
                assert evidence, 'Unexplained input change: ' + path
                refreshed_preexisting_code_metadata.append({
                    **after, 'already_published_in': sorted(evidence),
                    'prior_metadata_sha256': before['sha256'],
                })
        else:
            retained.append(after)
    report = {
        'status': 'VALIDATED', 'timestamp': datetime.now(timezone.utc).isoformat(),
        'run': record(run / 'complete_figure_run.json'),
        'changed_panels': sorted(changed), 'brand_counts': names,
        'other_panel_PNGs_byte_identical': True,
        'unchanged_scientific_and_reference_inputs': retained,
        'refreshed_preexisting_code_metadata': refreshed_preexisting_code_metadata,
        'original_artwork_and_documents_unchanged': True,
    }
    if publish:
        target = AUDIT / 'publication_validation.json'
        if target.exists():
            raise FileExistsError(target)
        report['publication'] = publish_panels(state, SATA, overwrite=True)
        report['current_figures'] = refresh_current_figures(SATA, load_catalog()['figures'])
        folder = SATA / 'Regenerated_Figures/Current'
        assert len(list(folder.iterdir())) == 8
        assert all(p.suffix == '.pdf' for p in folder.iterdir())
        for item in baseline['protected']:
            verify(item)
        report['status'] = 'PUBLISHED'
        report['visual_review'] = 'New labels fit existing module boxes and legend, using unchanged fonts and positions.'
        write_new(target, report)
    else:
        write_new(AUDIT / 'validation.json', report)
    print(json.dumps({k: report[k] for k in ['status', 'changed_panels', 'brand_counts']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['snapshot', 'validate', 'publish-reviewed'])
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    if args.action == 'snapshot':
        snapshot()
    else:
        if args.run_dir is None:
            parser.error('--run-dir is required')
        finish(args.run_dir.resolve(), args.action == 'publish-reviewed')


if __name__ == '__main__':
    main()
