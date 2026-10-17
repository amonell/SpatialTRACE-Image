"""Capture and verify the scope of the 6 September branding/mask display edit."""
from paper_paths import Path, lock_record, rendering_script
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil

CODE = Path(__file__).resolve().parents[1]
SATA = Path('/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript')
AUDIT = SATA / 'audits/figure_branding_mask_2026_09_06'
CHANGED = {'Figure_1': {'C', 'D'}, 'Figure_2': {'A'}, 'Figure_3_Extended': {'A'}}
FILES = [
    'Figure_1/assembly/prepare.py', 'Figure_1/assembly/layout.json',
    'Figure_1/assembly/input_lock.json', 'Figure_1/assembly/README.md',
    'Figure_2/assembly/prepare.py', 'Figure_2/assembly/layout.json',
    'Figure_2/assembly/README.md', 'Figure_3_Extended/README.md',
    'Figure_3_Extended/panel_A/processing/README.md',
    'Figure_3_Extended/panel_A/figure_script/make_panel_A_paired_pretraining_views.py',
]


def record(path):
    path = Path(path)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def verify(item):
    assert record(item['path'])['sha256'] == item['sha256'], item['path']


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def baseline():
    AUDIT.mkdir(parents=True, exist_ok=False)
    files = []
    for rel in FILES:
        src = CODE / rel
        dst = AUDIT / 'code_before' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        files.append(record(dst))
    protected = [record(p) for p in sorted((SATA / 'Figures').glob('*.pdf'))]
    for folder in ['Methods', 'Figure_Legends', 'Manuscript']:
        docs = list((SATA / folder).glob('*.docx'))
        assert len(docs) == 1
        protected.append(record(docs[0]))
    figures = []
    for manifest in sorted(SATA.glob('Figure_*/final_panel_manifest.json')):
        m = json.loads(manifest.read_text())
        page = {'path': m['complete_page'], 'sha256': m['complete_page_sha256']}
        verify(page)
        panels = []
        for panel in m['panels']:
            png = {'path': panel['png'], 'sha256': panel['png_sha256']}
            verify(png)
            panels.append({'panel': panel['panel'], **png})
        figures.append({'package': m['package'], 'pdf': page, 'panels': panels})
    assert len(figures) == 8
    registry = CODE / 'frozen_preprocessing/frozen_model_registry.json'
    previous = SATA / 'audits/figure3_training_inputs_v1/code_before/frozen_preprocessing/frozen_model_registry.json'
    a, b = [json.loads(p.read_text()) for p in [previous, registry]]
    a['models']['xenium_image_coordinate_model'].pop('figure3_display_source')
    b['models']['xenium_image_coordinate_model'].pop('figure3_display_source')
    assert a == b
    protected.append(record(registry))
    lock = json.loads((CODE / 'Figure_1/assembly/input_lock.json').read_text())
    for path, item in lock['inputs'].items():
        if path == str(registry):
            assert item['sha256'] == record(previous)['sha256']
            continue
        verify({'path': path, **item})
        protected.append({'path': path, 'sha256': item['sha256']})
    mask = SATA / 'Figure_3_Extended/panel_A/panel_A_masked_patch_indices.tsv'
    if mask.exists():
        protected.append(record(mask))
    paired = SATA / 'Figure_3_Extended/stored_outputs/paired_view_repair_v1/manifest.json'
    paired_data = json.loads(paired.read_text())
    protected += [record(paired), record(paired_data['views']['path'])]
    write(AUDIT / 'baseline.json', {'created_at': datetime.now(timezone.utc).isoformat(),
          'figures': figures, 'protected_files': protected, 'archived_code': files,
          'registry_change_verified': 'Only main Figure 3 display-source metadata changed; all model checkpoints and other registry fields are identical.'})
    print('Baseline captured:', AUDIT)


def published():
    before = json.loads((AUDIT / 'baseline.json').read_text())
    for item in before['protected_files']:
        verify(item)
    checks = []
    for figure in before['figures']:
        package = figure['package']
        m = json.loads((SATA / package / 'final_panel_manifest.json').read_text())
        page = {'path': m['complete_page'], 'sha256': m['complete_page_sha256']}
        verify(page)
        if package not in CHANGED:
            verify(figure['pdf'])
        unchanged = []
        for old in figure['panels']:
            new = next(p for p in m['panels'] if p['panel'] == old['panel'])
            verify({'path': new['png'], 'sha256': new['png_sha256']})
            verify({'path': new['pdf'], 'sha256': new['pdf_sha256']})
            if old['panel'] not in CHANGED.get(package, set()):
                assert old['sha256'] == new['png_sha256'], (package, old['panel'])
                unchanged.append(old['panel'])
        checks.append({'package': package, 'current': page, 'unchanged_panel_PNGs': unchanged})
    current = SATA / 'Regenerated_Figures/Current'
    assert len(list(current.iterdir())) == 8 and all(p.suffix == '.pdf' for p in current.iterdir())
    assert list(current.parent.iterdir()) == [current]
    write(AUDIT / 'publication_checks.json', {'status': 'PASS', 'figures': checks,
          'protected_files_verified': len(before['protected_files']),
          'documents_and_original_artwork_unchanged': True,
          'unrequested_panels_pixel_identical': True, 'other_five_complete_figures_byte_identical': True,
          'followup': 'The Extended Data Figure 3 legend still says Black patches; changing it to Gray patches requires a document edit, outside this figure-only task.'})
    print('Publication scope checks PASS')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['baseline', 'published'])
    a = p.parse_args()
    baseline() if a.action == 'baseline' else published()
