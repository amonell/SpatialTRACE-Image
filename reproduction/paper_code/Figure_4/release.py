"""Read-only validation of the authorized scientific repair release."""
import json
from paper_paths import Path, lock_record, rendering_script
from Figure_4.common import require_file


def load_release(path):
    path=require_file(path);release=json.loads(path.read_text())
    if release.get('status')!='FROZEN_EVALUATED_RELEASE' or release.get('test_n')!=4843:
        raise RuntimeError('The repaired IF release is not complete')
    if release['test_membership_changed'] or release['test_reference_version']!='if_shared_training_p99_v1':
        raise RuntimeError('Unexpected test membership or reference release')
    for key in ['train_validation_crop_overlaps','test_training_crop_overlaps','test_validation_crop_overlaps']:
        if release[key]!=0:raise RuntimeError(f'Failed crop separation: {key}')
    require_file(release['checkpoint']['path'],release['checkpoint']['sha256'])
    require_file(release['lock']['path'],release['lock']['sha256'])
    return release


def release_file(path, release):
    requested=Path(path).resolve()
    records=list(release['outputs'].values())+release['inputs']+[release['checkpoint'],release['xenium_initialization'],release['test_identity_manifest'],release['lock']]
    matches=[item for item in records if Path(item['path']).resolve()==requested]
    if not matches or len({item['sha256'] for item in matches})!=1:
        raise RuntimeError(f'Input not unambiguously hash-locked in the repaired IF release: {path}')
    return require_file(requested,matches[0]['sha256'])
