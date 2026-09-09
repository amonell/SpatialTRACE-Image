"""Lightweight hash-locked release resolution; no model fitting or inference."""
from paper_paths import Path, lock_record, rendering_script
import hashlib
import json

RELEASE_POINTER=Path(__file__).with_name('active_release.json')
VARIANT_ORDER=('frozen_manual_sparse','full_manual_sparse','frozen_gat_soft_sparse','full_gat_soft_sparse')
SELECTED_VARIANT='full_manual_sparse'


def require_record(item):
    path=Path(item['path'])
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
    if digest.hexdigest()!=item['sha256']:raise RuntimeError('Classifier release input changed: '+str(path))
    return path


def load_release():
    pointer=json.loads(RELEASE_POINTER.read_text())
    if pointer['status']!='FROZEN_AND_VERIFIED':
        raise RuntimeError('Representation-pretrained Peyer classifier is not yet frozen; previous classifier fallback is forbidden.')
    path=require_record({'path':pointer['release_path'],'sha256':pointer['release_sha256']})
    release=json.loads(path.read_text())
    if (release['status']!='FROZEN_REPRESENTATION_PRETRAINED_PEYER_RELEASE'
            or release['selected_variant']!=SELECTED_VARIANT or release['threshold']!=.5
            or release['representation_checkpoint']['sha256']!='d4698fe67251bb56b805624a1e810cacc86f1a553b969ae4b23d54ba5ff3c23a'
            or release['architecture']['encoder_architecture']!='shared_scale_aware'
            or release['architecture']['local_readout']!='center_4x'
            or release['architecture']['context_readout']!='center_4x'):
        raise RuntimeError('The Peyer classifier no longer matches the approved representation/axis architecture')
    return path,release
