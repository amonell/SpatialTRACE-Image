"""Verify identities of intentionally undistributed Graph checkpoints.

This is metadata validation, not a substitute checkpoint or a binary hash check.
Scientific display tables and all other distributed inputs remain hash-locked.
"""
import json
from paper_paths import ROOT, Path

CHECKED = {}


def validate_graph_checkpoint_reference(path, expected_sha256):
    path = Path(path).resolve()
    manifest = json.loads((ROOT/'graph_checkpoint_lineage.json').read_text())
    if manifest.get('binaries_distributed') is not False:
        raise ValueError('Unexpected Graph checkpoint distribution policy')
    records = {str((ROOT/r['path']).resolve()): r for r in manifest['checkpoints']}
    record = records.get(str(path))
    if record is None or record['sha256'] != expected_sha256:
        raise ValueError(f'Graph checkpoint identity differs from frozen lineage: {path}')
    result = {**record, 'verification_scope': 'identity_only; binary not distributed or read'}
    CHECKED[record['path']] = result
    return result
